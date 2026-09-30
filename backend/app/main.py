from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .core.config import (
    DATABASE_PATH,
    FRONTEND_DIST,
    PROJECT_ROOT,
    SESSIONS_DIR,
    SettingsStore,
)
from .domain.schemas import ClassificationConfirmation, RoiSettings, TareCaptureRequest
from .domain.workflow import class_labels
from .hardware.camera import CameraService
from .hardware.scale import ScaleService
from .services.analysis_service import AnalysisService
from .services.capture_service import CaptureService
from .services.compression_service import ArtifactCompressionService
from .services.report_service import ReportService
from .storage.repository import CaptureRepository


settings_store = SettingsStore()
repository = CaptureRepository(DATABASE_PATH, SESSIONS_DIR)
camera = CameraService(settings_store.get)
scale = ScaleService(settings_store.get)
reports = ReportService()
artifact_compression = ArtifactCompressionService(repository)
captures = CaptureService(camera, scale, settings_store, repository, artifact_compression)
analysis = AnalysisService(repository, artifact_compression)
logger = logging.getLogger(__name__)


def _state(capture_id: str) -> dict[str, Any]:
    state = repository.get(capture_id)
    if not state:
        raise HTTPException(status_code=404, detail="Capture not found")
    return analysis.public_state(state)


def _artifact_urls(state: dict[str, Any]) -> dict[str, Any]:
    capture_id = state["id"]
    session_dir = repository.session_dir(capture_id).resolve()

    def url_for(path_value: Any) -> str | None:
        if not path_value:
            return None
        try:
            relative = Path(str(path_value)).resolve().relative_to(session_dir).as_posix()
        except (ValueError, OSError):
            return None
        return f"/api/captures/{capture_id}/artifacts/{relative}"

    result = state.get("result") or {}
    classification_items = {
        int(item.get("index", 0)): item
        for item in ((state.get("classification") or {}).get("items") or [])
        if int(item.get("index", 0)) > 0
    }
    item_media = []
    for item in result.get("items") or []:
        item_index = int(item.get("index", 0))
        classified_item = classification_items.get(item_index) or {}
        item_media.append({
            "index": item_index,
            "crop": url_for((classified_item.get("paths") or {}).get("working")),
            "dimensions": url_for((item.get("dimensions") or {}).get("image")),
            "beads": url_for((item.get("beads") or {}).get("image")),
            "stones": url_for((item.get("stones") or {}).get("image")),
        })
    return {
        "original": url_for((state.get("paths") or {}).get("original")),
        "evidence": url_for((state.get("paths") or {}).get("evidence")),
        "mask": url_for((state.get("paths") or {}).get("mask")),
        "dimensions": url_for((result.get("dimensions") or {}).get("image")),
        "beads": url_for((result.get("beads") or {}).get("image")),
        "stones": url_for((result.get("stones") or {}).get("image")),
        "items": item_media,
        "pdf": f"/api/captures/{capture_id}/report.pdf" if state.get("status") == "complete" else None,
        "result_image": f"/api/captures/{capture_id}/result.png" if state.get("status") == "complete" else None,
    }


def _public_capture(state: dict[str, Any]) -> dict[str, Any]:
    payload = analysis.public_state(state)
    payload["media"] = _artifact_urls(payload)
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        analysis.preload_models()
        try:
            captures.packet_scanner.preload()
            logger.info("Packet detector and OCR are ready")
        except Exception:
            logger.exception("Packet scanner could not be preloaded; tare capture will retry")
        camera.start()
        scale.start()
        yield
    finally:
        # Stop hardware producers before tearing down inference resources.
        # Each cleanup is independent so one device cannot prevent the others
        # from being released.
        for name, cleanup in (
            ("camera", camera.stop),
            ("scale", scale.stop),
            ("analysis", analysis.shutdown),
        ):
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001
                print(f"Could not cleanly stop {name}: {exc}")


app = FastAPI(title="Jewellery Capture", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "camera": camera.status(), "scale": scale.snapshot()}


@app.post("/api/shutdown", status_code=202)
def shutdown_backend(request: Request, background_tasks: BackgroundTasks):
    """Finish this response, then ask Uvicorn to shut down gracefully."""
    callback = getattr(request.app.state, "request_shutdown", None)
    if not callable(callback):
        raise HTTPException(
            status_code=503,
            detail="Graceful shutdown is available when the backend is started with run.py.",
        )
    if getattr(request.app.state, "shutdown_requested", False):
        return {"ok": True, "message": "Backend shutdown is already in progress."}
    request.app.state.shutdown_requested = True
    background_tasks.add_task(callback)
    return {"ok": True, "message": "Backend is shutting down safely."}


@app.get("/api/live")
def live_state():
    latest = repository.latest()
    return {
        "camera": camera.status(),
        "scale": scale.snapshot(),
        "latest": _public_capture(latest) if latest else None,
    }


@app.get("/api/video")
def video():
    return StreamingResponse(camera.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/camera/reconnect", status_code=202)
def reconnect_camera():
    return {"ok": True, "camera": camera.request_reconnect()}


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            latest = repository.latest()
            await websocket.send_json({
                "camera": camera.status(),
                "scale": scale.snapshot(),
                "latest": _public_capture(latest) if latest else None,
            })
            await asyncio.sleep(0.4)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.get("/api/settings")
def get_settings():
    return {
        "settings": settings_store.get(),
        "labels": class_labels(PROJECT_ROOT / "Classification" / "jewelry_prompts.json"),
    }


@app.put("/api/settings/rois")
def save_rois(payload: RoiSettings):
    return settings_store.update({"rois": payload.model_dump()})


@app.post("/api/captures")
def create_capture():
    try:
        return _public_capture(captures.capture())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/tare-captures")
def create_tare_capture(payload: TareCaptureRequest):
    try:
        return _public_capture(captures.capture_tare(payload.mode))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/captures")
def recent_captures(limit: int = 20):
    return repository.list_recent(limit)


@app.get("/api/captures/{capture_id}")
def get_capture(capture_id: str):
    return _public_capture(_state(capture_id))


@app.post("/api/captures/{capture_id}/item-type")
def check_item_type(capture_id: str):
    try:
        return _public_capture(analysis.classify(capture_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Capture not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"Item type check failed for {capture_id}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="The item type could not be checked. Please try again.",
        ) from exc


@app.post("/api/captures/{capture_id}/confirm")
def confirm_item_type(capture_id: str, payload: ClassificationConfirmation):
    allowed_labels = class_labels(
        PROJECT_ROOT / "Classification" / "jewelry_prompts.json"
    )
    requested_labels = [item.label for item in (payload.items or [])]
    if payload.label:
        requested_labels.append(payload.label)
    if not requested_labels or any(label not in allowed_labels for label in requested_labels):
        raise HTTPException(status_code=422, detail="Choose an item type from the list.")
    try:
        return _public_capture(analysis.confirm_and_start(
            capture_id,
            payload.label,
            payload.learn,
            [item.model_dump() for item in payload.items] if payload.items else None,
        ))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Capture not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/captures/{capture_id}/artifacts/{relative_path:path}")
def artifact(capture_id: str, relative_path: str):
    base = repository.session_dir(capture_id).resolve()
    target = (base / relative_path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/api/captures/{capture_id}/report.pdf")
def report(capture_id: str):
    state = _state(capture_id)
    if state.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Results are not ready")
    return StreamingResponse(
        reports.generate(state),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="jewellery_{capture_id}.pdf"',
            "Cache-Control": "no-store, max-age=0",
        },
    )


@app.get("/api/captures/{capture_id}/result.png")
def result_image(capture_id: str, download: bool = False):
    state = _state(capture_id)
    if state.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Results are not ready")
    disposition = "attachment" if download else "inline"
    if state.get("capture_type") == "tare":
        evidence = Path(str((state.get("paths") or {}).get("evidence") or ""))
        if not evidence.is_file():
            raise HTTPException(status_code=404, detail="Tare result image is unavailable")
        return FileResponse(
            evidence,
            media_type="image/png",
            headers={
                "Content-Disposition": f'{disposition}; filename="tare_{capture_id}.png"',
                "Cache-Control": "no-store, max-age=0",
            },
        )
    return StreamingResponse(
        reports.generate_result_image(state),
        media_type="image/png",
        headers={
            "Content-Disposition": f'{disposition}; filename="jewellery_{capture_id}.png"',
            "Cache-Control": "no-store, max-age=0",
        },
    )


@app.get("/brand/company")
def company_logo():
    return FileResponse(PROJECT_ROOT / "assets" / "branding" / "embsys_logo.png")


@app.get("/brand/client")
def client_logo():
    return FileResponse(
        PROJECT_ROOT / "assets" / "branding" / "logo.jpg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
