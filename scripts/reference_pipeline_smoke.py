from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.domain.workflow import route_for_label
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.report_service import ReportService
from backend.app.storage.repository import CaptureRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the analysis pipeline on a completed reference session."
    )
    parser.add_argument("reference_session", type=Path)
    args = parser.parse_args()
    reference = args.reference_session.resolve()
    prior = json.loads((reference / "state.json").read_text(encoding="utf-8"))
    label = str(
        (prior.get("classification") or {}).get("confirmed_label") or "Necklace"
    )

    output_root = PROJECT_ROOT / "data" / "reference-smoke"
    repository = CaptureRepository(output_root / "app.db", output_root / "sessions")
    capture_id = "reference_" + reference.name
    capture_dir = repository.session_dir(capture_id) / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    source_dir = reference / "source"
    sources = {
        "working": source_dir / "working_source.png",
        "prepared": source_dir / "preprocessed.png",
        "mask": source_dir / "preprocessed_mask.png",
        "original": source_dir / "original_camera_capture.png",
        "evidence": source_dir / "original_camera_capture.png",
    }
    paths: dict[str, str] = {}
    for key, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = capture_dir / f"{key}{source.suffix}"
        shutil.copy2(source, target)
        paths[key] = str(target)

    prior_scale = (prior.get("source") or {}).get("stone_calibration") or {}
    scale_x = prior_scale.get("effective_horizontal_scale") or prior_scale.get(
        "horizontal_scale"
    )
    scale_y = prior_scale.get("effective_vertical_scale") or prior_scale.get(
        "vertical_scale"
    )
    now = datetime.now().astimezone().isoformat(timespec="milliseconds")
    state = {
        "id": capture_id,
        "status": "processing",
        "captured_at": now,
        "updated_at": now,
        "weight_g": (prior.get("weight_details") or {}).get("jewel_weight_g"),
        "paths": paths,
        "calibration": {
            "available": scale_x is not None and scale_y is not None,
            "mm_per_pixel_x": scale_x,
            "mm_per_pixel_y": scale_y,
        },
        "analysis_settings": {
            "color_correction": (prior.get("source") or {}).get("color_correction"),
            "background_calibration": (prior.get("source") or {}).get(
                "background_calibration"
            ),
            "normalization": (prior.get("source") or {}).get(
                "analysis_normalization"
            ),
            "learned_stone_profiles": (prior.get("source") or {}).get(
                "learned_stone_profiles"
            ),
        },
        "classification": {
            "predicted_label": label,
            "confirmed_label": label,
            "confirmed": True,
        },
        "route": route_for_label(label),
        "result": {},
        "job": {"status": "queued", "percent": 0},
    }
    repository.save(state)
    service = AnalysisService(repository)
    try:
        service._run(capture_id)
    finally:
        service.shutdown()
    completed = repository.get(capture_id) or {}
    if completed.get("status") != "complete":
        raise RuntimeError((completed.get("job") or {}).get("error") or "Pipeline failed")
    report = ReportService().generate(completed)
    report_path = repository.session_dir(capture_id) / "report.pdf"
    report_path.write_bytes(report.getvalue())
    print(json.dumps({
        "status": completed["status"],
        "label": label,
        "result_sections": sorted((completed.get("result") or {}).keys()),
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
