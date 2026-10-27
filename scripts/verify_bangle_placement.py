"""Replay all saved image sessions in temporary storage; never rewrite history."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from offline_runtime import ROOT, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-replay", action="store_true",
                        help="Recheck completed replay records without repeating model inference")
    args = parser.parse_args()
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis.bangle_placement import check_bangle_placement
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository

    output = ROOT.parent / "testing/bangle-placement-20261026"
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "data/sessions").glob("*/manifest.json"))
    protected = [ROOT / "data/app.db", *sources,
                 *(ROOT / "Classification").glob("*.json"),
                 *(ROOT / "Classification").glob("*.npz")]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

    # Single bangles at arbitrary rotations must remain allowed, including
    # a side view, a wide solid cuff, texture holes and an open contour.
    for angle in (0, 35, 90, 140):
        mask = np.zeros((400, 400), np.uint8)
        cv2.ellipse(mask, (200, 200), (130, 125), angle, 0, 360, 255, 15)
        assert check_bangle_placement(mask)["passed"]
        standing = np.zeros_like(mask)
        cv2.ellipse(standing, (200, 200), (130, 35), angle, 0, 360, 255, -1)
        assert check_bangle_placement(standing)["passed"]
        cuff = mask.copy()
        cv2.circle(cuff, (200, 70), 4, 0, -1)
        assert check_bangle_placement(cuff)["passed"]
        crossed = mask.copy()
        cv2.line(crossed, (200, 45), (200, 355), 255, 35)
        matrix = cv2.getRotationMatrix2D((200, 200), angle, 1)
        crossed = cv2.warpAffine(crossed, matrix, (400, 400), flags=cv2.INTER_NEAREST)
        assert not check_bangle_placement(crossed)["passed"]
    assert check_bangle_placement(None)["passed"]

    rows = json.loads((output / "replay.json").read_text()) if args.reuse_replay else []
    thumbs = []
    with tempfile.TemporaryDirectory(prefix="bangle-placement-replay-") as temp:
        repo = CaptureRepository(Path(temp) / "app.db", Path(temp) / "sessions")
        service = AnalysisService(repo)
        try:
            for manifest in ([] if args.reuse_replay else sources):
                old = json.loads(manifest.read_text())
                sid = manifest.parent.name
                if old.get("capture_type") == "tare":
                    rows.append({"id": sid, "skipped": "tare capture; no jewellery analysis"})
                    continue
                paths = old.get("paths") or {}
                required = ("original", "working", "mask", "testbed_source_mask")
                missing = [key for key in required if not paths.get(key) or not Path(paths[key]).is_file()]
                if missing:
                    rows.append({"id": sid, "skipped": "missing analysis inputs", "missing": missing})
                    continue
                state = copy.deepcopy(old)
                state.update(status="captured", classification={}, route={}, result={}, job={})
                for key in ("rejection_kind", "rejection_reason", "placement_check", "foreign_object_check"):
                    state.pop(key, None)
                (repo.session_dir(sid) / "capture").mkdir(parents=True, exist_ok=True)
                repo.save(state)
                current = service.classify(sid)
                classification = current["classification"]
                check = current.get("foreign_object_check") or {}
                row = {"id": sid, "saved_status": old["status"],
                       "saved_count": (old.get("classification") or {}).get("count"),
                       "status": current["status"], "count": classification["count"],
                       "labels": classification["labels"],
                       "rejection_kind": current.get("rejection_kind"),
                       "reason": current.get("rejection_reason"),
                       "geometry_check": current.get("placement_check"),
                       "model_placement_check": check.get("placement_check")}
                rows.append(row)
                print(sid, row["status"], row["count"], row["labels"], row["rejection_kind"], flush=True)
                if row["geometry_check"]:
                    cv2.imwrite(str(output / f"{sid}-rejected.jpg"), cv2.imread(current["paths"]["evidence"]))
                image = cv2.imread(paths["working"])
                thumb = cv2.resize(image, (220, 220))
                colour = (0, 0, 190) if row["status"] == "rejected" else (0, 100, 0)
                cv2.putText(thumb, sid[0:15], (3, 16), 0, .40, colour, 1)
                cv2.putText(thumb, f'{row["rejection_kind"] or "clear"}: {row["count"]}', (3, 33), 0, .45, colour, 1)
                thumbs.append(thumb)

            indexed = {row["id"]: row for row in rows}
            # Reviewed touching/overlapping bangle references, including the
            # screenshot and the earlier tilted flat+standing pair.
            for short in ("105852", "110056", "110154", "110240", "110308", "110436", "110519", "144908"):
                row = next(row for row in rows if row["id"].startswith("20261026_" + short))
                # Existing contamination rejections take priority over placement.
                assert row["status"] == "rejected" and row["rejection_kind"] in {"placement", "foreign_objects"}, row
            for short in ("105913", "105946", "135650", "140229", "144601", "144653", "144818"):
                row = next(row for row in rows if row["id"].startswith("20261026_" + short))
                assert row["status"] == "awaiting_confirmation", row
            for short in ("144653", "144818"):
                row = next(row for row in rows if row["id"].startswith("20261026_" + short))
                assert row["count"] == 2, row

            # Check the server gates independently of the early rejection.
            source = json.loads((ROOT / "data/sessions/20261026_144908_d34e1fca/manifest.json").read_text())
            state = copy.deepcopy(source)
            state.update(id="manual-confirmation", status="awaiting_confirmation")
            state.pop("rejection_reason", None)
            repo.save(state)
            with patch.object(service._executor, "submit") as submit:
                try:
                    service.confirm_and_start(state["id"], "Bangle", False)
                except RuntimeError as exc:
                    assert "overlapping" in str(exc)
                else:
                    raise AssertionError("Merged group was confirmed")
                submit.assert_not_called()
            assert repo.get(state["id"])["status"] == "rejected"
            state = copy.deepcopy(source)
            state.update(id="old-queue", status="processing")
            for item in state["classification"]["items"]:
                item["confirmed_label"] = "Bangle"
            repo.save(state)
            with patch.object(service, "_run_dimension") as dimension, patch.object(service, "_run_stones") as stones:
                service._run(state["id"])
                dimension.assert_not_called()
                stones.assert_not_called()
            assert repo.get(state["id"])["status"] == "rejected"
        finally:
            service.shutdown()
            (output / "replay.json").write_text(json.dumps(rows, indent=2))

    geometry_rows = []
    for manifest in sources:
        saved = json.loads(manifest.read_text())
        for item in (saved.get("classification") or {}).get("items", []):
            if (item.get("confirmed_label") or item.get("predicted_label")) != "Bangle":
                continue
            check = check_bangle_placement(cv2.imread(item["paths"]["mask"], 0))
            geometry_rows.append({"id": saved["id"], "item": item["index"], **check})
    flagged = {r["id"][9:15] for r in geometry_rows if not r["passed"]}
    assert flagged == {"105852", "110056", "110154", "110240", "110308", "110436", "110519", "144908"}, flagged
    (output / "saved-bangle-geometry.json").write_text(json.dumps(geometry_rows, indent=2))
    if args.reuse_replay:
        for row in rows:
            if "skipped" in row:
                continue
            source = json.loads((ROOT / "data/sessions" / row["id"] / "manifest.json").read_text())
            image = cv2.imread(source["paths"]["working"])
            thumb = cv2.resize(image, (220, 220))
            colour = (0, 0, 190) if row["status"] == "rejected" else (0, 100, 0)
            cv2.putText(thumb, row["id"][0:15], (3, 16), 0, .40, colour, 1)
            cv2.putText(thumb, f'{row["rejection_kind"] or "clear"}: {row["count"]}', (3, 33), 0, .45, colour, 1)
            thumbs.append(thumb)
    for p, digest in hashes.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f"Replay modified {p}"
    for start in range(0, len(thumbs), 24):
        page = thumbs[start:start + 24]
        while len(page) % 6:
            page.append(np.full((220, 220, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(page[i:i + 6]) for i in range(0, len(page), 6)])
        cv2.imwrite(str(output / f"contact-{start // 24 + 1}.jpg"), sheet)
    from collections import Counter
    summary = {"sessions": len(rows), "replayed": sum("skipped" not in r for r in rows),
               "skipped": [r for r in rows if "skipped" in r],
               "statuses": dict(Counter(r.get("status", "skipped") for r in rows)),
               "geometry_rejections": [r["id"] for r in rows if r.get("geometry_check")],
               "saved_bangle_components_checked": len(geometry_rows),
               "saved_bangle_groups_flagged": len(flagged),
               "protected_files_unchanged": True, "regression_checks": "passed",
               "limitations": ["Saved-session replay is not a labelled generalisation benchmark.",
                               "Hidden stacks, opaque heaps and broken contours may be missed.",
                               "Geometry rules apply only to components classified or confirmed as bangles."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
