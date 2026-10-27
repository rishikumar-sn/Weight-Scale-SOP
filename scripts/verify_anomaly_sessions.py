"""Replay real raw captures and classification in temporary sessions, without hardware.

Example (session IDs are relative to --sessions):
  .venv/bin/python scripts/verify_anomaly_sessions.py \
    --plastic-session SESSION_ID --clean-session CLEAN_ID --report data/anomaly-replay.json
Original captures, manifests, database, and correction galleries are not modified.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from offline_runtime import configure, ROOT, BLOCKED_CONNECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=ROOT / "data/sessions")
    parser.add_argument("--plastic-session", required=True)
    parser.add_argument("--clean-session", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    from backend.app.analysis.testbed_segmentation import TestbedSegmenter
    from backend.app.core.config import FOREIGN_OBJECT_MODEL_PATH, TESTBED_MODEL_PATHS
    from unittest.mock import patch
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository

    detector = ForeignObjectDetector(FOREIGN_OBJECT_MODEL_PATH)
    segmenters = {}
    replays = []
    for manifest in sorted(args.sessions.glob("20*/manifest.json")):
        state = json.loads(manifest.read_text())
        model = state["settings_snapshot"]["scale"]["model"]
        if model not in segmenters:
            segmenters[model] = TestbedSegmenter(
                TESTBED_MODEL_PATHS[model], inset_px=12 if model == "essae_jx500" else 2,
                outer_inset_ratio=0.035 if model == "essae_jx500" else 0.0,
            )
        raw = cv2.imread(str(manifest.parent / "capture/original.png"))
        saved_mask = cv2.imread(str(manifest.parent / "capture/testbed_source_mask.png"), 0)
        # Re-run the selected scale's testbed model, including its circle class.
        testbed = segmenters[model].segment(raw)
        check = detector.detect(raw, testbed.source_mask)
        saved_check = detector.detect(raw, saved_mask)
        assert check["passed"] == saved_check["passed"], manifest.parent.name
        if state["id"] == args.plastic_session:
            assert any(d["label"] == "plastic_packet" for d in check["detections"]), check
        else:
            assert check["passed"], (state["id"], check)
        replays.append({"id": state["id"], "scale": model,
                        "fresh_mask_matches_saved": bool(np.array_equal(testbed.source_mask > 0, saved_mask > 0)),
                        "check": check})
        print(state["id"], "rejected" if not check["passed"] else "clear", flush=True)

    assert {args.plastic_session, args.clean_session} <= {row["id"] for row in replays}
    workflows = []
    with tempfile.TemporaryDirectory(prefix="anomaly-replay-") as directory:
        base = Path(directory)
        repository = CaptureRepository(base / "app.db", base / "sessions")
        service = AnalysisService(repository)
        service._foreign_objects = detector
        try:
            for capture_id, expected in ((args.plastic_session, "rejected"),
                                         (args.clean_session, "awaiting_confirmation")):
                original = args.sessions / capture_id
                target = repository.session_dir(capture_id)
                shutil.copytree(original / "capture", target / "capture")
                state = copy.deepcopy(json.loads((original / "manifest.json").read_text()))
                state["paths"] = {key: str(target / "capture" / Path(value).name)
                                  for key, value in state["paths"].items()}
                state["classification"] = {}
                state.pop("foreign_object_check", None)
                state.pop("rejection_reason", None)
                state["status"] = "captured"
                repository.save(state)
                result = service.classify(capture_id)
                assert result["status"] == expected, result["status"]
                if expected == "rejected":
                    assert any(item["predicted_label"] == "Not Gold Jewelry"
                               for item in result["classification"]["items"]), "Mixed-capture regression not exercised"
                    try:
                        service.confirm_and_start(capture_id, None, False)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("Rejected capture proceeded to analysis")
                evidence = cv2.imread(result["paths"]["evidence"])
                raw = cv2.imread(result["paths"]["original"])
                assert evidence is not None, "Review evidence was not saved"
                workflows.append({"id": capture_id, "status": result["status"],
                                  "labels": result["classification"]["labels"],
                                  "check": result["foreign_object_check"],
                                  "evidence_shape_with_panel": list(evidence.shape)})
                print("Full classification:", capture_id, result["status"], flush=True)
            # A fully rejected first gate must stop before the anomaly model.
            with patch.object(service._get_classifier(), "check_non_jewelry",
                              return_value=(True, "verification: existing rejection gate")):
                with patch.object(service, "_check_foreign_objects") as second_gate:
                    try:
                        service.classify(args.clean_session)
                    except RuntimeError as exc:
                        assert "No gold jewellery was detected" in str(exc), str(exc)
                    else:
                        raise AssertionError("Existing jewellery rejection was bypassed")
                    assert not second_gate.called, "Anomaly gate ran after first-gate rejection"
            print("First gate rejection stops before anomaly inference: passed", flush=True)
        finally:
            service.shutdown()
    assert not BLOCKED_CONNECTIONS, BLOCKED_CONNECTIONS
    report = {"status": "passed", "raw_capture_replays": replays,
              "first_gate_rejection_stops_before_anomaly": True,
              "full_classification_replays": workflows, "external_connections": BLOCKED_CONNECTIONS,
              "limits": "Saved captures replayed on CPU; no live camera or serial-scale test. All supplied non-plastic sessions are expected to be clear."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Anomaly replay verification passed:", args.report, flush=True)


if __name__ == "__main__":
    main()
