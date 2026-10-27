"""Verify bangle pose geometry and dispatch using saved captures, without hardware."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from offline_runtime import ROOT, configure


def main():
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis.bangle_orientation import detect_bangle_orientation
    from backend.app.domain.workflow import route_for_label
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository

    for angle in (0, 30, 90, 145):
        for axes, expected in (((110, 105), "flat"), ((110, 30), "standing")):
            mask = np.zeros((300, 300), np.uint8)
            cv2.ellipse(mask, (150, 150), axes, angle, 0, 360, 255, -1)
            inner = tuple(round(axis * 0.7) for axis in axes)
            cv2.ellipse(mask, (150, 150), inner, angle, 0, 360, 0, -1)
            evidence = detect_bangle_orientation(mask)
            assert evidence["orientation"] == expected, evidence

    solid = np.zeros((300, 300), np.uint8)
    cv2.circle(solid, (150, 150), 110, 255, -1)
    assert detect_bangle_orientation(solid)["orientation"] == "unknown"
    assert detect_bangle_orientation(None)["orientation"] == "unknown"
    assert detect_bangle_orientation(np.zeros((50, 50), np.uint8))["orientation"] == "unknown"
    gold = solid.copy()
    cv2.circle(gold, (150, 150), 80, 0, -1)
    assert detect_bangle_orientation(solid, gold_mask=gold)["orientation"] == "flat"
    # A circular stone elsewhere in an edge-on silhouette must not imply flat.
    edge = np.zeros((300, 300), np.uint8)
    cv2.rectangle(edge, (120, 30), (180, 270), 255, -1)
    assert detect_bangle_orientation(edge, gold_mask=gold)["orientation"] == "standing"
    for pose in ("standing", "unknown"):
        assert route_for_label("Bangle", bangle_orientation=pose)["stones"]
        assert not route_for_label("Bangle", bangle_orientation=pose)["dimension"]
    assert route_for_label("Finger Ring")["key"] == "capture_only"
    assert route_for_label("Chain Necklace")["beads"]

    def load(session):
        return json.loads((ROOT / "data/sessions" / session / "manifest.json").read_text())

    standing = load("20261026_140229_5679a8d5")
    flat = load("20261026_140111_d21bbb44")
    with tempfile.TemporaryDirectory(prefix="bangle-orientation-") as temp:
        repo = CaptureRepository(Path(temp) / "app.db", Path(temp) / "sessions")
        service = AnalysisService(repo)
        try:
            for session, pose in ((standing, "standing"), (flat, "flat")):
                item = copy.deepcopy(session["classification"]["items"][0])
                route = service._route_for_item(item, "Bangle")
                assert item["bangle_orientation"]["orientation"] == pose
                assert route["dimension"] == (pose == "flat")
                assert route["stones"] == (pose == "standing")
            item = copy.deepcopy(standing["classification"]["items"][0])
            assert service._run_dimension(standing, item)["applicable"] is False
            # Reconfirming an initially different prediction also checks pose.
            state = copy.deepcopy(standing)
            state.update(id="confirmation", status="awaiting_confirmation")
            state["classification"]["items"][0]["predicted_label"] = "Bracelet"
            repo.save(state)
            with patch.object(service, "_save_classified_evidence", return_value="preview.jpg"), \
                    patch.object(service._executor, "submit"):
                confirmed = service.confirm_and_start("confirmation", "Bangle", False)
            assert confirmed["route"]["stones"] and not confirmed["route"]["dimension"]

            # A mixed queued capture with old dimension-only bangle routes must
            # dispatch each item by its actual pose and persist the new policy.
            state = copy.deepcopy(standing)
            state.update(id="mixed", status="processing")
            items = [copy.deepcopy(s["classification"]["items"][0]) for s in (standing, flat)]
            for index, item in enumerate(items, 1):
                item["index"] = index
                item["confirmed_label"] = "Bangle"
                item["route"] = route_for_label("Bangle")
            state["classification"].update(items=items, count=2)
            repo.save(state)
            with patch.object(service, "_run_stones", return_value={"found": True}) as stones, \
                    patch.object(service, "_run_dimension", return_value={"applicable": True}) as dimensions:
                service._run("mixed")
            result = repo.get("mixed")
            assert result["status"] == "complete", result.get("job")
            assert stones.call_count == dimensions.call_count == 1
            assert stones.call_args.args[1]["index"] == 1
            assert dimensions.call_args.args[1]["index"] == 2
            assert result["route"]["stones"] and result["route"]["dimension"]
            assert [i["bangle_orientation"]["orientation"] for i in result["result"]["items"]] == ["standing", "flat"]
            assert result["classification"]["items"][0]["route"]["stones"]
        finally:
            service.shutdown()
    print("PASS: rotated poses, uncertain masks, saved standing/flat captures, confirmation and mixed queued routing")


if __name__ == "__main__":
    main()
