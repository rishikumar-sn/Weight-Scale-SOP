"""Check placement rejection before learning, confirmation and queued analysis."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from offline_runtime import ROOT, configure


def main():
    configure()
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository

    source = json.loads((ROOT / "data/sessions/20261026_144908_d34e1fca/manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="bangle-placement-gates-") as directory:
        repo = CaptureRepository(Path(directory) / "app.db", Path(directory) / "sessions")
        service = AnalysisService(repo)
        try:
            # Explicitly selecting Bangle must not learn a composite crop,
            # including when the correction action immediately recaptures.
            for save_only in (False, True):
                state = copy.deepcopy(source)
                state.update(id=f"confirmation-{save_only}", status="awaiting_confirmation")
                item = state["classification"]["items"][0]
                item["predicted_label"] = "Other Gold Jewellery"
                repo.save(state)
                with patch.object(service._executor, "submit") as submit, \
                        patch.object(service, "_get_classifier") as classifier:
                    try:
                        service.confirm_and_start(state["id"], "Bangle", True, save_only=save_only)
                    except RuntimeError as exc:
                        assert "overlapping" in str(exc)
                    else:
                        raise AssertionError("Composite was confirmed or learned")
                    submit.assert_not_called()
                    classifier.assert_not_called()
                rejected = repo.get(state["id"])
                assert rejected["status"] == "rejected"
                assert rejected["classification"]["count"] == 0
                assert rejected["placement_check"]["affected_items"][0]["item_index"] == 1
                assert rejected["route"] == rejected["result"] == {}
            state = copy.deepcopy(source)
            state.update(id="queued", status="processing")
            for item in state["classification"]["items"]:
                item["confirmed_label"] = "Bangle"
            repo.save(state)
            with patch.object(service, "_run_dimension") as dimensions, patch.object(service, "_run_stones") as stones:
                service._run(state["id"])
                dimensions.assert_not_called()
                stones.assert_not_called()
            assert repo.get(state["id"])["status"] == "rejected"
            item = {"placement_check": {"passed": False}}
            assert service._check_bangle_placement(item, "Chain / Necklace")["passed"]
            assert "placement_check" not in item
        finally:
            service.shutdown()
    print("PASS: classification evidence, manual confirmations, recapture corrections, queued jobs and non-bangle isolation")


if __name__ == "__main__":
    main()
