"""Verify ONNX, optional checkpoint parity, and recapture-only workflow gates."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-pt", action="store_true")
    args = parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image
    from backend.app.analysis.thread_classifier import JewelThreadClassifier
    from backend.app.services.analysis_service import AnalysisService, PredictionResult
    from backend.app.storage.repository import CaptureRepository

    gate = JewelThreadClassifier(ROOT / "models/classification/jewel-thread-small.onnx")
    pt = None
    if args.compare_pt:
        import torch
        torch.set_num_threads(2)
        pt = torch.load(gate.model_path.with_suffix(".pt"), map_location="cpu", weights_only=False)["model"].float().eval()
        pt.model[-1].export = True
    samples = []
    # The saved tassel failure, a complete necklace with thread, several chains
    # and rings, and blank captures. All saved inputs remain read-only.
    session_prefixes = ["20261030_", "20261028_143446", "20261028_143618",
                        "20261028_143746", "20261028_144729", "20261028_162305",
                        "20261024_111330", "20261024_111503"]
    for prefix in session_prefixes:
        samples.extend(sorted((ROOT / "data/sessions").glob(prefix + "*/items/*_raw.png")))
    assert samples, "Saved verification captures missing"
    results = []
    for path in samples:
        image = cv2.imread(str(path))
        result = gate.classify(image)
        result["source"] = str(path.relative_to(ROOT))
        if pt is not None:
            rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            reference_input = pt.transforms(rgb).numpy()[None]
            np.testing.assert_array_equal(gate.prepare(image), reference_input)
            with torch.inference_mode():
                reference = pt(torch.from_numpy(reference_input)).numpy().reshape(-1)
            actual = np.array(list(result["scores"].values()))
            np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=2e-4)
            assert result["label"] == gate.names[int(np.argmax(reference))]
            result["pt_max_absolute_error"] = float(np.max(np.abs(actual - reference)))
        if "20261030_144133" in str(path):
            assert result["label"] == "thread_only", result
        elif "20261028_143446" in str(path):
            # Both jewel labels allow this necklace. Record the model's
            # with/without-thread confusion rather than claiming class accuracy.
            assert result["passed"], result
            result["visual_note"] = "Necklace with thread; model predicts jewel_without_thread"
        elif "20261028_162305" in str(path):
            result["visual_note"] = "Visible gold ring falsely classified as thread_only by both PT and ONNX"
        elif any(prefix in str(path) for prefix in ["20261028_143746", "20261028_144729"]):
            assert result["label"] == "jewel_without_thread", result
        elif path.name == "item_01_raw.png" and any(prefix in str(path) for prefix in ["20261024_111330", "20261024_111503"]):
            assert result["label"] == "jewel_with_thread", result
        results.append(result)
        print(path.parent.parent.name, path.name, result["label"], round(result["confidence"], 6), flush=True)

    source = json.loads((ROOT / "data/sessions/20261030_144133_8c1dc894/manifest.json").read_text())
    image = cv2.imread(source["paths"]["working"])
    mask = cv2.imread(source["paths"]["mask"], cv2.IMREAD_GRAYSCALE)
    component = {"bbox": {"x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]},
                 "area_px": int(np.count_nonzero(mask)), "crop_bgr": image,
                 "raw_crop_bgr": image, "mask": (mask > 0).astype(np.uint8)}
    workflows = []
    output = ROOT.parent / "testing/jewel-thread-20261030"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jewel-thread-") as temporary:
        repo = CaptureRepository(Path(temporary) / "test.db", Path(temporary) / "sessions")
        service = AnalysisService(repo)
        try:
            # An anomaly must stop both classifiers even when a separated jewel
            # remains. Placement rejection has the same ordering requirement.
            for number, foreign_check in enumerate([
                {"passed": False, "detections": [{"label": "paper"}], "foreign_component_indices": [1]},
                {"passed": True, "detections": [], "placement_check": {"passed": False}},
            ]):
                state = copy.deepcopy(source)
                state.update(id=f"anomaly-first-{number}", status="captured")
                repo.save(state)
                with patch("backend.app.services.analysis_service.separate_jewellery_items", return_value=[component, component]), \
                        patch.object(service, "_check_foreign_objects", return_value=foreign_check), \
                        patch.object(service, "_get_jewel_thread") as thread_model, \
                        patch.object(service, "_get_classifier") as jewel_model:
                    rejected = service.classify(state["id"])
                    assert rejected["status"] == "rejected"
                    assert rejected["route"] == rejected["result"] == {}
                    thread_model.assert_not_called()
                    jewel_model.assert_not_called()
            state = copy.deepcopy(source)
            state.update(id="real-tassel-replay", status="captured")
            repo.save(state)
            service._jewel_thread = gate
            with patch.object(service, "_check_foreign_objects", return_value={"passed": True, "detections": []}), \
                    patch.object(service, "_get_classifier") as get_classifier:
                replay = service.classify(state["id"])
                assert replay["status"] == "rejected" and replay["rejection_kind"] == "thread_only", replay
                get_classifier.assert_not_called()
            shutil.copyfile(replay["paths"]["evidence"], output / "tassel-rejected.jpg")
            (output / "tassel-replay.json").write_text(json.dumps(replay, indent=2) + "\n")
            cases = [["thread_only"], ["jewel_with_thread"], ["jewel_without_thread"],
                     ["jewel_without_thread", "thread_only"]]
            for number, labels in enumerate(cases):
                state = copy.deepcopy(source)
                state.update(id=f"gate-{number}", status="captured")
                state.pop("thread_only_check", None)
                repo.save(state)
                (repo.session_dir(state["id"]) / "capture").mkdir(exist_ok=True)
                classifier = Mock()
                classifier.text_model_id = "verification"
                classifier.check_non_jewelry.return_value = (False, "")
                picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                classifier.classify_image.return_value = PredictionResult(
                    label="Other Gold Jewellery", confidence=.99, scores=[], image_path="",
                    original_image=picture, cropped_image=picture, is_gold_jewelry=True)
                checks = [{"label": label, "passed": label != "thread_only", "confidence": .99} for label in labels]
                with patch("backend.app.services.analysis_service.separate_jewellery_items", return_value=[component] * len(labels)), \
                        patch.object(service, "_check_foreign_objects", return_value={"passed": True, "detections": []}), \
                        patch.object(service, "_get_jewel_thread", return_value=SimpleNamespace(classify=Mock(side_effect=checks))), \
                        patch.object(service, "_get_classifier", return_value=classifier) as get_classifier:
                    classified = service.classify(state["id"])
                rejected = "thread_only" in labels
                assert classified["status"] == ("rejected" if rejected else "awaiting_confirmation")
                resulting_status = classified["status"]
                assert bool(classified["thread_only_check"]["passed"]) != rejected
                for checked in classified["thread_only_check"]["items"]:
                    assert all(Path(path).is_file() for path in checked["paths"].values())
                assert Path(classified["paths"]["evidence"]).is_file()
                with patch.object(service._executor, "submit") as submit:
                    if rejected:
                        get_classifier.assert_not_called()
                        assert classified["route"] == classified["result"] == {}
                        for override, save_only in [(False, False), (True, False), (True, True)]:
                            try:
                                service.confirm_and_start(state["id"], "Chain / Necklace", True,
                                                          override_not_gold=override, save_only=save_only)
                            except RuntimeError as exc:
                                assert "recapture" in str(exc).lower()
                            else:
                                raise AssertionError("Thread-only rejection was bypassed")
                        # A status-only change must not bypass the persisted gate.
                        classified["status"] = "awaiting_confirmation"
                        repo.save(classified)
                        try:
                            service.confirm_and_start(state["id"], "Chain / Necklace", False)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError("Persisted thread check was bypassed")
                        with patch.object(service, "_run_stones") as stones:
                            service._run(state["id"])
                            stones.assert_not_called()
                        submit.assert_not_called()
                    else:
                        confirmed = service.confirm_and_start(state["id"], "Other Gold Jewellery", False)
                        assert confirmed["status"] == "processing"
                        submit.assert_called_once()
                    classifier.learn_correction.assert_not_called()
                workflows.append({"labels": labels, "status": resulting_status, "passed": True})
        finally:
            service.shutdown()
    result_name = "verification.json" if args.compare_pt else "verification-onnx.json"
    (output / result_name).write_text(json.dumps({"samples": results, "workflows": workflows,
        "checkpoint_parity": args.compare_pt, "status": "passed",
        "model_accuracy_status": "known_failures",
        "limitations": ["Seven saved gold rings are falsely labeled thread_only by both PT and ONNX",
                        "One necklace with thread is labeled jewel_without_thread by both models",
                        "Blank captures can receive jewel labels; existing validation remains required"]}, indent=2) + "\n")
    print(f"PASS: {len(results)} ONNX captures, 4 workflow cases, API override and queued-job guards")


if __name__ == "__main__":
    main()
