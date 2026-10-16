from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import pytest

from backend.app.packet_scanner.detector import Detector
from backend.app.packet_scanner.ocr_reader import OCRReader
from backend.app.services.analysis_service import AnalysisService
from scripts import verify_cpu


def test_gpu_environment_override_cannot_enable_cuda(monkeypatch):
    monkeypatch.setenv("JEWELLERY_ONNX_PROVIDERS", "CUDAExecutionProvider")
    assert AnalysisService._onnx_providers() == ["CPUExecutionProvider"]


def test_packet_detector_uses_cpu_even_when_cuda_is_available():
    session = Mock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.get_inputs.return_value = [SimpleNamespace(name="images", shape=[1, 3, 640, 640], type="tensor(float)")]
    session.get_outputs.return_value = []
    session.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map={})
    with patch("backend.app.packet_scanner.detector.ort.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), patch(
        "backend.app.packet_scanner.detector.ort.InferenceSession", return_value=session,
    ) as create:
        Detector("model.onnx")
    create.assert_called_once_with("model.onnx", providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("legacy", [False, True])
def test_ocr_explicitly_selects_cpu(monkeypatch, legacy):
    constructor = Mock(side_effect=[TypeError("old API"), object()] if legacy else None)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=constructor))
    OCRReader()
    assert constructor.call_args_list[0].kwargs["device"] == "cpu"
    if legacy:
        assert constructor.call_args.kwargs["use_gpu"] is False


def test_setup_verifier_rejects_leftover_nvidia_packages(monkeypatch):
    monkeypatch.setattr(verify_cpu, "distributions", lambda: [SimpleNamespace(metadata={"Name": "nvidia-cublas-cu12"})])
    with pytest.raises(SystemExit, match="GPU packages found: nvidia-cublas-cu12"):
        verify_cpu.main()
