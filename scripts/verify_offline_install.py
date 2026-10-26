"""Smoke test CPU libraries, deployed models and OCR without external connections."""
from pathlib import Path
import argparse
import json
import os

from offline_runtime import configure, ROOT, BLOCKED_CONNECTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    configure()
    os.chdir(ROOT)
    from verify_cpu import main as verify_cpu
    verify_cpu()
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import serial
    from PyQt5 import QtCore
    import reportlab
    import zxingcpp
    from transformers import SiglipImageProcessor, Siglip2Tokenizer
    import torch
    # Real CPU operations, not just import checks.
    assert (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8
    assert hasattr(cv2, "aruco"), "OpenCV contrib/AprilTag support missing"
    model_root = ROOT / "Classification/siglip2-base-patch16-384"
    SiglipImageProcessor.from_pretrained(str(model_root), local_files_only=True)
    Siglip2Tokenizer.from_pretrained(str(model_root), local_files_only=True)
    paths = ["models/detection/bead_finder.onnx", "models/packet/barcodescanner.onnx",
             "models/segmentation/testbed.onnx", "models/segmentation/essaetestbed.onnx",
             "models/segmentation/anomolies.onnx", "Classification/siglip2-base-patch16-384/vision_model.onnx",
             "Classification/siglip2-base-patch16-384/text_model.onnx"]
    for relative in paths:
        options = ort.SessionOptions()
        if relative.endswith("text_model.onnx"):
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(ROOT / relative), sess_options=options, providers=["CPUExecutionProvider"])
        print("ONNX loaded on CPU:", relative, flush=True)
        del session
    from Classification.siglip384_classifier import Siglip384Classifier
    classifier = Siglip384Classifier()
    sample = np.full((384, 384, 3), 255, dtype=np.uint8)
    cv2.circle(sample, (192, 192), 100, (180, 140, 40), 16)
    prediction = classifier.classify_image(Image.fromarray(sample))
    print("Local classifier inference:", prediction.label, flush=True)
    del classifier
    from backend.app.packet_scanner.ocr_reader import OCRReader
    reader = OCRReader()
    image = np.full((180, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "EKM9005L-00001659", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
    results = list(reader._items(image))
    print("Bundled OCR inference:", results, flush=True)
    assert results, "OCR produced no text on smoke-test image"
    # Verify the entire backend import graph, without acquiring camera/scale.
    from backend.app.main import app
    assert app is not None
    assert not BLOCKED_CONNECTIONS, f"Unexpected connection attempts: {BLOCKED_CONNECTIONS}"
    report = {"status": "passed", "python": __import__("sys").version,
              "external_connections": BLOCKED_CONNECTIONS, "ocr": results, "onnx_models_loaded": paths,
              "limits": "Camera, serial scale, complete captures and Linux system libraries require target-device testing."}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Offline installation smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
