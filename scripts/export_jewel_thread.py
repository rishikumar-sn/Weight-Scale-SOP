"""Development-only ONNX export; deployment needs only ONNX Runtime."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    import onnx
    import torch

    torch.set_num_threads(2)
    source = ROOT / "models/classification/jewel-thread-small.pt"
    target = source.with_suffix(".onnx")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint["model"].float().eval()
    size = int(checkpoint["train_args"]["imgsz"])
    model.model[-1].export = True
    torch.onnx.export(model, torch.zeros(1, 3, size, size), str(target),
                      input_names=["images"], output_names=["probabilities"],
                      opset_version=17, dynamo=False)
    exported = onnx.load(target)
    onnx.helper.set_model_props(exported, {
        "names": str(model.names), "task": "classify", "imgsz": str([size, size]),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preprocessing": "RGB; PIL bilinear shortest-edge resize; center crop; float32 / 255",
    })
    onnx.checker.check_model(exported)
    onnx.save(exported, target)
    metadata = {"source": source.name, "onnx": target.name, "imgsz": size,
                "names": model.names, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "torch_version": torch.__version__, "opset": 17}
    target.with_suffix(".onnx.infer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
