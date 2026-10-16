"""Verify a deployment has CPU inference packages and no GPU dependencies."""
from importlib.metadata import distributions


def main() -> None:
    forbidden = []
    for distribution in distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if name.startswith(("nvidia-", "cuda-", "tensorrt", "triton")) or name in {
            "onnxruntime-gpu", "onnxruntime-directml", "paddlepaddle-gpu", "paddlepaddle-xpu",
        }:
            forbidden.append(name)
    if forbidden:
        raise SystemExit(
            "GPU packages found: " + ", ".join(sorted(forbidden))
            + ". Create a fresh virtual environment and install requirements.txt."
        )

    import torch
    import paddle
    import onnxruntime as ort

    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise SystemExit("PyTorch is a GPU build. Reinstall requirements.txt in a fresh environment.")
    if paddle.device.is_compiled_with_cuda():
        raise SystemExit("PaddlePaddle is a CUDA build. Install the paddlepaddle CPU package.")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("ONNX Runtime CPUExecutionProvider is unavailable.")
    print(f"CPU dependencies verified: torch {torch.__version__}, paddle {paddle.__version__}, ONNX Runtime {ort.__version__}")


if __name__ == "__main__":
    main()
