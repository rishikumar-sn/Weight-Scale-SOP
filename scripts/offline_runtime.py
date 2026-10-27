"""Set bundled model paths and prohibit external network connections."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import socket
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_CONNECTIONS = []
OCR_MODEL_ROOT = ROOT / "offline/models/paddlex/official_models"
OCR_MODELS = ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec")


def configure():
    # Model weights travel with the bundle; writable caches belong to this
    # device's interpreter. Never copy a venv or native compiler cache to deploy.
    for name in OCR_MODELS:
        for filename in ("inference.json", "inference.pdiparams", "inference.yml"):
            path = OCR_MODEL_ROOT / name / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Bundled OCR model missing or empty: {path}. Restore it from the USB bundle.")
    cache_root = Path(sys.prefix) / ".cache" if sys.prefix != sys.base_prefix else ROOT / ".cache"
    caches = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "PADDLE_PDX_CACHE_HOME": cache_root / "paddlex",
        "PADDLE_EXTENSION_DIR": cache_root / "paddle_extension",
        "CCACHE_DIR": cache_root / "ccache",
    }
    for key, path in caches.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_UPDATE_CHECK": "1",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "WEIGHT_OFFLINE_OCR_MODEL_ROOT": str(OCR_MODEL_ROOT),
        "CUDA_VISIBLE_DEVICES": "", "JEWELLERY_CLASSIFIER_MODEL": "base384",
        "PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INDEX": "1",
    })
    # Prebuilt CPU wheels perform inference without compiling C++ extensions.
    # Paddle checks for this optional developer tool even during normal import.
    warnings.filterwarnings(
        "ignore", message=r"No ccache found\. Please be aware that recompiling all source files may be required\.",
        category=UserWarning, module=r"paddle\.utils\.cpp_extension\.extension_utils",
    )
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def permitted(address):
        # Keep local IPC available. Incoming browser connections are unaffected.
        return not isinstance(address, tuple) or address[0] in {"127.0.0.1", "::1", "localhost"}

    def connect(sock, address):
        if not permitted(address):
            BLOCKED_CONNECTIONS.append(str(address))
            raise OSError(f"Offline deployment blocked external connection: {address}")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if not permitted(address):
            BLOCKED_CONNECTIONS.append(str(address))
            raise OSError(f"Offline deployment blocked external connection: {address}")
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


if __name__ == "__main__":
    configure()
    os.chdir(ROOT)
    entry = sys.argv[1] if len(sys.argv) > 1 else "run.py"
    if entry != "run.py":
        raise SystemExit("Unsupported offline entry point")
    sys.argv = [str(ROOT / entry)]
    runpy.run_path(str(ROOT / entry), run_name="__main__")
