"""Set bundled model paths and prohibit external network connections."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_CONNECTIONS = []


def configure():
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_UPDATE_CHECK": "1",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "PADDLE_PDX_CACHE_HOME": str(ROOT / "offline/models/paddlex"),
        "CUDA_VISIBLE_DEVICES": "", "JEWELLERY_CLASSIFIER_MODEL": "base384",
        "PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INDEX": "1",
    })
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
