#!/usr/bin/env bash
# Start one hardware-owning backend; it also serves the production frontend.
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This launcher is for Linux. On Windows use scripts/start_backend.ps1." >&2
    exit 1
fi
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
    echo "Linux environment or built frontend missing. Run bash startup.sh first." >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "Missing flock. Install util-linux as described in README_LINUX.md." >&2
    exit 1
fi
mkdir -p data
# The descriptor remains open across exec and is released when Python exits.
exec 9>data/backend.lock
if ! flock -n 9; then
    echo "A backend launched by startbackend.sh is already running for this checkout." >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

.venv/bin/python - <<'PY'
from pathlib import Path
import os

required = [
    Path("models/detection/bead_finder.onnx"),
    Path("models/packet/barcodescanner.onnx"),
    Path("models/segmentation/testbed.onnx"),
]
model = os.environ.get("JEWELLERY_CLASSIFIER_MODEL", "base384").strip().lower()
if model == "base384":
    root = Path("Classification/siglip2-base-patch16-384")
    required += [root / name for name in (
        "vision_model.onnx", "vision_model.onnx_data", "text_model.onnx",
        "text_model.onnx_data", "config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json",
    )]
elif model == "patch32":
    required.append(Path("Classification/siglip2-base-patch32-256_vision_encoder.sim.onnx"))
else:
    raise SystemExit("JEWELLERY_CLASSIFIER_MODEL must be base384 or patch32")
errors = []
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"Missing or empty asset: {path}")
    else:
        with path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                errors.append(f"Git LFS pointer instead of model data: {path}")
if errors:
    raise SystemExit("\n".join(errors) + "\nCopy the deployed model assets; see README_LINUX.md.")
PY

echo "Starting Jewellery Capture at http://127.0.0.1:8000 (Ctrl+C to stop)."
# Keep run.py's graceful shutdown callback and normal lifespan cleanup.
exec .venv/bin/python scripts/start_linux_backend.py
