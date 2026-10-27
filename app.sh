#!/usr/bin/env bash
# One Linux entry point: install once; start for everyday operation.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$ROOT"
BUNDLE="$(dirname -- "$ROOT")/offline"
cd -- "$ROOT"

usage() {
    echo 'Usage: bash app.sh [start|install]'
    echo '  start    Start the installed backend and UI (default).'
    echo '  install  Install/repair dependencies and verify models, then exit.'
}
[[ $# -le 1 ]] || { usage >&2; exit 2; }
ACTION="${1:-start}"
case "$ACTION" in
    -h|--help|help) usage; exit 0 ;;
    start|install) ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$(uname -s)" == Linux ]] || { echo 'Use the PowerShell scripts on Windows.' >&2; exit 1; }
OFFLINE=false
[[ ! -f "$BUNDLE/linux-cp310/requirements-lock.txt" ]] || OFFLINE=true

install_online() {
    for command in "${PYTHON_BIN:-python3}" node npm flock v4l2-ctl; do
        if ! command -v "$command" >/dev/null 2>&1; then
            echo "Missing command: $command. Follow README_LINUX.md first." >&2
            exit 1
        fi
    done
    "${PYTHON_BIN:-python3}" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12 for this deployment"'
    node -e 'const [a,b] = process.versions.node.split(".").map(Number); if (!((a === 20 && b >= 19) || (a === 22 && b >= 12) || a > 22)) { console.error("Vite requires Node 20.19+ or 22.12+; use Node 22 LTS."); process.exit(1); }'

    if [[ -e .venv && ! -x .venv/bin/python ]]; then
        echo "The existing .venv is not a Linux environment. Move it aside and rerun; do not copy a Windows .venv." >&2
        exit 1
    fi
    if [[ ! -x .venv/bin/python ]]; then
        "${PYTHON_BIN:-python3}" -m venv .venv
    fi
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    .venv/bin/python scripts/setup_state.py --mode online --invalidate
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip check
    .venv/bin/python scripts/verify_cpu.py
    # npm ci installs native dependencies for this Linux host from the lockfile.
    npm ci --prefix frontend --include=dev
    npm run build --prefix frontend
    .venv/bin/python scripts/setup_state.py --mode online --write
}

install_offline() {
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Ubuntu x86_64'; exit 1; }
    source /etc/os-release
    [[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]] || { echo 'Supported: Ubuntu 22.04/24.04'; exit 1; }
    for command in flock v4l2-ctl; do
        command -v "$command" >/dev/null || { echo "Missing $command. Run bash scripts/install_ubuntu_system_offline.sh first; it must finish successfully."; exit 1; }
    done
    export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1
    PACKAGES="$BUNDLE/linux-cp310"
    [[ -f "$PACKAGES/requirements-lock.txt" && -f frontend/dist/index.html ]] || { echo 'Missing wheel lock or built frontend'; exit 1; }
    if [[ ! -x .offline-python/python/bin/python3 ]]; then
        mkdir -p .offline-python
        tar -xzf "$BUNDLE/installers/python-3.10.19-linux-x86_64.tar.gz" -C .offline-python
    fi
    PYTHON="${PYTHON_BIN:-$ROOT/.offline-python/python/bin/python3}"
    "$PYTHON" -c 'import sys,struct; assert sys.version_info[:2] == (3,10) and struct.calcsize("P") == 8'
    if [[ ! -x .venv/bin/python ]]; then
        [[ ! -e .venv ]] || { echo 'Existing .venv is not a Linux environment. Move it aside.'; exit 1; }
        "$PYTHON" -m venv .venv
    fi
    .venv/bin/python scripts/setup_state.py --mode offline --invalidate
    .venv/bin/python -m pip --isolated install --no-index --no-cache-dir --no-compile --only-binary=:all: --find-links "$PACKAGES/wheels" -r "$PACKAGES/requirements-lock.txt"
    .venv/bin/python -m pip --isolated check
    .venv/bin/python scripts/verify_offline_install.py --report data/offline-verification.json
    .venv/bin/python scripts/setup_state.py --mode offline --write
}

if [[ "$ACTION" == install ]]; then
    echo 'Installing and verifying the application. The backend will not start.'
    if $OFFLINE; then install_offline; else install_online; fi
    echo 'Installation complete. Start the application with: bash app.sh'
    exit 0
fi

[[ -x .venv/bin/python && -f frontend/dist/index.html ]] || {
    echo 'Application is not installed. Run: bash app.sh install' >&2
    exit 1
}
command -v flock >/dev/null || { echo 'Missing flock. Follow README_LINUX.md for system prerequisites.' >&2; exit 1; }
mkdir -p data
# Preserve the descriptor across exec so only one process owns the hardware.
exec 9>data/backend.lock
flock -n 9 || { echo 'Backend already running for this checkout.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
if $OFFLINE && [[ "$EUID" != 0 ]]; then
    case " $(id -nG) " in
        *" dialout "*) ;;
        *) echo 'Run scripts/install_ubuntu_system_offline.sh, then log out and back in for scale access.' >&2; exit 1 ;;
    esac
fi
.venv/bin/python - <<'PY'
from pathlib import Path
import os

required = [
    Path("models/detection/bead_finder.onnx"),
    Path("models/packet/barcodescanner.onnx"),
    Path("models/segmentation/testbed.onnx"),
    Path("models/segmentation/essaetestbed.onnx"),
    Path("models/segmentation/anomolies.onnx"),
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

echo 'Starting Jewellery Capture at http://127.0.0.1:8000 (Ctrl+C to stop).'
echo 'Loading the models needed by the backend; installation smoke tests are skipped.'
if $OFFLINE; then
    exec .venv/bin/python scripts/offline_runtime.py
fi
exec .venv/bin/python scripts/start_linux_backend.py
