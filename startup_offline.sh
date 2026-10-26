#!/usr/bin/env bash
# Offline installer: never invokes apt update, pip indexes, npm, curl or wget.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname -- "$ROOT")/offline"
cd -- "$ROOT"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Ubuntu x86_64'; exit 1; }
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --setup-only ) ]] || { echo 'Usage: bash startup_offline.sh [--setup-only]'; exit 2; }
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]] || { echo 'Supported: Ubuntu 22.04/24.04'; exit 1; }
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
.venv/bin/python -m pip --isolated install --no-index --no-cache-dir --no-compile --only-binary=:all: --find-links "$PACKAGES/wheels" -r "$PACKAGES/requirements-lock.txt"
.venv/bin/python -m pip --isolated check
.venv/bin/python scripts/verify_offline_install.py --report data/offline-verification.json
for command in flock v4l2-ctl; do
    command -v "$command" >/dev/null || { echo "Missing $command. Run scripts/install_ubuntu_system_offline.sh first."; exit 1; }
done
echo 'Offline setup complete.'
[[ "${1:-}" == --setup-only ]] && exit 0
exec bash "$ROOT/startbackend_offline.sh"
