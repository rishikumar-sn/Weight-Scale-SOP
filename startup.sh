#!/usr/bin/env bash
# Install application dependencies, build the UI, and start the production app.
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--setup-only" ) ]]; then
    echo "Usage: bash startup.sh [--setup-only]" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This launcher is for Linux. On Windows use scripts/setup_windows.ps1." >&2
    exit 1
fi
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
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
# npm ci installs native dependencies for this Linux host from the lockfile.
npm ci --prefix frontend --include=dev
npm run build --prefix frontend
if [[ "${1:-}" == "--setup-only" ]]; then
    echo "Setup complete. Copy the model assets listed in README_LINUX.md, then run bash startbackend.sh."
    exit 0
fi
exec bash "$PROJECT_ROOT/startbackend.sh"
