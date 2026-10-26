#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"
[[ -x .venv/bin/python && -f frontend/dist/index.html ]] || { echo 'Run bash startup_offline.sh --setup-only first.'; exit 1; }
command -v flock >/dev/null || { echo 'Install offline Ubuntu system packages first.'; exit 1; }
mkdir -p data
exec 9>data/backend.lock
flock -n 9 || { echo 'Backend already running'; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
if [[ "$EUID" != 0 ]]; then
    case " $(id -nG) " in
        *" dialout "*) ;;
        *) echo 'Run scripts/install_ubuntu_system_offline.sh, then log out and back in for scale access.'; exit 1 ;;
    esac
fi
exec .venv/bin/python scripts/offline_runtime.py
