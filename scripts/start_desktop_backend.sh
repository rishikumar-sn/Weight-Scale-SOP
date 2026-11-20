#!/usr/bin/env bash
# Keep startup errors visible when launched from the desktop.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd -- "$ROOT" || exit 1
bash app.sh
status=$?
if [[ "$status" -ne 0 && "$status" -ne 130 ]]; then
    printf '\nApplication exited with status %s. Press Enter to close this window.\n' "$status"
    read -r
fi
exit "$status"
