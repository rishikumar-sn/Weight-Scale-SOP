"""Verify copied deployment files against the USB bundle's SHA-256 manifest.

Run before first setup; application settings may change during normal operation.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
failures = []
for relative, expected in manifest.items():
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(f"Missing or invalid path: {relative}")
        continue
    if path.stat().st_size != expected["bytes"]:
        failures.append(f"Size mismatch: {relative}")
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["sha256"]:
        failures.append(f"Hash mismatch: {relative}")
if failures:
    raise SystemExit("\n".join(failures))
print(f"Integrity verified: {len(manifest)} deployment files.")
