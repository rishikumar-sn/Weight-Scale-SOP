"""Cheap setup receipt checks using only the standard library, never AI imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def signature(root: Path, mode: str) -> dict:
    lock = root.parent / "offline/linux-cp310/requirements-lock.txt"
    inputs = [lock] if mode == "offline" else [root / "requirements.txt"]
    if mode == "online":
        frontend = root / "frontend"
        inputs += [frontend / "package.json", frontend / "package-lock.json", frontend / "index.html"]
        for directory in (frontend / "src", frontend / "public"):
            if directory.is_dir():
                inputs += sorted(p for p in directory.rglob("*") if p.is_file())
        inputs += sorted(frontend.glob("*config*.*"))
    config = root / ".venv/pyvenv.cfg"
    return {
        "version": 1, "mode": mode, "root": str(root.resolve()),
        "prefix": str(Path(sys.prefix).resolve()), "python": sys.version,
        "environment_created_ns": config.stat().st_mtime_ns,
        "inputs": {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(set(inputs))},
    }


def ready(root: Path, mode: str) -> bool:
    if not (root / "frontend/dist/index.html").is_file():
        return False
    try:
        expected = signature(root, mode)
        receipt = root / "data" / f"setup-{mode}.json"
        if receipt.exists():
            return json.loads(receipt.read_text()) == expected
        # Upgrade an already verified offline installation without repeating
        # the expensive smoke test. Bind the old report to this exact runtime
        # and require it to postdate both environment creation and the lock.
        if mode == "offline":
            report_path = root / "data/offline-verification.json"
            report = json.loads(report_path.read_text())
            lock = root.parent / "offline/linux-cp310/requirements-lock.txt"
            if (report.get("status") == "passed" and report.get("python") == sys.version
                    and report.get("runtime_cache") == str(Path(sys.prefix) / ".cache")
                    and report.get("ocr") and report.get("onnx_models_loaded")
                    and report.get("external_connections") == []
                    and report_path.stat().st_mtime_ns >= max(
                        lock.stat().st_mtime_ns, expected["environment_created_ns"])):
                write(root, mode)
                return True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return False


def write(root: Path, mode: str) -> None:
    receipt = root / "data" / f"setup-{mode}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(signature(root, mode), indent=2) + "\n")
    temporary.replace(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("online", "offline"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true")
    action.add_argument("--invalidate", action="store_true")
    args = parser.parse_args()
    if args.invalidate:
        receipt = ROOT / "data" / f"setup-{args.mode}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text('{"status": "incomplete"}\n')
    elif args.write:
        write(ROOT, args.mode)
    elif not ready(ROOT, args.mode):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
