"""Check dependency closure for the target OS using actual wheel metadata."""
import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def audit(folder, platform):
    environment = default_environment()
    environment.update({"python_version": "3.10", "python_full_version": "3.10.11",
                        "sys_platform": "win32" if platform == "windows" else "linux",
                        "os_name": "nt" if platform == "windows" else "posix",
                        "platform_system": "Windows" if platform == "windows" else "Linux",
                        "platform_machine": "AMD64" if platform == "windows" else "x86_64"})
    packages = {}
    for wheel in (folder / "wheels").glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            meta = BytesParser().parsebytes(archive.read(name))
        key = canonicalize_name(meta["Name"])
        assert not key.startswith(("nvidia-", "cuda-", "triton", "tensorrt")), wheel
        assert key not in {"onnxruntime-gpu", "paddlepaddle-gpu"}, wheel
        packages[key] = (meta, wheel.name)
    errors = []
    for line in (folder / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        req = Requirement(line)
        key = canonicalize_name(req.name)
        if key not in packages or Version(packages[key][0]["Version"]) not in req.specifier:
            errors.append(f"Missing pinned wheel: {line}")
    for key, (meta, filename) in packages.items():
        for raw in meta.get_all("Requires-Dist", []):
            req = Requirement(raw)
            extras = ["", "ocr-core"] if key == "paddlex" else [""]
            if req.marker and not any(req.marker.evaluate(dict(environment, extra=e)) for e in extras):
                continue
            dependency = canonicalize_name(req.name)
            if dependency not in packages:
                errors.append(f"{key} requires missing {req}")
            elif Version(packages[dependency][0]["Version"]) not in req.specifier:
                errors.append(f"{key} requires {req}, bundled {packages[dependency][0]['Version']}")
        if key in {"torch", "torchvision"} and "+cpu" not in meta["Version"]:
            errors.append(f"Non-CPU {key}: {meta['Version']}")
    report = {"platform": platform, "packages": len(packages), "errors": errors,
              "scope": "Static wheel dependency closure; Linux execution still requires Ubuntu testing."}
    (folder / "wheel-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--platform", choices=["windows", "linux"], required=True)
    args = parser.parse_args()
    audit(args.folder, args.platform)
