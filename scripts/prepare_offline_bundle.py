"""Online preparation only. Never called by the offline installers.

Run with the working deployment's Python 3.10 environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OMIT = {"__pycache__", ".pytest_cache", "node_modules", ".git", ".cache"}


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(destination)
            return
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def copy_clean(source, target):
    def ignore(directory, names):
        return [n for n in names if n in OMIT or n.endswith((".pyc", ".log", ".bak", ".tmp", ".tsbuildinfo"))]
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)


def pins(platform):
    excluded = {"pytest", "pluggy", "iniconfig", "torch", "torchvision", "pip", "setuptools", "wheel"}
    if platform == "linux":
        excluded |= {"comtypes", "pygrabber", "pyreadline3"}
    result = []
    for d in metadata.distributions():
        name = d.metadata["Name"]
        if name.lower().replace("_", "-") not in excluded:
            result.append(f"{name}=={d.version}")
    # Explicit local version selects official CPU wheels on both platforms.
    result += [f"{name}=={metadata.version(name).split('+')[0]}+cpu" for name in ["torch", "torchvision"]]
    return sorted(result, key=str.lower)


def wheels(target, platform):
    folder = target / "offline" / ("windows-cp310" if platform == "windows" else "linux-cp310")
    folder.mkdir(parents=True, exist_ok=True)
    requirements = pins(platform)
    (folder / "requirements-lock.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    logs = target / "preparation-logs"
    logs.mkdir(exist_ok=True)

    def download(pin):
        name = pin.split("==")[0].lower()
        cmd = [sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "--only-binary=:all:",
               "--no-deps", "--progress-bar", "off", "--dest", str(folder / "wheels")]
        if platform == "linux":
            cmd += ["--python-version", "310", "--implementation", "cp", "--abi", "cp310", "--abi", "abi3", "--abi", "none"]
            for tag in [*(f"manylinux_2_{n}_x86_64" for n in range(35, 16, -1)),
                        "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]:
                cmd += ["--platform", tag]
        if name in {"torch", "torchvision"}:
            cmd += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        else:
            cmd += ["--index-url", "https://pypi.org/simple"]
        proc = subprocess.run(cmd + [pin], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (logs / f"{platform}-{name}.log").write_text(proc.stdout, encoding="utf-8")
        print(f"{platform}: {pin}: {'OK' if proc.returncode == 0 else 'FAILED (see log)'}", flush=True)
        return None if proc.returncode == 0 else pin

    # Also provide bootstrap tools for bundled Python/venv.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        failures = [f for f in pool.map(download, requirements + ["pip==25.3", "setuptools==80.9.0", "wheel==0.45.1"]) if f]
    if failures:
        raise RuntimeError(f"Missing {platform} wheels: {failures}")


def copy_project(target):
    app = target / "WEIGHT-INTEGRATION-APP"
    app.mkdir(parents=True, exist_ok=True)
    for name in ["backend/app", "assets", "models", "Classification", "Dimension", "StoneDetection", "frontend"]:
        copy_clean(ROOT / name, app / name)
    for file in (app / "StoneDetection").glob("test_*.py"):
        file.unlink()
    for file in (app / "Classification").glob("*.pre_*"):
        file.unlink()
    # Only operational scripts; omit experiments, audits, benchmark helpers.
    scripts = ["setup_windows.ps1", "start_backend.ps1", "start_linux_backend.py", "restart_scale_windows.ps1",
               "repair_scale_windows.ps1", "stop_backend.ps1", "verify_cpu.py", "setup_windows_offline.ps1",
               "start_windows_offline.ps1", "offline_runtime.py", "verify_offline_install.py", "test_offline_windows.ps1",
               "install_ubuntu_system_offline.sh", "audit_offline_wheels.py", "check_windows_offline_paths.py",
               "verify_bundle_integrity.py"]
    (app / "scripts").mkdir(exist_ok=True)
    for name in scripts:
        shutil.copy2(ROOT / "scripts" / name, app / "scripts" / name)
    for name in ["run.py", "requirements.txt", "startup.sh", "startbackend.sh", "startup_offline.sh",
                 "startbackend_offline.sh", "README.md", "README_LINUX.md", "README_OFFLINE.md"]:
        shutil.copy2(ROOT / name, app / name)
    copy_clean(ROOT / "data/settings", app / "data/settings")
    # Portable OCR assets only: never copy user account caches or credentials.
    copy_clean(Path.home() / ".paddlex/official_models", app / "offline/models/paddlex/official_models")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--step", choices=["wheels", "runtimes", "copy", "manifest"], required=True)
    args = parser.parse_args()
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if args.step == "wheels":
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_cpu.py")], check=True)
        for platform in ["windows", "linux"]:
            wheels(target, platform)
    elif args.step == "runtimes":
        fetch("https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe", target / "offline/installers/python-3.10.11-amd64.exe")
        fetch("https://aka.ms/vs/17/release/vc_redist.x64.exe", target / "offline/installers/vc_redist.x64.exe")
        fetch("https://github.com/WCH-IC/download/releases/download/CH341/CH341SER.EXE", target / "offline/installers/CH341SER.EXE")
        fetch("https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.10.19%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
              target / "offline/installers/python-3.10.19-linux-x86_64.tar.gz")
    elif args.step == "copy":
        app = copy_project(target)
        # Installers resolve sibling offline directory; models live with app.
        print(f"Clean application copied to {app}")
    elif args.step == "manifest":
        entries = {}
        for base in [target / "offline", target / "WEIGHT-INTEGRATION-APP"]:
            for file in sorted(base.rglob("*")):
                if file.is_file() and not any(x in file.parts for x in [".venv", "__pycache__"]):
                    with file.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
                    if digest is None:
                        h = hashlib.sha256()
                        with file.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                h.update(chunk)
                        digest = h.hexdigest()
                    entries[file.relative_to(target).as_posix()] = {"bytes": file.stat().st_size, "sha256": digest}
        (target / "bundle-manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Manifest: {len(entries)} files")


if __name__ == "__main__":
    main()
