"""Fail before installation if Windows MAX_PATH would break a bundled wheel."""
from pathlib import Path
import sys
import winreg
import zipfile

venv = Path(sys.argv[1]).resolve()
wheels = Path(sys.argv[2])
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
    enabled = winreg.QueryValueEx(key, "LongPathsEnabled")[0]
longest = ""
for wheel in wheels.glob("*.whl"):
    with zipfile.ZipFile(wheel) as archive:
        candidate = max(archive.namelist(), key=len)
        if len(candidate) > len(longest):
            longest = candidate
length = len(str(venv / "Lib/site-packages")) + 1 + len(longest)
if not enabled and length >= 259:
    raise SystemExit(
        f"Windows long paths are disabled; a wheel would require {length} characters. "
        "Copy the bundle to a shorter folder such as D:\\WS and rerun, "
        "or enable Windows long-path support and restart PowerShell. "
        "For testing, use -TestRoot D:\\WSOfflineTest."
    )
print(f"Windows path check passed (longest wheel path: {length} characters).")
