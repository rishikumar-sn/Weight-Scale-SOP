# CPU-only installation from a USB drive

Keep these folders together when copying from USB to the target's local disk:

```
weightScaleinstall/
  offline/
    windows-cp310/wheels/          Windows x64 Python 3.10
    linux-cp310/wheels/            Linux x86_64 Python 3.10
    installers/                   Windows Python/VC runtime; Linux Python
    ubuntu-22.04-amd64/            Ubuntu 22.04 system packages
    ubuntu-24.04-amd64/            Ubuntu 24.04 system packages
  WEIGHT-INTEGRATION-APP/
    frontend/dist/                Built UI; no Node/npm installation needed
    Classification/               Local classifier weights and tokenizer
    models/                       CPU ONNX assets
    offline/models/paddlex/        Local OCR weights
  testing/                       Test launcher and verification reports
```

Copy the application and `offline` folder to a writable local disk before setup.
Use a short bundle path such as `D:\WS`; deeply nested Windows folders can exceed
the OS path limit when extracting PyTorch. Setup checks this before installation.
Do not run virtual environments from the FAT32 USB drive. Do not copy `.venv`
between devices or move it after installation. The existing online installers
remain available, but use the new scripts below for offline deployment.

## Windows 10/11 x64

1. Run `offline\installers\python-3.10.11-amd64.exe`. Install 64-bit Python with
   pip and the Python launcher. If Python 3.10 x64 already exists, reuse it.
2. Run `offline\installers\vc_redist.x64.exe` if the VC++ runtime is absent.
3. Open PowerShell in `WEIGHT-INTEGRATION-APP` and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows_offline.ps1 -VerifyModels
.\scripts\start_windows_offline.ps1
```

If the Python launcher is unavailable, pass the installed interpreter:

```powershell
.\scripts\setup_windows_offline.ps1 -PythonExe 'C:\Path\To\Python310\python.exe' -VerifyModels
```

Open `http://127.0.0.1:8000`. Use the app's Shut down button to close it.
If the CH340/CH341 scale adapter is not recognized, run the included
`offline\installers\CH341SER.EXE` driver installer from the manufacturer's
[official release](https://github.com/WCH-IC/download/releases/tag/CH341).
Other USB chipsets require their manufacturer's matching offline driver.

## Ubuntu 22.04/24.04 Intel/AMD x86_64

Use the normal operator account, not a root shell. Open a terminal in the copied
`WEIGHT-INTEGRATION-APP` folder:

```bash
bash scripts/install_ubuntu_system_offline.sh
```

This uses only the matching Ubuntu version's local `.deb` files. It requires sudo
for system libraries and camera/serial access. Log out and back in afterwards.
Then run:

```bash
bash startup_offline.sh --setup-only
bash startbackend_offline.sh
```

Or use `bash startup_offline.sh` to set up and start in one command.

The installer extracts a bundled CPython 3.10.19 runtime into `.offline-python`
and creates `.venv` from it. This avoids replacing Ubuntu 24.04's system Python
3.12. Optionally set `PYTHON_BIN=/absolute/path/to/python3.10` to use an existing
3.10 x64 installation. The original `startup.sh` instructions about npm and
rebuilding the frontend do not apply to this prebuilt offline deployment.

The `.deb` bundle contains official repository runtime packages and dependencies.
APT validates compatibility with the actual target. It is forbidden to download
missing dependencies and will stop if the target's installed packages conflict
with the bundle. Do not force downgrades; prepare an updated bundle for that
target if necessary. Linux hardware and native library execution must be checked
on Ubuntu; a Windows test cannot certify them.

## Models and network access

The offline launchers force the deployed base384 classifier, local PaddleX model
cache, Hugging Face offline mode, and CPU inference. External Python socket
connections are blocked. Local browser access and incoming LAN clients still work.
No Hugging Face account credentials or unrelated user caches are copied.

The verification command loads every deployed production ONNX model on CPU,
loads the local classifier processor/tokenizer, runs OCR on a generated image,
checks CPU package builds, and imports the backend without acquiring the camera
or scale. It fails if a model tries to contact an external service.

```powershell
.\.venv\Scripts\python.exe scripts\verify_offline_install.py --report data\offline-verification.json
```

```bash
.venv/bin/python scripts/verify_offline_install.py --report data/offline-verification.json
```

## Fresh offline Windows installation test

Disconnect internet if you want to repeat the test with the physical connection
removed. From the application folder:

```powershell
.\scripts\test_offline_windows.ps1
```

The test creates a separate `%TEMP%\ws-offline-test\.venv`, uses
`pip --no-index --no-cache-dir` with local wheels, and runs model verification
with external connections blocked. It never changes the working project venv.
For another fresh test, pass a new empty path:

```powershell
.\scripts\test_offline_windows.ps1 -TestRoot 'D:\offline-test-2'
```

Read `installation.log` and `verification.json` in the test directory. Successful
package/model checks do not replace a camera, scale, capture and PDF test on the
actual deployment hardware. Camera calibration/areas were copied as a starting
point; calibrate each physical setup.

## Contents and preparation

The clean copy excludes sessions, history database, caches, Git history,
node_modules, experiments, benchmarks, datasets, audits, logs and test output.
Settings, calibration, learned classifier galleries, model assets and application
source are retained. Source files in the original working project are not deleted.

Only these maintenance builders use internet. They live in the original
development checkout; run them there, not from the clean deployment or an
offline target:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_offline_bundle.py --destination E:\weightScaleinstall --step wheels
.\.venv\Scripts\python.exe scripts\prepare_offline_bundle.py --destination E:\weightScaleinstall --step runtimes
.\.venv\Scripts\python.exe scripts\prepare_ubuntu_debs.py --destination E:\weightScaleinstall
.\.venv\Scripts\python.exe scripts\prepare_offline_bundle.py --destination E:\weightScaleinstall --step copy
.\.venv\Scripts\python.exe scripts\prepare_offline_bundle.py --destination E:\weightScaleinstall --step manifest
```

Wheel locks snapshot the working Python 3.10 deployment and explicitly select
PyTorch/torchvision CPU wheels. Windows and Linux have separate locks. The bundle
manifest records SHA-256 hashes and sizes. Each Ubuntu package is checked against
the SHA-256 value in the official HTTPS repository package index. Preparation
logs and wheel dependency audit reports are included separately from app data.

Before first setup, you can verify a copied bundle using any Python 3.10+:

```powershell
py -3.10 scripts\verify_bundle_integrity.py
```

```bash
python3 scripts/verify_bundle_integrity.py
```

Run this before normal operation changes settings or learned galleries.
