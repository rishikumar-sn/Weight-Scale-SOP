# CPU-only installation from a USB drive

On every backend start, the UI begins at **Capture image** with a fresh workflow. Unfinished captures from an earlier run are not resumed automatically. Saved captures/reports and camera/scale settings are retained. A browser left open also resets when it reconnects to a restarted backend.

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
The installer stages the bundled archives in a temporary APT cache because
`--no-download` also prevents APT from acquiring uncached local `.deb` files.
To check the offline package plan without sudo or changing the system, run
`bash scripts/install_ubuntu_system_offline.sh --check`.
If an older installer ended with `Unable to fetch some archives`, rerun this
updated installer; that error means system setup did not finish. The
`automatically installed and are no longer required` message is informational.
Then run:

```bash
# Install once (or repair/reverify the installation):
bash app.sh install

# Start the application every day:
bash app.sh start
```

`app.sh install` installs dependencies and runs the package/model/OCR checks, then
exits. `app.sh start` starts the installed backend and opens the UI; it does
not install packages or run installation smoke tests. The launcher automatically
selects the offline bundle, so no separate offline command is needed. Node.js and
npm are not needed for the bundled UI.

Starting the backend still loads the AI libraries and models it needs. Keep the
terminal open and wait for `Application startup complete`, then open
`http://127.0.0.1:8000`. Ctrl+C stops the backend. A traceback ending in
`KeyboardInterrupt` means it was interrupted.

`app.sh` is the only application launcher. Running `bash app.sh` without an
argument also starts the backend. The previous launcher scripts have been removed.

Camera and scale settings have separate Windows and Linux profiles, selected
automatically by the operating system. In the app, select the connected scale
model on each OS: ESSAE JX500 uses 115200 baud and `WEIGHT = ... g/ct` messages;
EMBSYS WM uses its saved baud rate (9600 by default) and numeric weight lines.
Both supported scale parsers run on Linux through USB serial ports. The current
Linux profile is configured for the verified ESSAE JX500 adapter at 115200 baud;
its saved USB identity follows it if the `/dev/ttyUSB` number changes.
Selecting a model restores its own saved port and baud rate. Grams enable
weight capture; carat readings are displayed without enabling gram capture.

The installer extracts a bundled CPython 3.10.19 runtime into `.offline-python`
and creates `.venv` from it. This avoids replacing Ubuntu 24.04's system Python
3.12. Optionally set `PYTHON_BIN=/absolute/path/to/python3.10` to use an existing
3.10 x64 installation. The online installation instructions about npm and
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

OCR loads `PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec` directly from
`offline/models/paddlex/official_models` inside the application folder. These
bundled weights are the portable model cache: copy them with the application.
Missing weights stop startup with a local-file error instead of a download.
Each device creates its own writable caches under its active venv's `.cache`
directory (PaddleX, Hugging Face, Torch and optional compiler caches). Paths are
resolved at launch, so no previous device's absolute paths are required.

`Model files already exist. Using cached files` in older launchers confirms
successful local model reuse. `Creating model` means loading the engine into
memory; each new backend process must do this. The optional `ccache` executable
speeds C/C++ extension compilation and is not needed by this deployment's
prebuilt CPU inference wheels. Paddle checks for it during import anyway. The
offline launcher filters only that specific missing-ccache warning; it does not
install ccache or hide other warnings. No compiler output needs to be prefilled
or shared across devices. The earlier bundle included inference dependencies
and model weights, but had not handled this optional development-tool warning.

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

## Anomaly detection for both scale models

`models/segmentation/anomolies.onnx` is shared by ESSAE JX500 and EMBSYS WM.
It loads during backend startup. The Linux launcher checks that the
anomaly and both testbed model files are present before starting the app.
For captures containing jewellery, the anomaly gate runs even when another
separated item is classified as Not Gold Jewelry. Existing Not Gold decisions
remain blocked and cannot be overwritten.

The anomaly model receives `capture/original.png`, the full raw camera frame;
only the model's normal resize and letterbox preprocessing is applied. Its
predicted segmentation masks are then intersected with the full-frame testbed
mask. For JX500, `essaetestbed.onnx` supplies the separate `circle` class; that
hole and its rim are excluded from the testbed mask before filtering anomaly
results. The raw input is never cropped or painted over to hide the hole.
The review display may crop the tray for readability; that display image is
never used as anomaly-model input.

To replay saved captures on CPU without acquiring camera/scale hardware or
changing the original sessions, supply a plastic-cover session and a clean
jewellery session (all other supplied sessions must also be clear):

```bash
.venv/bin/python scripts/verify_anomaly_sessions.py \
  --plastic-session PLASTIC_SESSION_ID --clean-session CLEAN_SESSION_ID \
  --report data/anomaly-replay.json
```

This re-runs the scale-specific testbed and anomaly models on every raw capture,
then runs the actual classifier and rejection/confirmation flow for the selected
two sessions in a temporary database. It checks mixed Not Gold/plastic rejection,
blocks rejected captures from analysis, and verifies that the first jewellery
rejection gate stops before anomaly inference.

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


## Combined bead detection

Bead analysis runs both `models/detection/bead_finder.onnx` (960 x 960) and
`models/detection/beadfinder_old.onnx` (640 x 640). Each model receives the
same original raw camera frame with its own letterbox resize. The cropped
testbed image is never used as bead-model input.

Boxes are mapped back to raw coordinates and accepted only when their centers
fall inside both the requested jewel bounds and the testbed mask. The accepted
boxes are merged by confidence before count, calibrated size, color analysis
and evidence rendering. Cross-model duplicate boxes use IoU 0.45; within each
model, the existing size-adaptive IoU thresholds remain 0.45/0.60/0.72. Both
models use the existing confidence threshold of 0.35. Result metadata lists
each model, its input size and its accepted candidate count before merging.
Both model files are required at startup. CPU inference now runs two models
for each analysed jewel. Restart the backend to load this change.

Run `.venv/bin/python scripts/verify_bead_ensemble.py` to check dual-resolution raw-image preprocessing, duplicate removal,
unique predictions from both models, region filtering and a saved-jewel replay.
Results are written to `../testing/bead-ensemble/`. The replay uses a temporary
repository and does not change saved captures. If saved captures are absent,
the script runs synthetic checks and both real ONNX sessions. Use
`--replay-session-dir /path/to/session` to replay another saved capture.
