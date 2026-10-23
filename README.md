# Jewellery Capture

Cross-platform edge application for a rotated Full-HD USB camera, serial weighing
scale, jewellery analysis, and branded PDF reports. For Ubuntu deployment, see
the separate [Linux setup and operation guide](README_LINUX.md).

## Project layout

```text
assets/                 Company and client branding
backend/app/            FastAPI application
Classification/         SigLIP ONNX model, prompts, and correction gallery
Dimension/              Bangle and ring measurement
frontend/               React/Vite operator interface
models/detection/        Bead-detection ONNX model
models/packet/           Packet barcode label ONNX model
models/segmentation/     Testbed and foreign-object segmentation ONNX models
StoneDetection/         Stone detection and weight estimation
reference/hardware/      Original camera, scale, and shutter scripts
scripts/                 Setup, launch, and regression helpers
data/                    Runtime settings, captures, metadata, and reports
run.py                   Production entry point
```

## Foreign-object rejection

Jewellery captures first pass the existing per-item non-jewellery checks. When
every item passes, `models/segmentation/anomolies.onnx` checks the raw camera
image for objects mixed with jewellery. Only object segmentation pixels inside
the detected testbed polygon count. Confidence thresholds are `foreign_obj=0.45`,
`plastic_packet=0.35`, and `paper=0.25`; the `jewellery` label does not reject a
capture. Any qualifying foreign object blocks confirmation and analysis and
shows a recapture-only screen. Missing models or capture masks block the check
rather than allowing an unchecked capture through. The model runs on CPU and is
loaded once on its first eligible capture.

## First-time Windows setup

Open PowerShell in the project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

The script creates `.venv`, installs the Python packages, installs the frontend
packages, and builds the production frontend.

## Normal operation: one command

The production backend serves both the API and the built frontend:

```powershell
.\scripts\start_backend.ps1
```

Open `http://127.0.0.1:8000` on the PC. Other devices on the same network use
`http://<PC-IP-ADDRESS>:8000`.

To close the backend, use **Shut down** in the top-right corner of the app. It
finishes the current request and safely releases the camera, scale, and analysis
resources before the process exits. The button is disabled while a capture or
analysis is running. `Ctrl+C` in the backend terminal uses the same cleanup path.

Only one backend process may run because it owns the camera and COM port. Open
the camera's **Settings** icon for flicker correction, camera areas and rotation. The
testbed is detected automatically; the marker search and lighting metering areas
can be drawn and saved while the live preview remains visible.

Under **Camera rotation**, select **Turn right (+90°)** or **Turn left (−90°)**
and click **Apply rotation**. These orientations are relative to the original
camera image and keep the full-resolution output at 1080 × 1920. The live view
updates without reconnecting the camera, and saved camera areas turn with the
image. The last applied rotation is restored on the next app launch.

The live feed is processed in the backend. Windows initially keeps the existing
90-frame rolling average and driver-managed exposure. Select **Average · 90
frames**, or another correction method, and click **Apply method** to save it.
The selected method is restored after shutdown, backend restart or reboot; the
average warms up with new frames before capture is enabled. This means 90 frames,
approximately 3 seconds at 30 FPS, rather than a 90-second averaging window.
Preview JPEG encoding runs on a separate
thread and discards older pending frames. The camera header shows the preview
frame rate; `/api/health` reports it as `camera.stream_fps` alongside the camera
processing rate (`camera.processed_fps`). The live JPEG preview is limited to a
960 pixel longest side by default, while saved captures retain the camera's full
resolution.

Lighting setup runs once when a method is applied or the camera reconnects.
Adaptive methods keep the exposure/filter chosen during that setup; placing
jewellery or changing lighting does not trigger repeated capture lockouts.
Click **Apply method** again to recalibrate for different lighting. Fixed methods
apply their correction without measuring flicker on every frame. Capture still
requires a connected camera with a current frame and an available scale.

### Weight scale selection

Open **Settings > Weight scale**, select **EMBSYS WM** (the original CH340
scale) or **ESSAE JX500**, then click **Apply scale**. Applying saves the model
in the current OS hardware profile and reconnects the serial reader immediately.
Startup restores that selection. Switching models preserves each model's serial
settings; existing installations keep EMBSYS WM selected by default.

Applying a scale also loads its testbed ROI model and releases the previous
model: EMBSYS uses `models/segmentation/testbed.onnx`; ESSAE uses
`models/segmentation/essaetestbed.onnx`. Captures restore the matching model
after startup. ESSAE fills gaps inside the platform mask, trims its outer edge
by 12 pixels, and excludes the detected `Circle` with a 12-pixel margin.
Excluded pixels are white and ignored throughout jewellery and foreign-object
analysis. Keep the circular hole visible: if it cannot be detected, capture
asks for a recapture instead of processing the hole as jewellery.

ESSAE uses the existing Arduino/USB decoder output at 115200 baud, including
lines such as `1026 ms WEIGHT = 3.640 g RAW: ...`. It ignores startup/debug
lines, handles fragmented lines, and displays both `g` and `ct` with three
decimal places. Jewellery and tare capture accept only grams. Attempting a
capture in `ct` shows **ct capture is not allowed**; the backend also rejects it.
Carat values are never stored as gram weights.

ESSAE first tries COM11 on Windows, then probes available serial devices for
valid ESSAE messages. Linux probes USB serial devices such as `/dev/ttyUSB*`
and `/dev/ttyACM*`. After receiving a valid reading it saves the port and, when
available, the USB serial identity. It reconnects automatically after unplugging,
missing readings, and port-number changes. Close the standalone ESSAE display
or any serial monitor before using this app, since only one process can own
the port. The scale status exposes the active port, last raw reading, errors,
and discarded-frame count through `/api/health` and `/api/live`.

### Windows and Linux hardware settings

Run `scripts/start_backend.ps1` on Windows and `bash startbackend.sh` on Linux.
The backend selects the hardware profile using the operating system. Settings
are saved atomically in `data/settings/app.json`, under
`hardware_profiles.windows` and `hardware_profiles.linux`; each holds its own
`camera` and `scale` configuration. The original settings migrate automatically
on first startup, retaining the Windows COM port, averaging and camera settings.
Older top-level `camera`/`scale` entries are retained for reference; after migration,
edit the active profile or use the settings panel to change hardware settings.
Camera areas, analysis settings and AprilTag calibration remain shared.

Linux starts with the developed 50 Hz, fixed 1/100 s camera preset, brightness
96 and no software averaging. Its scale port defaults to automatic discovery;
set `hardware_profiles.linux.scale.port` to a persistent `/dev/serial/by-id/...`
path when required. Linux-only exact/adaptive shutter options appear only on
Linux. Windows keeps DirectShow with Media Foundation fallback and its existing
Logitech exposure settings. Changes on one OS do not reset the other OS's saved
method or serial port.

The developed UI, full-frame bead model input, and result-image layout are
included. Result images keep every jewellery row, growing vertically when needed
instead of cutting off entries after the available space is filled.

## Tare packet identifiers and replay

Tare captures use the same full-resolution processed camera frame as jewellery
captures, including the active Windows or Linux exposure, rotation, flicker
correction and averaging settings. Barcode verification is optional: the result
image shows the packet number, with a small confirmation only when the barcode
and printed packet ID match.

The packet scanner accepts the original 7- or 8-digit numbers and alphanumeric
IDs of 7 to 32 characters, including separators `-`, `/`, `.`, and `_` (for
example, `EKM9005L-00001659`). Barcode decoding, OCR and comparison use the same
validation. Letters, case, separators and leading zeros are preserved; OCR
spacing and typographic dashes are normalized. A match requires independent
barcode and OCR readings of the complete ID. Characters such as `O` and `0`
are never substituted to force a match.

Replay saved captures through the production detector and readers:

```powershell
.\.venv\Scripts\python.exe scripts\replay_packet_sessions.py "D:\path\to\session1" "D:\path\to\session2" --expect "EKM9005L-00001659"
```

`--expect` is optional and only checks the final results. Review images, input
crops, before/after JSON and timings are saved to `data/audits/packet-replay/`;
source session files are not modified. Deploy the updated backend files and
restart the backend on other capture devices to use the expanded ID support.

## Live classifier model trial

The main app now defaults to SigLIP2 **base-patch16-384** for the live trial.
Prompts, rejection rules, dimension fallback, gallery matching and manual
confirmation/learning keep the same behavior. Its local model/text/tokenizer
assets live in `Classification/siglip2-base-patch16-384/`; its gallery is
`Classification/jewelry_correction_gallery.base384.npz`. Both are independent
of the retained patch32 model and original gallery. Each gallery learns only
while its model is active; embeddings cannot be shared between these models.

The tested assets and gallery can be prepared from the local benchmark files:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_siglip384.py
```

This seeds the new gallery from confirmed historical item embeddings and
re-embeds the existing deterministic calibration split. It keeps a backup of
the original gallery and writes a migration report under
`data/classification-evaluation/base384_migration/`. Run it once and wait for
completion before starting the live trial. Restart the backend to load a model
change; no frontend rebuild is needed.

Start the default 384 trial:

```powershell
$env:JEWELLERY_CLASSIFIER_MODEL = "base384"
.\scripts\start_backend.ps1
```

To switch back to the original model and its original gallery, shut down the
backend and run:

```powershell
$env:JEWELLERY_CLASSIFIER_MODEL = "patch32"
.\scripts\start_backend.ps1
```

## Packet tare captures

The **Tare weight** tab supports pledge and release captures. Each capture uses the
camera frame and current scale reading, detects the packet label, reads the printed
7 or 8 digit packet number with PaddleOCR, reads its Code128 barcode with ZXing,
and compares the two. The saved result image places the packet number, barcode
status, tare weight, date, and time beside the photograph. A missing read or
mismatch is shown explicitly; it is never labeled as matched.

The backend loads the packet detector and OCR at startup. PaddleOCR may download
recognition models on the first startup, so allow network access then. Captures
reuse those loaded models. The original frame is retained alongside the result
image in the capture session.

Packet OCR is limited to four CPU threads by default so a tare scan leaves
capacity for the live camera. Set `PADDLE_PDX_CPU_NUM_THREADS` before starting
the backend to choose another limit.

## Frontend and backend separately for development

Terminal 1 — backend and API:

```powershell
.\scripts\start_backend.ps1
```

Terminal 2 — frontend development server:

```powershell
.\scripts\start_frontend.ps1
```

Open `http://127.0.0.1:5173`. Vite forwards API, video, logo, and WebSocket
requests to the backend on port `8000`.

After changing frontend code, rebuild the production UI with:

```powershell
npm run build --prefix frontend
```

## Manual commands

If PowerShell scripts are not desired:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
python run.py
```

## Linux setup

Follow [README_LINUX.md](README_LINUX.md) for system packages, model transfers,
and camera/scale permissions. Then install, build, and launch with:

```bash
bash startup.sh
```

Subsequent launches use `bash startbackend.sh`. The backend serves the built UI
and API together on port 8000.

The Linux camera backend uses V4L2. The scale service automatically searches
serial devices, including CH340 adapters. The edge PC must continue to own the USB
devices even when completed sessions are later synchronized to a cloud server.

## Runtime notes

Finger rings are capture-only: dimension, bead, and stone analysis are disabled,
including for queued captures carrying an older analysis route. Bangles retain
OD/ID measurement. Other measurement failures remain analysis warnings.

Result images enlarge small crops by at most 2x and leave space around the jewel.
Stone exports use the selected jewel mask to hide surrounding objects. Stone
markings are explained in a small neutral footer; dimension exports omit notes.

Stone refinement rejects neutral regions whose original pixels are predominantly
dark, saturated gold, including gold recesses with small glare highlights. This
prevents edge enclosure and low LAB chroma from turning warm metal shadows into
stones. Image analysis cannot guarantee zero false detections for every capture;
neutral cavities and black stones can have indistinguishable pixels.

- Default camera: index `0`, 1920x1080 at 30 FPS, displayed with 270-degree rotation.
- Default scale: `COM12` on Windows, auto-detection on Linux, at 9600 baud;
  CH340 devices are preferred during auto-detection.
- Inference always uses CPU. Requirements pin the CPU-only PyTorch wheel and
  use the CPU ONNX Runtime and PaddlePaddle packages; CUDA is not required.
- Runtime captures and reports are written under `data/sessions/`.
- Saved camera areas and device settings are stored in `data/settings/app.json`.
- One capture may contain multiple jewels. Keep a visible background gap between
  jewels so Otsu component separation can assign one box, class, and analysis
  result to each item. Bangles receive OD/ID analysis, finger rings skip analysis,
  and other jewel types receive their configured bead/stone analysis independently.

### Bead analysis

`models/detection/bead_finder.onnx` is an Ultralytics detector. The backend reads
its `imgsz` metadata and input tensor dimensions instead of assuming a fixed
resolution, and uses a `0.35` confidence threshold. Each accepted box is counted
directly; there is no secondary MobileNet verifier.

When AprilTag calibration is available, bead size is reported from the geometric
mean of the calibrated box width and height: tiny below 3 mm, small from 3 mm to
below 6 mm, and large from 6 mm. Color uses one HSV conversion and at most a
24x24 inner-ellipse sample per bead. The result also reports continuous/repetitive,
well-spaced, or mixed spacing based on box-size and nearest-neighbor consistency.

## Regression checks

```powershell
python -m pytest StoneDetection/test_stone_analysis_v2.py -q
python scripts/reference_pipeline_smoke.py "D:\path\to\runtime_sessions\session_name"
```

## Jewellery classifier calibration

Button/coin batteries are checked through dedicated foreign-object text prompts
on the natural camera crop before segmentation. A rejection requires similarity
of at least 0.11, a margin of at least 0.03 over every jewellery class, and gold
similarity below 0.06. This check does not use the correction gallery. Keep the
raw crop check: isolated masks can remove the solid battery face and resemble
a finger ring. Broad silver colour or a missing ring hole alone is insufficient
to reject jewellery.

Replay battery checks and inspect individual ring prompt similarities with
`python scripts/evaluate_button_battery.py`. Reports go to
`data/audits/battery_cell_20261013/`; sessions and corrections are not modified.
The eight reviewed battery captures are views of one test object; broader
reliability still needs independent batteries and jewellery samples.

Production choices are intentionally limited to Bangle, Bracelet, Chain / Necklace,
Mattal, Earrings / Nosepin, Finger Ring, Other Gold Jewellery, and Not Gold Jewelry.
Legacy labels are canonicalized when the correction gallery is loaded.

Build a deterministic 70/30 calibration/validation split, reject folder/filename
mismatches, add only calibration ROIs to the gallery, and write a held-out report:

```powershell
python scripts/calibrate_jewelry_classifier.py `
  --dataset-root "C:\path\to\UniversalDatasetCollector" `
  --output "data\classification-evaluation\calibrated_report.json" `
  --build-gallery `
  --historical-root "data\sessions"
```

Use `--disable-gallery` to measure SigLIP prompts alone. The script groups `_full`
and `_roi` files by capture ID, uses both as calibration views, and still counts
the pair as one sample. Validation uses ROI by default; pass `--image-kind full`
for the separate full-frame robustness check.
