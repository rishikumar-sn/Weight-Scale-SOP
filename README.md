# Jewellery Capture

Windows-first edge application for a rotated Full-HD USB camera, serial weighing
scale, jewellery analysis, and branded PDF reports. The same backend is structured
to run on Linux later.

## Project layout

```text
assets/                 Company and client branding
backend/app/            FastAPI application
Classification/         SigLIP ONNX model, prompts, and correction gallery
Dimension/              Bangle and ring measurement
frontend/               React/Vite operator interface
models/detection/        Bead-detection ONNX model
models/packet/           Packet barcode label ONNX model
StoneDetection/         Stone detection and weight estimation
reference/hardware/      Original camera, scale, and shutter scripts
scripts/                 Setup, launch, and regression helpers
data/                    Runtime settings, captures, metadata, and reports
run.py                   Production entry point
```

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

Only one backend process may run because it owns the camera and COM port. On the
first run, open **Set camera areas** and save both the jewellery and marker areas.

The live feed is processed in the backend. Its temporal anti-flicker filter uses
the configured 90-frame rolling average. Preview JPEG encoding runs on a separate
thread and discards older pending frames. The camera header shows the preview
frame rate; `/api/health` reports it as `camera.stream_fps` alongside the camera
processing rate (`camera.processed_fps`). The live JPEG preview is limited to a
960 pixel longest side by default, while saved captures retain the camera's full
resolution.

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

Install Python 3.10+, Node.js, npm, and `v4l2-ctl`, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
python run.py
```

The Linux camera backend uses V4L2. The scale service automatically searches
serial devices, including CH340 adapters. The edge PC must continue to own the USB
devices even when completed sessions are later synchronized to a cloud server.

## Runtime notes

- Default camera: index `0`, 1920x1080 at 30 FPS, displayed with 270-degree rotation.
- Default scale: `COM12` at 9600 baud; CH340 devices are also auto-detected.
- ONNX uses CPU by default. A verified provider list can be supplied through
  `JEWELLERY_ONNX_PROVIDERS`.
- Runtime captures and reports are written under `data/sessions/`.
- Saved camera areas and device settings are stored in `data/settings/app.json`.
- One capture may contain multiple jewels. Keep a visible background gap between
  jewels so Otsu component separation can assign one box, class, and analysis
  result to each item. Bangles/rings receive OD/ID analysis; other jewel types
  receive their configured bead/stone analysis independently.

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
