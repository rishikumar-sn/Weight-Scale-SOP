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
