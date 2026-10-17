# Jewellery Capture on Ubuntu

Deployment target: Ubuntu 22.04 or 24.04 LTS, 64-bit x86 PC, Python
3.10–3.12, and Node.js 22 LTS (22.12 or newer). The backend uses CPU inference
and V4L2 for the USB camera. ARM/Raspberry Pi needs separately verified wheels
and models. Linux hardware operation must be verified on the target machine.

This parent project now contains the Linux developments and Windows support.
Use `bash startbackend.sh` on Linux and `scripts/start_backend.ps1` on Windows.
Hardware settings persist separately in `data/settings/app.json` under
`hardware_profiles.linux` and `hardware_profiles.windows`. See the
[hardware settings guide](README.md#windows-and-linux-hardware-settings) before
editing a saved camera or serial setting; older top-level hardware entries are
retained as migration references, rather than the active configuration.

The Linux default retains the developed 50 Hz, manual 1/100 s shutter and tone
controls (brightness 96), with a single frame. Apply a flicker method in the
camera's Settings panel to save a different method across backend restarts.
Exact/adaptive shutter controls require Linux V4L2. The launcher prepares the
account's persistent `dialout` membership and activates it for the backend;
it may request sudo on first use. The serial port starts with automatic
discovery. Set `hardware_profiles.linux.scale.port` to a persistent
`/dev/serial/by-id/...` path if multiple USB serial devices are connected.

## 1. Install system dependencies

Run these commands on the Ubuntu PC:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential \
  git git-lfs curl ca-certificates v4l-utils util-linux \
  libgl1 libglib2.0-0 libgomp1 fonts-dejavu-core \
  libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0
```

Install Node.js 22 LTS from your preferred Node installer, then check:

```bash
python3 --version
node --version
npm --version
```

The locked Vite version requires Node 20.19+ or 22.12+; the older Node package
included with some Ubuntu releases is insufficient. The setup script checks
this before installation. The app imports OpenCV without opening a desktop
window; the listed shared libraries support its non-headless wheel. Keep one
OpenCV distribution in the environment to avoid conflicting `cv2` packages.

References: [PaddlePaddle Linux installation](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html)
and [Transformers installation](https://huggingface.co/docs/transformers/v5.15.0/en/installation).

### CPU-only dependencies

`requirements.txt` pins `torch==2.13.0+cpu` from the official
[PyTorch CPU wheel index](https://pytorch.org/get-started/previous-versions/).
The `+cpu` pin prevents pip selecting a CUDA build on Linux. `onnxruntime` and
`paddlepaddle` are the CPU packages. No NVIDIA driver, CUDA toolkit, cuDNN,
TensorRT, or GPU packages are needed. All application ONNX sessions and PaddleOCR
explicitly use CPU, even if GPU hardware is present.

Setup runs `scripts/verify_cpu.py` to reject GPU builds and NVIDIA dependencies.
For an existing environment with CUDA packages, stop the backend, move `.venv`
aside (for example `mv .venv .venv.previous`), and rerun `bash startup.sh` to
create a clean CPU environment. Installing requirements does not remove leftover
GPU dependencies from an old environment. Verify manually with:

```bash
.venv/bin/python scripts/verify_cpu.py
```

## 2. Transfer source and model assets

Clone the project or copy it to a directory owned by the deployment user, for
example `~/WEIGHT-INTEGRATION-APP`. Preserve the exact capitalization of paths
such as `Classification`, `Dimension`, and `StoneDetection`.

Do **not** copy Windows `.venv`, `frontend/node_modules`, or `frontend/dist`.
Linux must create its own virtual environment and native npm dependencies.
If a Windows `.venv` is present, move it aside before setup; the script rejects it.
On a Git clone, also run:

```bash
git lfs install
git lfs pull
```

A Git clone alone is insufficient: several production assets are ignored by Git.
Copy these from the working Windows deployment, preserving their paths:

| Path | Purpose |
| --- | --- |
| `Classification/siglip2-base-patch16-384/` (entire folder) | Default classifier: vision/text ONNX files, external `.onnx_data` weights, processor/tokenizer files, and embedding caches |
| `Classification/jewelry_correction_gallery.base384.npz` | Learned corrections for the default model; copy to retain learning |
| `models/detection/bead_finder.onnx` | Bead detection |
| `models/packet/barcodescanner.onnx` | Packet label detection |
| `models/segmentation/testbed.onnx` | Testbed segmentation |
| `assets/` and tracked `Classification/*.json` files | Branding and classifier prompts |
| `data/settings/app.json` | Optional existing camera areas, calibration, and settings |

Copy whole model folders because ONNX can reference external weight files.
The launcher detects missing required assets and Git LFS pointer files.
Model/processor compatibility is checked when the application loads the models.
`scripts/prepare_siglip384.py` is a development migration tool that depends on
local benchmark files and datasets; it is not the Linux deployment installer.

For the legacy classifier, copy
`Classification/siglip2-base-patch32-256_vision_encoder.sim.onnx`, its matching
embedding caches and `Classification/jewelry_correction_gallery.npz`, then
export `JEWELLERY_CLASSIFIER_MODEL=patch32` before launch. The legacy text model
may download from Hugging Face if not cached. Never mix the two models' galleries.

PaddleOCR downloads recognition assets on its first use. Allow network access
on the first startup and keep the deployment user's model caches. For offline
deployment, populate and validate those caches under the same Linux user first.

Existing Windows capture manifests contain absolute Windows artifact paths.
Copying `data/app.db` and `data/sessions/` alone does not migrate historical
image/report references to Linux; retain a backup and migrate those paths
separately if old captures must remain accessible. New Linux captures use Linux paths.

## 3. Give the deployment user access to USB devices

Run as the account that will operate the application:

```bash
sudo usermod -aG video,dialout "$USER"
```

Log out and back in for group membership to take effect, then inspect devices:

```bash
id
v4l2-ctl --list-devices
ls -l /dev/video* /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id/ 2>/dev/null
```

Stop the backend before editing `data/settings/app.json`. Set `camera.index`
to the number of the camera's capture device (`/dev/video0` means `0`); a camera
can expose multiple video nodes. Check supported formats with:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --list-ctrls
```

Prefer an explicit persistent scale port, for example:

```json
"scale": {
  "port": "/dev/serial/by-id/usb-YOUR_SCALE_ADAPTER_ID",
  "baud_rate": 9600
}
```

Use the actual symlink from the device listing. `/dev/ttyUSB0` and `/dev/ttyACM0`
also work but their numbers can change after reboot. An explicit Linux path is
retried on disconnect; the backend does not switch to another serial device.
An empty port enables automatic selection (CH340 preferred, otherwise the first
serial device). A copied `COM12` setting also falls back to detection on Linux.
Use an explicit path when multiple serial devices are connected.

The backend sets exposure through `v4l2-ctl`. Logitech Options+ settings from
Windows do not transfer. Unsupported camera controls are reported as `false` in
`camera.exposure.controls` at `/api/health`; verify brightness and flicker on the
target camera. Close other camera/serial applications before launch.

## 4. Set up and start

From the project directory:

```bash
chmod +x startup.sh startbackend.sh
./startup.sh
```

Alternatively use `bash startup.sh`. It creates `.venv`, installs the Python
requirements, checks dependency metadata, runs `npm ci`, builds the frontend,
and starts `run.py`. Installation/build failures stop the script. It does not
install system packages or change device permissions; complete steps 1–3 first.
Choose another supported Python with `PYTHON_BIN=python3.11 bash startup.sh`.
The chosen interpreter is used when creating the environment; an existing Linux
environment is reused.

To install/build without launching:

```bash
./startup.sh --setup-only
```

For everyday operation, no reinstall or rebuild is needed:

```bash
./startbackend.sh
```

Both scripts locate the project relative to themselves, so they also work when
called by absolute path from another directory. Run as the deployment user,
without `sudo`. Keep the project and model cache directories writable because
the app saves captures, settings, classifier caches and learned corrections.

Open **http://127.0.0.1:8000**. Other devices on the local network can use
`http://<UBUNTU-PC-IP>:8000` if the host firewall permits it. The backend serves
the compiled UI and API together; a separate frontend process is unnecessary.
Use this on a trusted local network; Internet exposure requires a separate
authenticated reverse-proxy deployment.

Use **Shut down** in the app or `Ctrl+C` in the terminal to release the camera
and scale safely. Keep one backend process; do not use Uvicorn reload or multiple
workers with shared USB devices. The launcher's lock prevents duplicate launches
through this script for the same checkout; it cannot guard manual `python run.py`
launches or another checkout.

After frontend changes, rerun `startup.sh` or build with
`npm run build --prefix frontend` before restarting the backend. Environment
options such as `JEWELLERY_CLASSIFIER_MODEL` can be exported before either
launcher. `JEWELLERY_ONNX_PROVIDERS` no longer overrides CPU inference.
The launchers do not automatically read `.env`.

## 5. Verify the target PC

```bash
curl --fail http://127.0.0.1:8000/api/health
```

`ok: true` means the API is alive; also inspect the camera and scale status/error
fields and confirm `scale.connected` is true. Open the UI, save the jewellery and
marker areas, verify AprilTag calibration, take a jewellery capture and a packet
tare capture, and open the generated result image and PDF. Confirm weight units,
camera orientation, dimensions, barcode/OCR results and report timestamps.
Set the Ubuntu timezone/time synchronization appropriately before production use.

Optional automated regression checks (no physical USB devices required):

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests backend/tests StoneDetection/test_stone_analysis_v2.py -q
```

If the camera/scale cannot open, check group membership, device permissions and
other programs using the device. If `cv2` cannot import, verify the shared
libraries in step 1. If native Python packages have no compatible wheel, confirm
the interpreter is 64-bit Python 3.10–3.12 on x86_64. This deployment has not been
validated for Ubuntu 26.04's default Python or ARM devices.
