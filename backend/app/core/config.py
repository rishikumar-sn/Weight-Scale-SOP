from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SETTINGS_DIR = DATA_DIR / "settings"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
SETTINGS_PATH = SETTINGS_DIR / "app.json"
DATABASE_PATH = DATA_DIR / "app.db"
TESTBED_MODEL_PATH = PROJECT_ROOT / "models" / "segmentation" / "testbed.onnx"
FOREIGN_OBJECT_MODEL_PATH = PROJECT_ROOT / "models" / "segmentation" / "anomolies.onnx"

DEFAULT_SETTINGS: dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "rotation": 270,
        "preview_quality": 78,
        "preview_max_dimension": 960,
        "power_line_hz": 50,
        "shutter_denominator": 100,
        "driver_managed_exposure": os.name == "nt",
        "linux_image_controls": {},
        "software_anti_flicker": False,
        "anti_flicker_row_normalize": False,
        "anti_flicker_temporal_alpha": 0.08,
        "anti_flicker_gain_limit": 1.35,
        "anti_flicker_mode": "temporal_average" if os.name == "nt" else "driver",
        "temporal_average_frames": 90 if os.name == "nt" else 1,
        "adaptive": {
            "target_luma": 190,
            "max_average_frames": 6,
            "flicker_threshold": 0.02,
            "row_correction": True,
        },
    },
    "scale": {
        "model": "embsys_wm",
        "port": "COM12" if os.name == "nt" else "",
        "baud_rate": 9600,
    },
    "rois": {
        "processing": None,
        "apriltag": None,
    },
    "apriltag": {
        "family": "tag36h11",
        "id": 1,
        "width_mm": 10.0,
        "height_mm": 10.0,
    },
    "analysis": {
        "item_separation": {
            "min_area_ratio": 0.00045,
            "min_area_px": 350,
            "padding_px": 18,
        },
        "color_correction": {
            "saturation": 1.15,
            "contrast": 1.1,
            "brightness": 16.0,
        },
        "normalization": {
            "white_balance": True,
            "clahe_clip_limit": 1.8,
            "shadow_gamma": 0.92,
            "color_boost": 1.15,
            "green_recovery": 1.15,
            "dark_green_recovery": 1.2,
        },
        "background_calibration": None,
        "learned_stone_profiles": [],
    },
    "time": {
        "source": "network-synchronized system clock",
    },
}


def _merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def hardware_defaults(platform: str) -> dict[str, Any]:
    camera = deepcopy(DEFAULT_SETTINGS["camera"])
    windows = platform == "windows"
    camera.update({
        "driver_managed_exposure": windows,
        "anti_flicker_mode": "temporal_average" if windows else "driver",
        "temporal_average_frames": 90 if windows else 1,
        "linux_image_controls": {} if windows else {
            "brightness": 96, "contrast": 128, "saturation": 128,
            "gain": 0, "backlight_compensation": 0, "white_balance_automatic": 1,
        },
    })
    return {"camera": camera, "scale": {"model": "embsys_wm", "port": "COM12" if windows else "", "baud_rate": 9600}}


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH, *, platform: str | None = None) -> None:
        self.path = path
        self.platform = platform or ("windows" if os.name == "nt" else "linux")
        if self.platform not in {"windows", "linux"}:
            raise ValueError("Unsupported hardware platform")
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        loaded = self._read()
        if "hardware_profiles" not in loaded:
            # Preserve pre-profile camera/serial settings on their original OS.
            # Port names identify copied installations even on their first run.
            port = str(loaded.get("scale", {}).get("port", ""))
            legacy_platform = ("windows" if port.upper().startswith("COM") else
                               "linux" if port.startswith("/dev/") else self.platform)
            profiles = {name: hardware_defaults(name) for name in ("windows", "linux")}
            legacy = {key: loaded[key] for key in ("camera", "scale") if key in loaded}
            profiles[legacy_platform] = _merge(profiles[legacy_platform], legacy)
            loaded["hardware_profiles"] = profiles
            self._write(loaded)

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def _read(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._read()
            result = _merge(DEFAULT_SETTINGS, loaded)
            profiles = result.pop("hardware_profiles", {})
            active = _merge(hardware_defaults(self.platform), profiles.get(self.platform, {}))
            result.update(active)
            return result

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            loaded = self._read()
            hardware = {key: value for key, value in changes.items() if key in {"camera", "scale"}}
            shared = {key: value for key, value in changes.items()
                      if key not in {"camera", "scale", "hardware_profiles"}}
            updated = _merge(loaded, shared)
            profiles = updated.setdefault("hardware_profiles", {})
            active = _merge(hardware_defaults(self.platform), profiles.get(self.platform, {}))
            profiles[self.platform] = _merge(active, hardware)
            self._write(updated)
            return self.get()

    def select_scale(self, model: str) -> dict[str, Any]:
        if model not in {"embsys_wm", "essae_jx500"}:
            raise ValueError("Unsupported scale model")
        with self._lock:
            current = self.get()["scale"]
            devices = deepcopy(current.get("devices", {}))
            previous_model = current.get("model", "embsys_wm")
            devices[previous_model] = {key: current.get(key, "") for key in ("port", "baud_rate", "usb_identity")}
            selected = devices.get(model, {
                "port": "COM11" if self.platform == "windows" and model == "essae_jx500" else
                        "COM12" if self.platform == "windows" else "",
                "usb_identity": "",
            })
            return self.update({"scale": {
                **selected, "model": model, "devices": devices,
                "baud_rate": 115200 if model == "essae_jx500" else int(selected.get("baud_rate") or 9600),
                "usb_identity": selected.get("usb_identity", ""),
            }})

    def remember_scale_device(self, model: str, port: str, identity: str) -> None:
        with self._lock:
            if self.get()["scale"].get("model") == model:
                self.update({"scale": {"port": port, "usb_identity": identity}})

    def update_camera_rotation(self, rotation: int) -> dict[str, Any]:
        """Save orientation and keep saved areas on the same physical objects."""
        if rotation not in (90, -90):
            raise ValueError("Camera rotation must be +90 or -90 degrees")
        with self._lock:
            previous = self.get()
            rotation %= 360
            turns = ((rotation - int(previous["camera"]["rotation"])) % 360) // 90
            rois = {}
            for name, roi in previous["rois"].items():
                if roi is not None:
                    roi = dict(roi)
                    for _ in range(turns):
                        roi = {
                            "x": max(0.0, 1 - roi["y"] - roi["height"]),
                            "y": roi["x"],
                            "width": roi["height"], "height": roi["width"],
                        }
                rois[name] = roi
            return self.update({"camera": {"rotation": rotation}, "rois": rois})
