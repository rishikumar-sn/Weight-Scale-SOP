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
        "driver_managed_exposure": True,
        "software_anti_flicker": False,
        "anti_flicker_row_normalize": False,
        "anti_flicker_temporal_alpha": 0.08,
        "anti_flicker_gain_limit": 1.35,
        "temporal_average_frames": 90,
    },
    "scale": {
        "port": "COM12",
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


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self.path = path
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(DEFAULT_SETTINGS)

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def get(self) -> dict[str, Any]:
        with self._lock:
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            return _merge(DEFAULT_SETTINGS, loaded)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            updated = _merge(self.get(), changes)
            self._write(updated)
            return deepcopy(updated)
