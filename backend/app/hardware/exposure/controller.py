from __future__ import annotations

import math
import os
import subprocess
from typing import Any

import cv2


class ExposureController:
    """Best-effort cross-platform anti-flicker camera settings."""

    def apply(
        self,
        cap: cv2.VideoCapture,
        camera_index: int,
        power_line_hz: int,
        shutter_denominator: int,
        driver_managed: bool = False,
    ) -> dict[str, Any]:
        if os.name == "nt":
            return self._apply_windows(cap, power_line_hz, shutter_denominator, driver_managed)
        return self._apply_linux(camera_index, power_line_hz, shutter_denominator)

    def _apply_windows(
        self,
        cap: cv2.VideoCapture,
        power_line_hz: int,
        shutter_denominator: int,
        driver_managed: bool,
    ) -> dict[str, Any]:
        if driver_managed:
            # Logitech Options+ persists the BRIO anti-flicker and exposure choices.
            # Do not write OpenCV exposure properties here: doing so switches the
            # camera back to DirectShow manual exposure and overrides those choices.
            return {
                "platform": "windows",
                "mode": "logitech-driver",
                "power_line_hz": power_line_hz,
                "driver_auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "driver_exposure_value": cap.get(cv2.CAP_PROP_EXPOSURE),
                "note": "Exposure is managed by Logitech Options+; select PAL 50 Hz there.",
            }

        # DirectShow exposes exposure as approximately log2(seconds).
        target = int(round(math.log2(1.0 / max(1, shutter_denominator))))
        auto_ok = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25))
        exposure_ok = bool(cap.set(cv2.CAP_PROP_EXPOSURE, float(target)))
        return {
            "platform": "windows",
            "mode": "manual",
            "requested_shutter": f"1/{shutter_denominator}",
            "driver_exposure_value": target,
            "auto_exposure_disabled": auto_ok,
            "exposure_sent": exposure_ok,
            "note": "Power-line frequency remains controlled by the Logitech Windows driver.",
        }

    def _apply_linux(
        self,
        camera_index: int,
        power_line_hz: int,
        shutter_denominator: int,
    ) -> dict[str, Any]:
        device = f"/dev/video{camera_index}"
        aligned = max(power_line_hz, round(shutter_denominator / power_line_hz) * power_line_hz)
        exposure_absolute = max(1, int(round(10000 / aligned)))
        controls = {
            "power_line_frequency": 1 if power_line_hz == 50 else 2,
            "exposure_auto": 1,
            "exposure_absolute": exposure_absolute,
        }
        applied: dict[str, bool] = {}
        for name, value in controls.items():
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device, "-c", f"{name}={value}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                applied[name] = True
            except (OSError, subprocess.CalledProcessError):
                applied[name] = False
        return {
            "platform": "linux",
            "mode": "manual",
            "requested_shutter": f"1/{shutter_denominator}",
            "aligned_shutter": f"1/{aligned}",
            "controls": applied,
        }
