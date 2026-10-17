from __future__ import annotations

import math
import os
import re
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
        image_controls: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if os.name == "nt":
            return self._apply_windows(cap, power_line_hz, shutter_denominator, driver_managed)
        return self._apply_linux(camera_index, power_line_hz, shutter_denominator, driver_managed, image_controls)

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
        driver_managed: bool = True,
        image_controls: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        device = f"/dev/video{camera_index}"
        applied: dict[str, bool] = {}
        errors: dict[str, str] = {}

        def run(*args: str) -> str:
            return subprocess.run(
                ["v4l2-ctl", "-d", device, *args], check=True,
                capture_output=True, text=True, timeout=3,
            ).stdout

        try:
            available = run("--list-ctrls")
        except (OSError, subprocess.SubprocessError) as exc:
            return {"platform": "linux", "mode": "unavailable", "error": str(exc), "controls": {}}

        # Kernel versions use either the legacy or newer UVC control names.
        auto_name = "auto_exposure" if "auto_exposure " in available else "exposure_auto"
        shutter_name = "exposure_time_absolute" if "exposure_time_absolute " in available else "exposure_absolute"
        priority_name = "exposure_dynamic_framerate" if "exposure_dynamic_framerate " in available else "exposure_auto_priority"
        controls = {"power_line_frequency": 1 if power_line_hz == 50 else 2}
        controls[auto_name] = 3 if driver_managed else 1
        if priority_name in available:
            controls[priority_name] = 0
        # Manual shutters must integrate a whole 100/120 Hz lighting period.
        period = 1.0 / (2 * power_line_hz)
        shutter = max(1, round((1.0 / max(1, shutter_denominator)) / period)) * period
        if not driver_managed:
            controls[shutter_name] = max(1, round(10000 * shutter))
        # Persist the calibrated tone controls as well as shutter settings.
        # BRIO controls can reset after unplugging, or retain another app's values.
        for name in ("brightness", "contrast", "saturation", "gain", "backlight_compensation"):
            if image_controls and name in image_controls:
                controls[name] = int(image_controls[name])
        # White balance is independent of shutter. Resolve the different UVC
        # names, and disable AWB before writing a manual temperature (otherwise
        # the temperature control is inactive). Never force a guessed Kelvin
        # value when no calibration has been configured.
        options = image_controls or {}
        wb_name = next((name for name in (
            "white_balance_automatic", "white_balance_temperature_auto",
            "auto_white_balance",
        ) if re.search(r"^\s*" + name + r"\s+0x", available, re.M)), "white_balance_temperature_auto")
        wb_auto = next((options[name] for name in (
            "white_balance_automatic", "white_balance_temperature_auto", "auto_white_balance",
        ) if name in options), None)
        temperature = options.get("white_balance_temperature")
        if wb_auto is not None or temperature is not None:
            controls[wb_name] = int(wb_auto) if wb_auto is not None else 0
        if temperature is not None:
            if controls[wb_name] == 0:
                controls["white_balance_temperature"] = int(temperature)
            else:
                errors["white_balance_temperature"] = "Manual temperature requires automatic white balance disabled"
                applied["white_balance_temperature"] = False
        readback: dict[str, int] = {}
        for name, value in controls.items():
            try:
                run("-c", f"{name}={value}")
                actual = int(run("-C", name).rsplit(":", 1)[1].strip().split()[0])
                readback[name] = actual
                applied[name] = actual == value
                if not applied[name]:
                    errors[name] = f"Requested {value}, read back {actual}"
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                applied[name] = False
                errors[name] = getattr(exc, "stderr", None) or str(exc)
        # Include untouched controls too: retained settings from other camera
        # apps matter when diagnosing overexposure and colour shifts. Keep this
        # separate from readback, which reports requested controls only.
        observed: dict[str, int] = {}
        observation_error = ""
        try:
            observed = {name: int(value) for name, value in re.findall(
                r"^\s*(\w+)\s+0x[^\n]*?\bvalue=(-?\d+)", run("--list-ctrls"), re.M,
            )}
        except (OSError, subprocess.SubprocessError) as exc:
            observation_error = getattr(exc, "stderr", None) or str(exc)
        return {
            "platform": "linux",
            "mode": "v4l2-driver" if driver_managed else "manual",
            "power_line_hz": power_line_hz,
            "controls": applied,
            "requested_controls": controls,
            "readback": readback,
            "errors": errors,
            "observed_controls": observed,
            "observation_error": observation_error,
            "exposure_limits": self._exposure_limits(available, shutter_name),
            **({} if driver_managed else {"requested_shutter": f"1/{shutter_denominator}", "shutter_seconds": shutter}),
        }

    @staticmethod
    def _exposure_limits(available: str, name: str) -> dict[str, int]:
        for line in available.splitlines():
            if line.strip().startswith(name + " "):
                limits = dict(re.findall(r"\b(min|max)=(-?\d+)", line))
                if "min" in limits and "max" in limits:
                    return {key: int(value) for key, value in limits.items()}
        return {"min": 5, "max": 2500}
