from __future__ import annotations

from collections import deque
from typing import Callable, Any

import cv2
import numpy as np


class AdaptiveCameraPipeline:
    """Meter the tray, adjust the hardware shutter, and use bounded temporal filtering.

    All methods run on the camera owner thread. Row correction compares temporal
    profiles of the same scene; it does not flatten the tray's spatial shading.
    """

    def __init__(
        self, config: dict[str, Any], roi: dict | None,
        exposure: int, set_exposure: Callable[[int], int],
        exposure_limits: tuple[int, int] = (5, 333),
        *, lock_after_setup: bool = True,
    ) -> None:
        self.config = config
        self.lock_after_setup = lock_after_setup
        self.configuration_locked = False
        self.roi = roi
        self.exposure = exposure
        self.set_exposure = set_exposure
        options = config.get("adaptive", {})
        self.target = float(np.clip(options.get("target_luma", 190), 100, 220))
        self.max_frames = max(1, min(6, int(options.get("max_average_frames", 6))))
        self.threshold = max(.005, float(options.get("flicker_threshold", .02)))
        self.row_correction = bool(options.get("row_correction", True))
        self.minimum, self.maximum = exposure_limits
        self._frames: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self._samples: deque[np.ndarray] = deque(maxlen=24)
        self._rect: tuple[int, int, int, int] | None = None
        self._previous: np.ndarray | None = None
        self._previous_motion: np.ndarray | None = None
        self._previous_valid_rows: np.ndarray | None = None
        self._last_adjustment = float("-inf")
        self._last_selection = float("-inf")
        self._skip = 0
        self._frame_count = 1
        self._luma = 0.
        self._raw_score = 0.
        self._output_score = 0.
        self._previous_output: np.ndarray | None = None
        self._motion = False
        self._moving_until = float("-inf")
        self._ready = False
        self._limited = False
        self._gain_range = [1., 1.]
        self._error = ""
        self._control_failed = False
        self._reference_rows: np.ndarray | None = None
        self._profile_count = 0
        self._meter_ema: float | None = None
        self._previous_mean: float | None = None

    def _sensor_roi(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        rotation = int(self.config.get("rotation", 0)) % 360
        display_width, display_height = (height, width) if rotation in (90, 270) else (width, height)
        mask = np.zeros((display_height, display_width), np.uint8)
        roi = self.roi or {"x": .15, "y": .15, "width": .7, "height": .7}
        x0 = max(0, min(display_width - 1, round(float(roi["x"]) * display_width)))
        y0 = max(0, min(display_height - 1, round(float(roi["y"]) * display_height)))
        x1 = max(x0 + 1, min(display_width, round((float(roi["x"]) + float(roi["width"])) * display_width)))
        y1 = max(y0 + 1, min(display_height, round((float(roi["y"]) + float(roi["height"])) * display_height)))
        mask[y0:y1, x0:x1] = 255
        inverse = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
        if rotation in inverse:
            mask = cv2.rotate(mask, inverse[rotation])
        return cv2.boundingRect(mask)

    def _sample(self, frame: np.ndarray) -> np.ndarray:
        if self._rect is None:
            self._rect = self._sensor_roi(frame.shape)
        x, y, width, height = self._rect
        crop = frame[y:y + height, x:x + width]
        return cv2.cvtColor(cv2.resize(crop, (192, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)

    @staticmethod
    def _normalized(sample: np.ndarray) -> np.ndarray:
        return sample / max(float(sample.mean()), 1.)

    @classmethod
    def _band_score(cls, first: np.ndarray, second: np.ndarray) -> float:
        difference = cls._normalized(second) - cls._normalized(first)
        return float(np.std(np.median(difference, axis=1)))

    def _candidate_score(self, count: int) -> float:
        samples = list(self._samples)
        if len(samples) < count + 4:
            return float("inf")
        # Evaluate several consecutive windows, including the newest one.
        windows = [np.mean(samples[i - count:i], axis=0) for i in range(max(count, len(samples) - 6), len(samples) + 1)]
        return float(np.median([self._band_score(a, b) for a, b in zip(windows, windows[1:])]))

    def _reset(self) -> None:
        self._frames.clear()
        self._samples.clear()
        self._previous_output = None
        self._previous = None
        self._previous_motion = None
        self._previous_valid_rows = None
        self._ready = False
        self._frame_count = 1
        self._gain_range = [1., 1.]
        self._reference_rows = None
        self._profile_count = 0
        self._meter_ema = None
        self._previous_mean = None

    def apply(self, frame: np.ndarray, now: float) -> np.ndarray | None:
        if self.configuration_locked:
            # Keep applying the chosen filter without metering, shutter writes,
            # motion checks or new capture-readiness decisions.
            self._frames.append(frame)
            count = min(self._frame_count, len(self._frames))
            output = frame
            if count > 1:
                accumulator = np.zeros(frame.shape, np.float32)
                for prior in list(self._frames)[-count:]:
                    np.add(accumulator, prior, out=accumulator, casting="unsafe")
                output = cv2.convertScaleAbs(accumulator, alpha=1. / count)
            return self._correct_rows(output)
        if self._skip:
            self._skip -= 1
            return None
        sample = self._sample(frame)
        mean = float(sample.mean())
        if self._previous_mean is not None:
            light_ratio = mean / max(self._previous_mean, 1.)
            if light_ratio < .65 or light_ratio > 1.4:
                self._reset()
        self._previous_mean = mean
        normalized = self._normalized(sample)
        row_luma = np.percentile(sample, 70, axis=1)
        motion_sample = cv2.GaussianBlur(sample / np.maximum(row_luma, 5.)[:, None], (3, 3), 0)
        valid_rows = row_luma > 25
        if self._previous is not None:
            difference = normalized - self._previous
            self._raw_score = float(np.std(np.median(difference, axis=1)))
        if self._previous_motion is not None:
            # Normalize each row multiplicatively before detecting movement.
            # Subtracting an additive band signal mistakes static gold for motion
            # when illumination strongly changes its contrast against the tray.
            valid = valid_rows & self._previous_valid_rows
            if np.any(valid):
                residual = motion_sample[valid] - self._previous_motion[valid]
                if float(np.mean(np.abs(residual) > .25)) > .03:
                    self._moving_until = now + .4
        self._motion = now < self._moving_until
        if self._motion:
            self._reset()
        self._previous = normalized
        self._previous_motion = motion_sample
        self._previous_valid_rows = valid_rows
        self._frames.append(frame)
        self._samples.append(sample)
        rows = np.median(normalized, axis=1)
        self._profile_count += 1
        alpha = 1. / min(self._profile_count, 25)
        self._reference_rows = rows if self._reference_rows is None else (1 - alpha) * self._reference_rows + alpha * rows

        # Meter several lighting phases, not whichever bright/dark band happens
        # to be passing through the tray on the current frame.
        if len(self._samples) >= 9 and not self._motion and now - self._last_adjustment >= 1.:
            metered = np.mean(list(self._samples)[-12:], axis=0)
            measured = float(np.percentile(metered, 80))
            self._meter_ema = measured if self._meter_ema is None else .9 * self._meter_ema + .1 * measured
            self._luma = self._meter_ema
            enough_phases = len(self._samples) >= 18 or self._luma > 225 or self._luma < 150
            if enough_phases and abs(self._luma - self.target) > 12 and not self._control_failed:
                upper_ratio = 2. if self._luma < 140 else 1.5
                ratio = float(np.clip((self.target / max(self._luma, 1.)) ** 2, .55, upper_ratio))
                if self._luma > 225:
                    ratio = min(ratio, .7)
                requested = max(self.minimum, min(self.maximum, round(self.exposure * ratio)))
                self._limited = requested == self.exposure
                if requested != self.exposure:
                    try:
                        actual = int(self.set_exposure(requested))
                        if not self.minimum <= actual <= self.maximum:
                            raise RuntimeError(f"Invalid exposure readback: {actual}")
                        if actual == self.exposure:
                            raise RuntimeError("Camera did not change exposure")
                        self.exposure = actual
                        self._last_adjustment = now
                        self._reset()
                        self._skip = 3
                        return None
                    except (OSError, RuntimeError, ValueError) as exc:
                        self._error = str(exc)
                        self._control_failed = True

        if self._motion:
            self._frame_count = 1
        elif len(self._samples) >= 8 and now - self._last_selection >= 1.:
            candidates = sorted({1, min(3, self.max_frames), self.max_frames})
            scores = {count: self._candidate_score(count) for count in candidates}
            acceptable = [count for count in candidates if scores[count] <= self.threshold]
            self._frame_count = min(acceptable) if acceptable else min(scores, key=scores.get)
            if not acceptable and 3 in scores and self._frame_count > 3 and scores[3] <= scores[self._frame_count] * 1.25:
                self._frame_count = 3
            self._last_selection = now
        count = min(self._frame_count, len(self._frames))
        output = frame
        if count > 1:
            accumulator = np.zeros(frame.shape, np.float32)
            for prior in list(self._frames)[-count:]:
                np.add(accumulator, prior, out=accumulator, casting="unsafe")
            output = cv2.convertScaleAbs(accumulator, alpha=1. / count)

        output = self._correct_rows(output)

        output_sample = self._sample(output)
        if self._previous_output is not None:
            self._output_score = self._band_score(self._previous_output, output_sample)
        self._previous_output = output_sample
        self._ready = (
            len(self._samples) >= 12 and now - self._last_adjustment >= 1.2
            and not self._motion and not self._control_failed
            and self._output_score < max(.06, self.threshold * 3)
        )
        if self._ready and self.lock_after_setup:
            self.configuration_locked = True
        return output

    def _correct_rows(self, output: np.ndarray) -> np.ndarray:
        self._gain_range = [1., 1.]
        if self.row_correction and not self._motion and len(self._samples) >= 12:
            # A temporal reference retains stationary shadows, printed marks,
            # and geometry. Apply only the remaining changing row component.
            current = self._normalized(self._sample(output))
            target_rows = self._reference_rows
            current_rows = np.median(current, axis=1)
            gains = target_rows / np.maximum(current_rows, .05)
            gains = cv2.GaussianBlur(gains.reshape(-1, 1), (1, 5), 0).ravel()
            gains = np.clip(gains, .8, 1.25)
            gains /= max(float(np.median(gains)), .05)
            gains = np.clip(gains, .8, 1.25)
            if float(np.max(np.abs(gains - 1))) > .012:
                _, y, _, height = self._rect
                rows = (np.arange(output.shape[0]) - y) * (len(gains) - 1) / max(height - 1, 1)
                full_gains = np.interp(rows, np.arange(len(gains)), gains).astype(np.float32)
                working = output.astype(np.float32)
                np.multiply(working, full_gains[:, None, None], out=working)
                np.clip(working, 0, 255, out=working)
                output = working.astype(np.uint8)
                self._gain_range = [round(float(gains.min()), 3), round(float(gains.max()), 3)]

        return output

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "configuration_locked": self.configuration_locked,
            "exposure_units": self.exposure,
            "exposure_ms": round(self.exposure / 10., 2),
            "target_luma": self.target,
            "metered_luma": round(self._luma, 2),
            "averaged_frames": min(self._frame_count, len(self._frames)) or 1,
            "max_average_frames": self.max_frames,
            "raw_band_score": round(self._raw_score, 5),
            "output_band_score": round(self._output_score, 5),
            "row_gain_range": self._gain_range,
            "motion_detected": self._motion,
            "exposure_limited": self._limited,
            "error": self._error,
        }
