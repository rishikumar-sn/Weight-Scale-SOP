import unittest

import cv2
import numpy as np

from backend.app.hardware.anti_flicker import AntiFlickerPipeline, ShutterController


class Capture:
    def __init__(self, backend="V4L2", reject=False, clamp=False):
        self.backend = backend
        self.reject = reject
        self.clamp = clamp
        self.values = {cv2.CAP_PROP_AUTO_EXPOSURE: 3.0, cv2.CAP_PROP_EXPOSURE: 333.0}

    def getBackendName(self):
        return self.backend

    def get(self, prop):
        return self.values[prop]

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_EXPOSURE and value in (100, 200):
            if self.reject:
                return False
            if self.clamp:
                value += 10
        self.values[prop] = value
        return True


class Hardware:
    supported = False
    error = "Windows only"
    _original_powerline = None

    def set_value(self, value):
        return False


class ShutterTests(unittest.TestCase):
    def test_presets_and_restore_original_after_switch(self):
        cap = Capture()
        shutter = ShutterController(cap)
        for denominator, expected in ((100, 100), (50, 200)):
            self.assertTrue(shutter.set_shutter(denominator))
            self.assertEqual(cap.get(cv2.CAP_PROP_EXPOSURE), expected)
            self.assertEqual(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), 1)
        self.assertTrue(shutter.restore())
        self.assertEqual(cap.get(cv2.CAP_PROP_EXPOSURE), 333)
        self.assertEqual(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), 3)

    def test_rejected_and_clamped_settings_restore_original(self):
        for cap in (Capture(reject=True), Capture(clamp=True)):
            shutter = ShutterController(cap)
            self.assertFalse(shutter.set_shutter(100))
            self.assertIn("rejected", shutter.error)
            self.assertEqual(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), 3)
            self.assertEqual(cap.get(cv2.CAP_PROP_EXPOSURE), 333)

    def test_pipeline_status_raw_frames_and_mode_exit(self):
        for backend, ready in (("V4L2", True), ("DSHOW", False)):
            cap = Capture(backend)
            updates = []
            pipeline = AntiFlickerPipeline(Hardware(), updates.append, ShutterController(cap))
            pipeline.configure("shutter_50", 0)
            frame = np.full((180, 320, 3), 128, np.uint8)
            self.assertIs(pipeline.process_frame(frame, 1), frame)
            self.assertEqual(updates[-1]["ready"], ready)
            self.assertEqual(updates[-1]["status"], "ACTIVE" if ready else "UNAVAILABLE")
            self.assertEqual(updates[-1]["target_frames"], 1)
            pipeline.configure("normal", 2)
            self.assertEqual(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), 3)
            self.assertEqual(cap.get(cv2.CAP_PROP_EXPOSURE), 333)

class FilterIntegrationTests(unittest.TestCase):
    def test_averaging_warmup_and_switch_flushes_previous_scene(self):
        for mode, count in (("average_30", 30), ("average_90", 90), ("hybrid", 90)):
            updates = []
            pipeline = AntiFlickerPipeline(Hardware(), updates.append)
            pipeline.configure(mode, 0, force=True)
            dark = np.full((90, 120, 3), 40, np.uint8)
            for index in range(count - 1):
                pipeline.process_frame(dark, index / 15)
            self.assertFalse(updates[-1]["ready"])
            pipeline.process_frame(dark, count / 15)
            self.assertTrue(updates[-1]["ready"])
            pipeline.configure("normal", 10)
            bright = np.full_like(dark, 200)
            np.testing.assert_array_equal(pipeline.process_frame(bright, 11), bright)

    def test_driver_frequency_choices(self):
        class Driver(Hardware):
            supported = True
            error = ""
            def set_value(self, value):
                self.value = value
                return True
        driver = Driver()
        updates = []
        pipeline = AntiFlickerPipeline(driver, updates.append)
        pipeline.configure("driver_50", 0, force=True)
        self.assertEqual(driver.value, 1)
        pipeline.configure("driver_60", 1)
        self.assertEqual(driver.value, 2)
        self.assertEqual(updates[-1]["target_frames"], 1)

    def test_capture_rejects_unsettled_or_unsupported_filter(self):
        from datetime import datetime
        from backend.app.hardware.camera import CameraService
        for detail in ("", "Shutter control is unavailable"):
            service = CameraService(lambda: {"camera": {"rotation": 0}})
            service._frame = np.zeros((90, 120, 3), np.uint8)
            service._frame_time = datetime.now().astimezone()
            service._method_pipeline = object()
            service._method_status = {"ready": False, "detail": detail}
            with self.assertRaises(RuntimeError):
                service.snapshot_frame()
            service._method_status = {"ready": True}
            frame, metadata = service.snapshot_frame()
            self.assertEqual(frame.shape, (90, 120, 3))

    def test_invalid_mode_cannot_be_saved(self):
        from pydantic import ValidationError
        from backend.app.domain.schemas import CameraFlickerSettings
        with self.assertRaises(ValidationError):
            CameraFlickerSettings(mode="unknown")


if __name__ == "__main__":
    unittest.main()
