"""Run with: .venv/bin/python -m unittest discover -s tests -v"""
import unittest

import cv2
import numpy as np

from backend.app.hardware.adaptive import AdaptiveCameraPipeline
from backend.app.hardware.camera import CameraService


class AdaptiveCameraTests(unittest.TestCase):
    def make_pipeline(self, **options):
        changes = []
        def set_exposure(value):
            changes.append(value)
            return value
        config = {"rotation": 0, "adaptive": options}
        # Exercise the controller's response while it is still configuring.
        pipeline = AdaptiveCameraPipeline(config, {"x": 0, "y": 0, "width": 1, "height": 1}, 100, set_exposure, lock_after_setup=False)
        return pipeline, changes

    def image(self, exposure=100, lamp=1., phase=None):
        base = np.full((96, 160, 3), 190., np.float32)
        # Stationary gold-coloured object and spatial shading must survive.
        base[35:60, 60:100] = (45, 105, 150)
        base *= np.linspace(.9, 1.05, 96)[:, None, None]
        base *= np.sqrt(exposure / 100.) * lamp
        if phase is not None:
            base *= (1 + .35 * np.sin(np.arange(96) * .15 + phase))[:, None, None]
        return np.clip(base, 0, 235).astype(np.uint8)

    def test_light_increase_and_decrease_adjust_actual_shutter(self):
        pipeline, changes = self.make_pipeline()
        index = 0
        for lamp in (1., 2., .7):
            for _ in range(100):
                pipeline.apply(self.image(pipeline.exposure, lamp), index * .1)
                index += 1
            self.assertLess(abs(pipeline.status()["metered_luma"] - 190), 15)
            self.assertTrue(pipeline.status()["ready"])
            if lamp == 2.:
                bright_exposure = pipeline.exposure
        self.assertLess(bright_exposure, 100)
        self.assertGreater(pipeline.exposure, bright_exposure)
        self.assertTrue(all(5 <= value <= 333 for value in changes))

    def test_clean_light_needs_only_one_frame_and_preserves_scene(self):
        pipeline, _ = self.make_pipeline()
        image = self.image()
        for index in range(40):
            result = pipeline.apply(image, index * .1)
        self.assertEqual(pipeline.status()["averaged_frames"], 1)
        self.assertEqual(pipeline.status()["row_gain_range"], [1., 1.])
        np.testing.assert_array_equal(result, image)

    def test_bands_choose_short_average_and_reduce_residual(self):
        pipeline, _ = self.make_pipeline()
        raw_scores, output_scores = [], []
        for index in range(150):
            pipeline.apply(self.image(pipeline.exposure, phase=index * 2 * np.pi / 3), index * .1)
            if index > 100:
                state = pipeline.status()
                raw_scores.append(state["raw_band_score"])
                output_scores.append(state["output_band_score"])
                self.assertLessEqual(state["averaged_frames"], 6)
        self.assertIn(pipeline.status()["averaged_frames"], (3, 6))
        self.assertLess(np.median(output_scores), np.median(raw_scores) / 3)

    def test_movement_flushes_history_and_avoids_trails(self):
        pipeline, _ = self.make_pipeline()
        for index in range(40):
            pipeline.apply(self.image(), index * .1)
        changed = self.image()
        changed[20:70, 15:65] = (25, 40, 65)
        output = pipeline.apply(changed, 4.)
        self.assertTrue(pipeline.status()["motion_detected"])
        self.assertEqual(pipeline.status()["averaged_frames"], 1)
        self.assertFalse(pipeline.status()["ready"])
        np.testing.assert_array_equal(output, changed)

    def test_hardware_failure_is_reported_without_repeated_writes(self):
        pipeline, _ = self.make_pipeline()
        attempts = []
        def reject(value):
            attempts.append(value)
            raise RuntimeError("driver rejected shutter")
        pipeline.set_exposure = reject
        for index in range(50):
            pipeline.apply(self.image(lamp=2), index * .1)
        self.assertEqual(len(attempts), 1)
        self.assertIn("driver rejected shutter", pipeline.status()["error"])

    def test_extreme_darkness_stops_at_exposure_limit(self):
        pipeline, changes = self.make_pipeline()
        for index in range(100):
            pipeline.apply(self.image(pipeline.exposure, lamp=.01), index * .1)
        self.assertEqual(pipeline.exposure, 333)
        self.assertTrue(pipeline.status()["exposure_limited"])
        self.assertTrue(pipeline.status()["ready"])

    def test_sensor_roi_maps_display_rotation(self):
        roi = {"x": .1, "y": .2, "width": .5, "height": .6}
        pipeline = AdaptiveCameraPipeline({"rotation": 270}, roi, 100, lambda value: value)
        x, y, width, height = pipeline._sensor_roi((100, 200, 3))
        self.assertEqual((x, y, width, height), (40, 10, 120, 50))

    def test_capture_waits_until_adaptation_settles(self):
        camera = CameraService(lambda: {"camera": {"rotation": 0}})
        camera._adaptive, _ = self.make_pipeline()
        camera._adaptive_status = {"ready": False}
        with self.assertRaisesRegex(RuntimeError, "adapting to lighting"):
            camera.snapshot_frame()

    def test_irregular_flicker_sampling_settles_without_shutter_hunting(self):
        pipeline, changes = self.make_pipeline()
        phase = 0.
        steady_changes = 0
        ready = []
        for index in range(220):
            phase += 2 * np.pi * 100 / 29.97 * (2 if index % 13 == 0 else 1)
            pipeline.apply(self.image(pipeline.exposure, lamp=1.5, phase=phase), index * .1)
            if index == 159:
                steady_changes = len(changes)
            if index >= 160:
                ready.append(pipeline.status()["ready"])
        self.assertLessEqual(len(changes) - steady_changes, 2)
        self.assertGreater(sum(ready) / len(ready), .7)

    def test_large_configured_window_is_capped_at_six(self):
        pipeline, _ = self.make_pipeline(max_average_frames=90)
        self.assertEqual(pipeline.max_frames, 6)


if __name__ == "__main__":
    unittest.main()
