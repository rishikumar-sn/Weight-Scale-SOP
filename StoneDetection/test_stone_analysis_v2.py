#!/usr/bin/env python3
"""Offline tests and optional artifact generator for stone analysis V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

import stone_analysis_v2 as v2
import stone_area_calculator as calculator


def circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, cv2.FILLED)
    return mask


class FakeFastSam:
    """Return one overgrown mask and one candidate-centred plausible mask."""

    def infer(self, crop, conf_thres, iou_thres, mask_thres):
        height, width = crop.shape[:2]
        huge = np.ones((height, width), dtype=np.uint8)
        plausible = circle_mask(
            (height, width),
            (width // 2, height // 2),
            max(4, min(height, width) // 7),
        )
        return [
            SimpleNamespace(mask=huge, score=0.99),
            SimpleNamespace(mask=plausible, score=0.82),
        ]


class StoneAnalysisV2Tests(unittest.TestCase):
    def setUp(self):
        self.shape = (120, 120)
        self.jewel = circle_mask(self.shape, (60, 60), 50)
        self.gold = self.jewel.copy()
        self.strict_gold = np.zeros(self.shape, dtype=np.uint8)
        self.image = np.full((*self.shape, 3), (30, 170, 210), dtype=np.uint8)
        cv2.circle(self.image, (60, 60), 13, (20, 90, 20), cv2.FILLED)
        self.seed = circle_mask(self.shape, (60, 60), 5)

    def candidate(self, seed=None, proposal=None):
        seed = self.seed if seed is None else seed
        proposal = seed if proposal is None else proposal
        return {
            "candidate_id": 1,
            "seed_mask": seed,
            "proposal_mask": proposal,
            "bbox": v2._bbox(seed),
            "centroid": v2._centroid(seed),
            "source_methods": ["hsv_lab", "structural"],
            "initial_color_votes": {"Green": 100.0},
            "seed_area_px": int(cv2.countNonZero(seed)),
            "confidence": 0.85,
            "structural_score": 0.80,
            "source_region": {},
        }

    def test_fastsam_rejects_overgrown_mask_and_selects_candidate(self):
        result = v2.refine_stone_candidate(
            self.image,
            self.jewel,
            self.candidate(),
            self.gold,
            self.strict_gold,
            fastsam_model=FakeFastSam(),
        )
        self.assertEqual(result["method"], "fastsam")
        self.assertLess(result["diagnostics"]["jewel_area_fraction"], 0.15)
        self.assertGreaterEqual(result["diagnostics"]["seed_overlap"], 0.45)

    def test_opencv_fallback_does_not_accept_giant_proposal(self):
        giant = self.jewel.copy()
        result = v2.refine_stone_candidate(
            self.image,
            self.jewel,
            self.candidate(proposal=giant),
            self.gold,
            self.strict_gold,
            fastsam_model=None,
        )
        self.assertIn(result["method"], {"opencv", "seed_fallback"})
        self.assertLess(result["diagnostics"]["jewel_area_fraction"], 0.20)

    def test_adjacent_candidates_remain_two_instances(self):
        left = circle_mask(self.shape, (52, 60), 7)
        right = circle_mask(self.shape, (68, 60), 7)
        candidates = [self.candidate(left, left), self.candidate(right, right)]
        candidates[1]["candidate_id"] = 2
        candidates[1]["centroid"] = v2._centroid(right)
        candidates[1]["bbox"] = v2._bbox(right)
        instances, diagnostics = v2.build_final_stone_instances(
            self.image,
            self.jewel,
            candidates,
            self.gold,
            self.strict_gold,
        )
        self.assertEqual(len(instances), 2)
        self.assertEqual(diagnostics["stone_instance_count"], 2)

    def test_candidate_near_jewel_boundary_does_not_expand_over_jewel(self):
        seed = circle_mask(self.shape, (105, 60), 4)
        result = v2.refine_stone_candidate(
            self.image,
            self.jewel,
            self.candidate(seed, self.jewel),
            self.gold,
            self.strict_gold,
        )
        self.assertLess(result["diagnostics"]["jewel_area_fraction"], 0.20)

    def test_dark_green_is_not_black(self):
        image = np.full((*self.shape, 3), (5, 45, 5), dtype=np.uint8)
        result = v2.classify_stone_instance_color(image, self.seed)
        self.assertEqual(result["color"], "Green")

    def test_true_black_remains_black(self):
        image = np.full((*self.shape, 3), (8, 8, 8), dtype=np.uint8)
        result = v2.classify_stone_instance_color(image, self.seed)
        self.assertEqual(result["color"], "Black")

    def test_black_candidate_uses_strict_not_broad_gold_overlap(self):
        # Dark warm onyx pixels fall inside the broad shadow-gold range used
        # by production (the failed captures measured 0.32--0.47 broad overlap)
        # while having no strict-gold overlap. They must remain valid stones.
        self.assertFalse(
            v2.should_reject_neutral_candidate(
                "Black", 588, 0.47, 0.0, ["structural", "dark_structure"]
            )
        )
        self.assertTrue(
            v2.should_reject_neutral_candidate(
                "Black", 588, 0.47, 0.31, ["structural", "dark_structure"]
            )
        )
        self.assertTrue(
            v2.should_reject_neutral_candidate(
                "Black", 72, 0.54, 0.0, ["hsv_lab", "non_gold_hsv_residual"]
            )
        )

    def test_white_candidate_still_uses_broad_gold_overlap(self):
        self.assertTrue(
            v2.should_reject_neutral_candidate("White/Colorless", 130, 0.21, 0.0)
        )
        self.assertFalse(
            v2.should_reject_neutral_candidate("White/Colorless", 130, 0.19, 0.0)
        )

    def test_white_and_faceted_yellow_have_structural_candidates(self):
        for expected_source, body in (
            ("white_colorless_structure", (220, 220, 220)),
            ("yellow_structure", (10, 190, 220)),
        ):
            with self.subTest(expected_source=expected_source):
                image = np.full((*self.shape, 3), (20, 170, 215), dtype=np.uint8)
                cv2.circle(image, (60, 60), 16, body, cv2.FILLED)
                cv2.line(image, (49, 60), (71, 60), (235, 245, 250), 3)
                cv2.line(image, (60, 49), (60, 71), (30, 100, 130), 3)
                candidates = v2.generate_stone_candidates(
                    image,
                    self.jewel,
                    [],
                    self.gold,
                )
                self.assertTrue(
                    any(
                        expected_source in candidate["source_methods"]
                        for candidate in candidates
                    )
                )

    def test_reflection_only_candidate_is_diagnostic_not_instance(self):
        candidate = self.candidate()
        candidate["source_methods"] = ["reflection_evidence"]
        instances, diagnostics = v2.build_final_stone_instances(
            self.image,
            self.jewel,
            [candidate],
            self.gold,
            self.strict_gold,
        )
        self.assertEqual(instances, [])
        self.assertFalse(diagnostics["candidates"][0]["accepted"])

    def test_structural_gold_texture_is_not_a_stone(self):
        candidate = self.candidate()
        candidate["source_methods"] = [
            "structural",
            "edge_enclosure",
            "local_contrast",
            "yellow_structure",
        ]
        instances, diagnostics = v2.build_final_stone_instances(
            self.image,
            self.jewel,
            [candidate],
            self.gold,
            self.strict_gold,
        )
        self.assertEqual(instances, [])
        self.assertTrue(diagnostics["candidates"][0]["metal_texture_rejected"])

    def test_isolated_tiny_black_instance_is_rejected_after_v2(self):
        instances = [
            {
                "color": "Black",
                "area_px": 45,
                "center": [347, 294],
                # This legacy flag can become stale when its neighbours are
                # rejected during V2 refinement.
                "micro_stone_supported": True,
                "learned_matches": [],
            }
        ]
        filtered, rejected = v2.filter_unsupported_neutral_instances(
            instances,
            min_single_area_px=61,
        )
        self.assertEqual(filtered, [])
        self.assertEqual(rejected, 1)

    def test_repeated_tiny_black_instances_remain_supported(self):
        instances = [
            {
                "color": "Black",
                "area_px": 30,
                "center": [40 + offset * 10, 50],
                "learned_matches": [],
            }
            for offset in range(3)
        ]
        filtered, rejected = v2.filter_unsupported_neutral_instances(
            instances,
            min_single_area_px=61,
        )
        self.assertEqual(len(filtered), 3)
        self.assertEqual(rejected, 0)
        self.assertTrue(all(item["final_micro_stone_supported"] for item in filtered))

    def test_lab_color_families(self):
        samples = {
            "Blue": (180, 60, 30),
            "Purple/Violet": (170, 40, 130),
            "Pink": (170, 120, 240),
            "Yellow/Gold": (20, 190, 220),
            "White/Colorless": (210, 210, 210),
        }
        for expected, bgr in samples.items():
            with self.subTest(expected=expected):
                image = np.full((*self.shape, 3), bgr, dtype=np.uint8)
                result = v2.classify_stone_instance_color(image, self.seed)
                self.assertEqual(result["color"], expected)

    def test_non_gold_hsv_residual_keeps_blue_green_and_purple_faces(self):
        image = np.full((*self.shape, 3), (20, 170, 215), dtype=np.uint8)
        gold = self.jewel.copy()
        expected_centers = [(40, 60), (60, 60), (80, 60)]
        for center, color in zip(
            expected_centers,
            ((180, 55, 25), (30, 105, 25), (135, 35, 120)),
        ):
            cv2.circle(image, center, 8, color, cv2.FILLED)
            cv2.circle(gold, center, 9, 0, cv2.FILLED)
        candidates = v2.generate_stone_candidates(image, self.jewel, [], gold)
        residual_candidates = [
            candidate
            for candidate in candidates
            if "non_gold_hsv_residual" in candidate["source_methods"]
        ]
        self.assertGreaterEqual(len(residual_candidates), 3)
        for center in expected_centers:
            self.assertTrue(
                any(candidate["seed_mask"][center[1], center[0]] for candidate in residual_candidates)
            )

    def test_non_gold_neutral_residual_keeps_complete_white_face(self):
        image = np.full((*self.shape, 3), (20, 170, 215), dtype=np.uint8)
        gold = self.jewel.copy()
        cv2.circle(image, (60, 60), 15, (185, 190, 195), cv2.FILLED)
        cv2.circle(image, (60, 60), 15, (70, 75, 80), 2)
        cv2.circle(gold, (60, 60), 16, 0, cv2.FILLED)
        candidates = v2.generate_stone_candidates(image, self.jewel, [], gold)
        neutral = [
            candidate
            for candidate in candidates
            if "non_gold_neutral_residual" in candidate["source_methods"]
        ]

        self.assertTrue(neutral)
        self.assertTrue(any(candidate["seed_mask"][60, 60] for candidate in neutral))

    def test_surface_risk_boundaries(self):
        self.assertEqual(v2.calculate_stone_surface_risk(0, 0)["level"], "NONE")
        self.assertEqual(v2.calculate_stone_surface_risk(4, 1)["level"], "LOW")
        self.assertEqual(v2.calculate_stone_surface_risk(19.99, 2)["level"], "MEDIUM")
        self.assertEqual(v2.calculate_stone_surface_risk(20, 2)["level"], "HIGH")
        self.assertEqual(v2.calculate_stone_surface_risk(4, 10)["level"], "HIGH")
        self.assertEqual(
            v2.calculate_stone_surface_risk(
                30, 2, stone_area_mm2=4.99, jewellery_area_mm2=100
            )["level"],
            "NONE",
        )
        self.assertEqual(
            v2.calculate_stone_surface_risk(2, 2, stone_area_mm2=5.0)["level"],
            "LOW",
        )
        self.assertEqual(
            v2.calculate_stone_surface_risk(2, 2, stone_area_mm2=49.999)["level"],
            "LOW",
        )
        self.assertEqual(
            v2.calculate_stone_surface_risk(2, 2, stone_area_mm2=50.0)["level"],
            "MEDIUM",
        )
        self.assertEqual(
            v2.calculate_stone_surface_risk(2, 2, stone_area_mm2=100.0)["level"],
            "MEDIUM",
        )
        calibrated = v2.calculate_stone_surface_risk(2, 2, stone_area_mm2=100.001)
        self.assertEqual(calibrated["level"], "HIGH")
        self.assertEqual(calibrated["basis"], "calibrated_final_stone_area_mm2")

    def test_sub_five_area_uses_small_jewellery_footprint(self):
        relative = v2.calculate_stone_surface_risk(
            25, 1, stone_area_mm2=4.0, jewellery_area_mm2=16.0
        )
        self.assertEqual(relative["level"], "HIGH")
        self.assertEqual(
            relative["basis"],
            "calibrated_stone_area_relative_to_jewellery_footprint",
        )

        noise = v2.calculate_stone_surface_risk(
            4.9, 1, stone_area_mm2=4.0, jewellery_area_mm2=16.0
        )
        self.assertEqual(noise["level"], "NONE")
        self.assertTrue(noise["suppressed_as_likely_false_detection"])

    def test_weight_range_is_ordered_and_not_area_times_jewel_weight(self):
        mask = circle_mask((80, 80), (40, 40), 10)
        measurements = calculator.calculate_stone_measurements(
            {"Green": mask},
            mm_per_pixel_x=0.1,
            mm_per_pixel_y=0.1,
        )
        estimate = calculator.estimate_stone_weight_range(
            measurements,
            calculator.STONE_SETTING_PROFILE_OPEN_BACK,
            jewel_weight_g=20.0,
        )
        minimum = estimate["estimated_total_minimum_g"]
        typical = estimate["estimated_total_typical_g"]
        maximum = estimate["estimated_total_maximum_g"]
        self.assertGreaterEqual(minimum, 0.0)
        self.assertLessEqual(minimum, typical)
        self.assertLessEqual(typical, maximum)
        self.assertNotAlmostEqual(typical, 20.0 * 0.10, places=3)
        self.assertIn("density", estimate["weight_method"])

    def test_weight_sanity_bound_keeps_raw_values_and_warning(self):
        estimate = calculator.calibrate_weight_estimate_to_jewel_weight(
            {
                "success": True,
                "v2_geometry_estimate": True,
                "estimated_total_minimum_g": 8.0,
                "estimated_total_average_g": 12.0,
                "estimated_total_maximum_g": 16.0,
                "weight_confidence": "Medium",
                "weight_confidence_score": 0.65,
                "weight_warnings": [],
            },
            10.0,
        )
        self.assertLessEqual(estimate["estimated_total_maximum_g"], 10.0)
        self.assertEqual(estimate["raw_estimated_total_maximum_g"], 16.0)
        self.assertTrue(estimate["sanity_constraint_applied"])
        self.assertTrue(estimate["weight_warnings"])

    def test_giant_merged_region_lowers_weight_confidence(self):
        estimate = calculator.estimate_stone_weight_range(
            {
                "success": True,
                "instances": [
                    {
                        "shape": "irregular",
                        "major_axis_mm": 40.0,
                        "minor_axis_mm": 30.0,
                        "area_mm2": 900.0,
                        "equivalent_diameter_mm": 33.85,
                        "color": "White/Colorless",
                    }
                ],
            },
            calculator.STONE_SETTING_PROFILE_OPEN_BACK,
            jewel_weight_g=None,
        )
        self.assertEqual(estimate["weight_confidence"], "Low")
        self.assertTrue(
            any("merged stones" in warning for warning in estimate["weight_warnings"])
        )


def generate_offline_artifacts(image_path: Path, output_dir: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, jewel = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    jewel = calculator.remove_small_components(jewel, max(20, jewel.size // 5000))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, np.array((8, 35, 35)), np.array((42, 255, 255)))
    strict_gold = cv2.inRange(hsv, np.array((12, 75, 55)), np.array((36, 255, 255)))
    candidates = v2.generate_stone_candidates(image, jewel, [], gold)
    instances, diagnostics = v2.build_final_stone_instances(
        image,
        jewel,
        candidates,
        gold,
        strict_gold,
    )
    color_image = v2.build_color_measurement_image(image, jewel)
    for instance in instances:
        instance.update(v2.classify_stone_instance_color(color_image, instance["mask"], gold))
    v2.save_debug_artifacts(output_dir, image, jewel, candidates, instances)
    serializable = {
        "stone_instance_count": len(instances),
        "segmentation_method_counts": diagnostics["segmentation_method_counts"],
        "instances": [
            {
                key: value
                for key, value in instance.items()
                if key not in {"mask", "seed_mask", "contour"}
            }
            for instance in instances
        ],
        "candidate_diagnostics": diagnostics["candidates"],
    }
    (output_dir / "result.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--image":
        parser = argparse.ArgumentParser()
        parser.add_argument("--image", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=Path("stone_analysis_v2_output"))
        options = parser.parse_args()
        generate_offline_artifacts(options.image, options.output)
    else:
        unittest.main()
