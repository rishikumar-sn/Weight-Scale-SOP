"""Check gross-weight capping without changing calibrated stone estimates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.domain.weights import stone_weight_fields, weight_rows, weight_summary


class WeightCapTests(unittest.TestCase):
    def state(self, gross=49.76, estimate=73.58):
        return {
            "weight_g": gross,
            "result": {"count": 1, "items": [{"stones": {
                "found": True,
                **stone_weight_fields({"success": True,
                                       "estimated_total_average_g": estimate}, True),
            }}]},
        }

    def test_exceeded_estimate_caps_all_displayed_weights(self):
        state = self.state()
        original = deepcopy(state)
        summary = weight_summary(state)
        for key in ("stone_g", "stone_min_g", "stone_max_g"):
            self.assertEqual(summary[key], 49.76)
        for key in ("net_g", "net_min_g", "net_max_g"):
            self.assertEqual(summary[key], 0.0)
        rows = weight_rows(summary)
        self.assertEqual(rows[2][1:], ["49.76 g", "49.76 g"])
        self.assertEqual(rows[3][1:], ["0.00 g", "0.00 g"])
        self.assertNotIn("73.58", str(rows))
        self.assertNotIn("74.58", str(rows))
        self.assertNotIn("+1", summary["note"])
        self.assertEqual(state, original)

    def test_normal_estimate_and_range_are_unchanged(self):
        summary = weight_summary(self.state(estimate=20))
        self.assertEqual(weight_rows(summary)[2][1:], ["20.00 g", "20.00 - 21.00 g"])
        self.assertEqual(summary["net_g"], 29.76)
        self.assertEqual((summary["net_min_g"], summary["net_max_g"]), (28.76, 29.76))

    def test_only_strictly_exceeded_estimates_trigger_cap(self):
        for estimate in (49.0, 49.76):
            with self.subTest(estimate=estimate):
                summary = weight_summary(self.state(estimate=estimate))
                self.assertEqual(summary["stone_g"], estimate)
                self.assertEqual(summary["stone_max_g"], estimate + 1)
                self.assertEqual(summary["net_min_g"], 0.0)

    def test_zero_gross_and_missing_gross(self):
        summary = weight_summary(self.state(gross=0))
        self.assertEqual(summary["stone_max_g"], 0.0)
        self.assertEqual(summary["net_g"], 0.0)
        summary = weight_summary(self.state(gross=None))
        self.assertEqual(summary["stone_g"], 73.58)
        self.assertIsNone(summary["net_g"])

    def test_missing_estimate_and_multiple_jewels_stay_unavailable(self):
        state = self.state(estimate=None)
        self.assertIsNone(weight_summary(state)["stone_g"])
        state = self.state()
        state["result"]["count"] = 2
        self.assertIsNone(weight_summary(state)["stone_g"])


if __name__ == "__main__":
    unittest.main()
