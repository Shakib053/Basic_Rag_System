import unittest

from evaluation.relevance_calibration import calibrate_threshold


class RelevanceCalibrationTests(unittest.TestCase):
    def test_calibration_prefers_perfect_separation_between_classes(self):
        result = calibrate_threshold(
            [
                {"score": 4.0, "relevant": True},
                {"score": 2.0, "relevant": True},
                {"score": -2.0, "relevant": False},
                {"score": -4.0, "relevant": False},
            ]
        )

        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertGreater(result.threshold, -2.0)
        self.assertLess(result.threshold, 2.0)

    def test_calibration_requires_both_classes(self):
        with self.assertRaises(ValueError):
            calibrate_threshold([{"score": 1.0, "relevant": True}])


if __name__ == "__main__":
    unittest.main()
