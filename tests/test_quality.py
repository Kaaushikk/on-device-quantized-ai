import unittest
from quantbench.experiment import quality_metrics


class QualityTests(unittest.TestCase):
    def test_confusion_and_macro_f1(self):
        result = quality_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 2)

    def test_different_input_counts_rejected(self):
        with self.assertRaises(ValueError):
            quality_metrics([0, 1], [0])
