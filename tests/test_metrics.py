import unittest
from quantbench.metrics import percentile, summarize


class MetricsTests(unittest.TestCase):
    def test_known_interpolation(self):
        self.assertEqual(percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)

    def test_throughput_counts_examples(self):
        result = summarize([1_000_000_000, 1_000_000_000], 4)
        self.assertEqual(result["examples_per_second"], 4)
        self.assertEqual(result["p50_batch_ms"], 1000)

    def test_invalid_measurements(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                summarize(values, 1)
        with self.assertRaises(ValueError):
            summarize([1], 0)


if __name__ == "__main__":
    unittest.main()
