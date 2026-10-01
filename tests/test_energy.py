import unittest
from quantbench.energy import integrate_power


class EnergyTests(unittest.TestCase):
    def test_integrates_irregular_samples(self):
        result = integrate_power([(0, 10), (1, 20), (3, 10)])
        self.assertEqual(result['energy_joules'], 45)
        self.assertEqual(result['mean_watts'], 15)

    def test_rejects_unusable_measurements(self):
        for samples in ([], [(0, 10)], [(0, 0), (1, 0)], [(1, 10), (1, 10)],
                        [(0, 10), (1, float('nan'))], [(0, 10), (-1, 10)]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                integrate_power(samples)
