import unittest

from utils.metrics import variation_ratio


class MetricTest(unittest.TestCase):

    def test_variation_ration(self):
        mc_counts = [([10, 0], 0), ([0, 10], 1)]
        expected_variation_ratio = [0, 0]
        self.assertEqual(variation_ratio(mc_counts), expected_variation_ratio)

        mc_counts = [([5, 5], 0), ([5, 5], 1)]
        expected_variation_ratio = [0.5, 0.5]
        self.assertEqual(variation_ratio(mc_counts), expected_variation_ratio)
