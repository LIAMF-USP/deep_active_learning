import unittest

import numpy as np

from utils.metrics import variation_ratio, entropy, bald


class MetricTest(unittest.TestCase):

    def test_variation_ration(self):
        mc_counts = [([10, 0], 0), ([0, 10], 1)]
        expected_variation_ratio = [0, 0]
        self.assertEqual(variation_ratio(mc_counts), expected_variation_ratio)

        mc_counts = [([5, 5], 0), ([5, 5], 1)]
        expected_variation_ratio = [0.5, 0.5]
        self.assertEqual(variation_ratio(mc_counts), expected_variation_ratio)

    def test_entropy(self):
        predictions = np.array([[3, 0], [3, 0], [3, 0]])
        num_samples = 3

        expected_entropy = np.array([0, 0, 0])
        np.testing.assert_array_equal(expected_entropy, entropy(predictions, num_samples))

        predictions = np.array([[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]])
        num_samples = 3

        expected_entropy = np.array([1, 1, 1])
        np.testing.assert_array_equal(expected_entropy, entropy(predictions, num_samples))

    def test_bald(self):
        predictions = np.array([[3, 0], [3, 0], [3, 0]])
        all_entropy_dropout = np.array([0, 0, 0])
        num_samples = 3

        expected_bald = np.array([0, 0, 0])
        np.testing.assert_array_equal(
            expected_bald, bald(predictions, all_entropy_dropout, num_samples))

        predictions = np.array([[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]])
        all_entropy_dropout = np.array([3, 3, 3])
        num_samples = 3

        expected_bald = np.array([0, 0, 0])
        np.testing.assert_array_equal(
            expected_bald, bald(predictions, all_entropy_dropout, num_samples))
