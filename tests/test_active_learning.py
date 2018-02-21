import unittest

import numpy as np

from active_learning import get_index


class ActiveLearningTest(unittest.TestCase):

    def test_get_index(self):
        train_labels = np.array([0, 1, 0, 1])
        label = 0
        size = 2

        expected_indexes = [0, 2]
        actual_indexes = get_index(train_labels, label, size)

        for index in expected_indexes:
            self.assertTrue(index in actual_indexes)
