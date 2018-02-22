import unittest

import numpy as np

from model.model_manager import ActiveLearningModelManager


class ActiveLearningTest(unittest.TestCase):

    def test_get_index(self):
        train_labels = np.array([0, 1, 0, 1])
        label = 0
        size = 2

        al_model_manager = ActiveLearningModelManager(None, None)

        expected_indexes = [0, 2]
        actual_indexes = al_model_manager.get_index(train_labels, label, size)

        for index in expected_indexes:
            self.assertTrue(index in actual_indexes)
