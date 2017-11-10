import unittest

from preprocess_dataset import create_validation_set


class PreprocessingDatasetTest(unittest.TestCase):

    def test_create_validation_set(self):
        test_pos = list(range(20))
        test_neg = list(range(20, 40))

        test_pos, test_neg, test_val_pos, test_val_neg = create_validation_set(
            test_pos, test_neg)

        expected_pos_len = 18
        expected_neg_len = 18

        self.assertEqual(expected_pos_len, len(test_pos))
        self.assertEqual(expected_neg_len, len(test_neg))

        expected_val_pos_len = 2
        expected_val_neg_len = 2

        self.assertEqual(expected_val_pos_len, len(test_val_pos))
        self.assertEqual(expected_val_neg_len, len(test_val_neg))

        for val_pos, val_neg in zip(test_val_pos, test_val_neg):
            self.assertTrue(val_pos not in test_pos)
            self.assertTrue(val_neg not in test_neg)
