import unittest

from preprocess_dataset import create_validation_set


class PreprocessingDatasetTest(unittest.TestCase):

    def test_create_validation_set(self):
        all_reviews = [(value, 0) for value in range(40)]

        all_reviews, validation_reviews = create_validation_set(all_reviews)

        expected_review_len = 36
        expected_val_len = 4

        self.assertEqual(expected_review_len, len(all_reviews))
        self.assertEqual(expected_val_len, len(validation_reviews))

        for validation_review in validation_reviews:
            self.assertTrue(validation_review not in all_reviews)
