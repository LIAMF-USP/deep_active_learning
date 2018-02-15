import unittest

from preprocessing.dataset import MovieReviewDataset


class PreprocessingDatasetTest(unittest.TestCase):

    def test_create_validation_set(self):
        movie_review_dataset = MovieReviewDataset(
            train_save_path=None,
            validation_save_path=None,
            test_save_path=None,
            data_dir=None,
            data_output_dir=None,
            output_dir=None,
            embedding_file=None,
            embed_size=None,
            embedding_path=None,
            embedding_wordindex_path=None,
            sentence_size=None)
        all_reviews = [(value, 0) for value in range(40)]

        all_reviews, validation_reviews = movie_review_dataset.create_validation_set(all_reviews)

        expected_review_len = 36
        expected_val_len = 4

        self.assertEqual(expected_review_len, len(all_reviews))
        self.assertEqual(expected_val_len, len(validation_reviews))

        for validation_review in validation_reviews:
            self.assertTrue(validation_review not in all_reviews)
