import unittest

import numpy as np
import tensorflow as tf

from unittest.mock import MagicMock, patch

from model.sentiment_analysis_model import SentimentAnalysisModel


class SentimentAnalysisModelTest(unittest.TestCase):

    @patch('model.model.Model.predict_on_batch')
    def test_evaluate(self, predict_mock):
        labels_values = [
                (None, np.array([0, 1])),
                (None, np.array([1, 1])),
                (None, np.array([0])),
                tf.errors.OutOfRangeError(None, None, 'test')]

        dataset_mock = MagicMock()
        dataset_mock.get_batch.side_effect = labels_values

        prediction_values = [
                np.array([[0.7, 0.1], [0.8, 0.2]]),
                np.array([[0.2, 0.8], [0.7, 0.3]]),
                np.array([[0.2, 0.8]])]

        predict_mock.side_effect = prediction_values

        sentiment_model = SentimentAnalysisModel(None)

        expected_accuracy = 0.4
        actual_accuracy = sentiment_model.evaluate(None, dataset_mock)

        self.assertEqual(expected_accuracy, actual_accuracy)
