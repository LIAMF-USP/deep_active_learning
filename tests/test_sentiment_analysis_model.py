import numpy as np
import tensorflow as tf

from unittest.mock import MagicMock, PropertyMock

from model.sentiment_analysis_model import SentimentAnalysisModel


class SentimentAnalysisModelTest(tf.test.TestCase):

    def test_evaluate(self):
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

        sentiment_model = SentimentAnalysisModel(None)
        type(sentiment_model).pred = PropertyMock(side_effect=prediction_values)

        expected_accuracy = 0.4

        with self.test_session():
            actual_accuracy = sentiment_model.evaluate(dataset_mock)
            self.assertAlmostEqual(expected_accuracy, actual_accuracy.eval())
