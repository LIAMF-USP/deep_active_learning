import tensorflow as tf
import numpy as np

from collections import namedtuple

from model.lstm_model import LSTMModel, sequence_length


class LSTMModelTest(tf.test.TestCase):

    def test_add_embedding(self):
        embeddings = [[0, 0, 0],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]]

        Config = namedtuple('Config', ['max_length', 'num_units', 'num_classes'])
        config = Config(3, 3, 2)

        lstm_model = LSTMModel(config, embeddings)

        x_hat = np.array([[1, 2, 3], [3, 2, 0], [4, 0, 0]])
        lstm_model.data_placeholder = x_hat

        expected_embedding = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       [[7, 8, 9], [4, 5, 6], [0, 0, 0]],
                                       [[10, 11, 12], [0, 0, 0], [0, 0, 0]]],
                                      dtype=np.float32)

        embedding_data = lstm_model.add_embedding()

        with self.test_session():
            init = tf.global_variables_initializer()
            init.run()

            expected_shape = expected_embedding.shape
            actual_shape = tf.shape(embedding_data).eval()

            for expected, actual in zip(expected_shape, actual_shape):
                self.assertEqual(expected, actual)

            self.assertTrue(np.array_equal(expected_embedding, embedding_data.eval()))

    def test_sequence_length(self):
        embeddings = [[0, 0, 0],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]]

        Config = namedtuple('Config', ['max_length', 'num_units', 'num_classes'])
        config = Config(3, 3, 2)

        lstm_model = LSTMModel(config, embeddings)

        x_hat = np.array([[1, 2, 3], [3, 2, 0], [4, 0, 0]])
        lstm_model.data_placeholder = x_hat

        embedding_data = lstm_model.add_embedding()

        expected_length = np.array([3, 2, 1])
        actual_length = sequence_length(embedding_data)

        with self.test_session():
            init = tf.global_variables_initializer()
            init.run()

            expected_shape = expected_length.shape
            actual_shape = tf.shape(actual_length).eval()

            for expected, actual in zip(expected_shape, actual_shape):
                self.assertEqual(expected, actual)

            self.assertTrue(np.array_equal(expected_length, actual_length.eval()))
