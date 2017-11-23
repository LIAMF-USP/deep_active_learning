import tensorflow as tf
import numpy as np

from model.lstm_model import LSTMModel


class LSTMModelTest(tf.test.TestCase):

    def test_add_embedding(self):
        embeddings = [[0, 0, 0],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]]

        lstm_model = LSTMModel(None, embeddings)

        x_hat = np.array([[1, 2, 3], [3, 2, 0], [4, 0, 0]])
        lstm_model.batch_data = x_hat

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
