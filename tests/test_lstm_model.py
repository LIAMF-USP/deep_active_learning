import tensorflow as tf
import numpy as np

from collections import namedtuple

from model.lstm_model import LSTMModel


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
        lstm_model.add_placeholder()

        x_hat = np.array([[1, 2, 3], [3, 2, 0], [4, 0, 0]])

        expected_embedding = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       [[7, 8, 9], [4, 5, 6], [0, 0, 0]],
                                       [[10, 11, 12], [0, 0, 0], [0, 0, 0]]],
                                      dtype=np.float32)

        embedding_data = lstm_model.add_embedding(x_hat)

        with self.test_session() as sess:
            init = tf.global_variables_initializer()
            init.run()

            feed = lstm_model.create_feed_dict(1.0, 1.0, 1.0)

            expected_shape = expected_embedding.shape
            actual_shape = tf.shape(embedding_data)

            actual_shape = sess.run(actual_shape, feed_dict=feed)

            for expected, actual in zip(expected_shape, actual_shape):
                self.assertEqual(expected, actual)

            embedding_data = sess.run(embedding_data, feed_dict=feed)

            self.assertTrue(np.array_equal(expected_embedding, embedding_data))

    def test_evaluate(self):
        embeddings = [[0, 0, 0],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]]

        batch_labels = np.array([0, 0, 1, 0, 1])
        batch_prediction = np.array([[0.7, 0.1], [0.8, 0.2],
                                     [0.2, 0.8], [0.7, 0.3],
                                     [0.8, 0.2]])

        Config = namedtuple('Config', ['max_length', 'num_units', 'num_classes'])
        config = Config(3, 3, 2)

        lstm_model = LSTMModel(config, embeddings)
        lstm_model.add_placeholder()
        lstm_model.pred = batch_prediction
        lstm_model.add_evaluation_op(batch_labels)

        expected_accuracy = 4.0
        expected_size = 5

        with self.test_session() as sess:
            init = tf.global_variables_initializer()
            init.run()

            actual_accuracy, actual_size = lstm_model.batch_evaluate(sess)

            self.assertEqual(expected_size, actual_size)
            self.assertAlmostEqual(expected_accuracy, actual_accuracy)
