import tensorflow as tf
import numpy as np

from collections import namedtuple

from model.recurrent_model import RecurrentModel


class RecurrentModelTest(tf.test.TestCase):

    def test_add_embedding(self):
        embeddings = [[0, 0, 0],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]]

        Config = namedtuple('Config', ['num_units', 'num_classes', 'embed_size'])
        config = Config(3, 2, 3)

        recurrent_model = RecurrentModel(config, embeddings, verbose=False)
        recurrent_model.add_placeholder()

        x_hat = np.array([[1, 2, 3], [3, 2, 0], [4, 0, 0]])

        expected_embedding = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       [[7, 8, 9], [4, 5, 6], [0, 0, 0]],
                                       [[10, 11, 12], [0, 0, 0], [0, 0, 0]]],
                                      dtype=np.float32)

        embedding_data = recurrent_model.add_embedding(x_hat)

        with self.test_session() as sess:
            init = tf.global_variables_initializer()
            init.run()

            feed = recurrent_model.create_feed_dict(1.0, 1.0, 1.0, 1.0)

            expected_shape = expected_embedding.shape
            actual_shape = tf.shape(embedding_data)

            actual_shape = sess.run(actual_shape, feed_dict=feed)

            for expected, actual in zip(expected_shape, actual_shape):
                self.assertEqual(expected, actual)

            recurrent_model.build_embedding_init()
            recurrent_model.initialize_embeddings(sess)

            embedding_data = sess.run(embedding_data, feed_dict=feed)

            np.testing.assert_array_equal(expected_embedding, embedding_data)
