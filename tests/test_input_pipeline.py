import numpy as np
import tensorflow as tf

from model.input_pipeline import InputPipeline, NumpyDataset


class InputPipelineTests(tf.test.TestCase):

    def test_input_pipeline(self):
        train_files = 'tests/test_data/train_data.tfrecord'
        validation_files = 'tests/test_data/validation_data.tfrecord'
        test_files = 'tests/test_data/test_data.tfrecord'
        batch_size = 2
        perform_shuffle = True
        bucket_width = 10
        num_buckets = 10

        input_pipeline = InputPipeline(
            train_files, validation_files, test_files, batch_size, perform_shuffle,
            bucket_width, num_buckets)

        input_pipeline.build_pipeline()

        with self.test_session():
            train_iterator = input_pipeline.train_iterator
            train_iterator.initializer.run()
            num_batches = 0
            expected_num_batches = 25

            while True:
                try:
                    tokens, labels, _ = train_iterator.get_next()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

            validation_iterator = input_pipeline.validation_iterator
            validation_iterator.initializer.run()
            num_batches = 0
            expected_num_batches = 6

            while True:
                try:
                    tokens, labels, _ = validation_iterator.get_next()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

            test_iterator = input_pipeline.test_iterator
            test_iterator.initializer.run()
            num_batches = 0
            expected_num_batches = 7

            while True:
                try:
                    tokens, labels, _ = test_iterator.get_next()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

            train_iterator.initializer.run()
            num_batches = 0
            expected_num_batches = 25

            while True:
                try:
                    tokens, labels, _ = train_iterator.get_next()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

    def test_numpy_dataset(self):
        reviews = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8],
            [9, 10, 11, 12, 13],
            [14, 15]]
        )
        labels = np.array([1, 0, 1, 1])
        sizes = np.array([10, 20, 30, 42])

        data = (reviews, labels, sizes)

        dataset = NumpyDataset(data, batch_size=1, perform_shuffle=False,
                               bucket_width=1, num_buckets=1)
        numpy_dataset = dataset.create_dataset()

        with tf.Session() as sess:
            index = 0
            value = numpy_dataset.make_one_shot_iterator()

            while True:
                try:
                    review, label, size = sess.run(value.get_next())

                    self.assertEqual(review.tolist()[0], reviews[index])
                    self.assertEqual(label, labels[index])
                    self.assertEqual(size, sizes[index])

                    index += 1
                except tf.errors.OutOfRangeError:
                    break
