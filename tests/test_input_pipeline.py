import tensorflow as tf

from model.input_pipeline import InputPipeline


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
            input_pipeline.train_iterator.run()
            num_batches = 0
            expected_num_batches = 25

            while True:
                try:
                    tokens, labels, _ = input_pipeline.make_batch()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

            input_pipeline.validation_iterator.run()
            num_batches = 0
            expected_num_batches = 6

            while True:
                try:
                    tokens, labels, _ = input_pipeline.make_batch()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

            input_pipeline.test_iterator.run()
            num_batches = 0
            expected_num_batches = 7

            while True:
                try:
                    tokens, labels, _ = input_pipeline.make_batch()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)

            input_pipeline.train_iterator.run()
            num_batches = 0
            expected_num_batches = 25

            while True:
                try:
                    tokens, labels, _ = input_pipeline.make_batch()
                    tokens.eval()

                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            self.assertEqual(num_batches, expected_num_batches)
