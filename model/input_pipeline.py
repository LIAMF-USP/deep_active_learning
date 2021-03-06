import tensorflow as tf


class SentimentAnalysisDataset:

    def __init__(self, data, batch_size, perform_shuffle, bucket_width, num_buckets):
        self.data = data
        self.batch_size = batch_size
        self.perform_shuffle = perform_shuffle
        self.bucket_width = bucket_width
        self.num_buckets = num_buckets

    def parser(self, tfrecord):
        context_features = {
            'size': tf.FixedLenFeature([], dtype=tf.int64),
            'label': tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        tfrecord_parsed = tf.parse_single_sequence_example(
            tfrecord, context_features, sequence_features)

        tokens = tfrecord_parsed[1]['tokens']
        label = tfrecord_parsed[0]['label']
        size = tfrecord_parsed[0]['size']

        return tokens, label, size

    def init_dataset(self):
        sentiment_dataset = tf.data.TFRecordDataset(self.data)
        sentiment_dataset = sentiment_dataset.cache()
        sentiment_dataset = sentiment_dataset.map(self.parser, num_parallel_calls=32)

        return sentiment_dataset

    def create_dataset(self):
        sentiment_dataset = self.init_dataset()

        if self.perform_shuffle:
            sentiment_dataset = sentiment_dataset.shuffle(buffer_size=self.batch_size * 2)

        def batching_func(dataset):
            return dataset.padded_batch(
                    self.batch_size,
                    padded_shapes=(
                        tf.TensorShape([None]),  # token
                        tf.TensorShape([]),  # label
                        tf.TensorShape([]))  # size
                    )

        def key_func(tokens, label, size):
            bucket_id = size // self.bucket_width

            return tf.to_int64(tf.minimum(bucket_id, self.num_buckets))

        def reduce_func(bucket_key, widowed_data):
            return batching_func(widowed_data)

        sentiment_dataset = sentiment_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=self.batch_size))

        self.sentiment_dataset = sentiment_dataset.prefetch(4)

        return self.sentiment_dataset


class NumpyDataset(SentimentAnalysisDataset):
    def init_dataset(self):
        """
        The data for this function is three distict numpy arrays, containing
        the movie reviews word-ids, labels and sizes.

        It will them create a dataset based on this array.
        """

        movie_reviews, labels, sizes = self.data

        def review_iterator():
            for review, label, size in zip(movie_reviews, labels, sizes):
                yield (review, label, size)

        dataset = tf.data.Dataset.from_generator(
            lambda: review_iterator(), (tf.int64, tf.int64, tf.int64),
            (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
        )

        return dataset


class InputPipeline:

    def __init__(self, train_files, validation_files, test_files, batch_size, perform_shuffle,
                 bucket_width, num_buckets):
        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.perform_shuffle = perform_shuffle
        self.bucket_width = bucket_width
        self.num_buckets = num_buckets

        self._train_iterator_op = None
        self._validation_iterator_op = None
        self._test_iterator_op = None

        self.train_batches = 0
        self.validation_batches = 0
        self.test_batches = 0

    @property
    def train_iterator(self):
        return self._train_iterator_op

    @property
    def validation_iterator(self):
        return self._validation_iterator_op

    @property
    def test_iterator(self):
        return self._test_iterator_op

    def create_datasets(self, dataset=SentimentAnalysisDataset):
        train_dataset = dataset(
            self.train_files, self.batch_size, self.perform_shuffle,
            self.bucket_width, self.num_buckets)
        validation_dataset = dataset(
            self.validation_files, self.batch_size, False,
            self.bucket_width, self.num_buckets)
        test_dataset = dataset(
            self.test_files, self.batch_size, False,
            self.bucket_width, self.num_buckets)

        self.train_dataset = train_dataset.create_dataset()
        self.validation_dataset = validation_dataset.create_dataset()
        self.test_dataset = test_dataset.create_dataset()

    def create_iterator(self):
        self._train_iterator_op = self.train_dataset.make_initializable_iterator()
        self._validation_iterator_op = self.validation_dataset.make_initializable_iterator()
        self._test_iterator_op = self.test_dataset.make_initializable_iterator()

    def get_num_batches(self, iterator):
        with tf.Session() as sess:
            num_batches = 0
            sess.run(iterator.initializer)
            next_batch = iterator.get_next()

            while True:
                try:
                    _, _, _ = sess.run(next_batch)
                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

        return num_batches

    def get_datasets_num_batches(self):
        self.train_batches = self.get_num_batches(self.train_iterator)
        self.validation_batches = self.get_num_batches(self.validation_iterator)
        self.test_batches = self.get_num_batches(self.test_iterator)

    def build_pipeline(self):
        self.create_datasets()
        self.create_iterator()


class ALInputPipeline(InputPipeline):

    def create_datasets(self, dataset=NumpyDataset):
        super().create_datasets(dataset)
