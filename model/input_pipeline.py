import tensorflow as tf


class InputPipeline:

    def __init__(self, train_files, validation_files,
                 test_files, batch_size, perform_shuffle):
        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.perform_shuffle = perform_shuffle

        self._train_iterator_op = None
        self._validation_iterator_op = None
        self._test_iterator_op = None

    @property
    def train_iterator(self):
        return self._train_iterator_op

    @property
    def validation_iterator(self):
        return self._validation_iterator_op

    @property
    def test_iterator(self):
        return self._test_iterator_op

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

    def create_datasets(self):
        train_dataset = tf.data.TFRecordDataset(self.train_files).map(self.parser)
        validation_dataset = tf.data.TFRecordDataset(self.validation_files).map(self.parser)
        test_dataset = tf.data.TFRecordDataset(self.test_files).map(self.parser)

        if self.perform_shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=self.batch_size * 2)
            validation_dataset = validation_dataset.shuffle(buffer_size=self.batch_size * 2)
            test_dataset = test_dataset.shuffle(buffer_size=self.batch_size * 2)

        self.train_dataset = train_dataset.batch(self.batch_size)
        self.validation_dataset = validation_dataset.batch(self.batch_size)
        self.test_dataset = test_dataset.batch(self.batch_size)

    def create_iterator(self):
        self._iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes)

        self._train_iterator_op = self._iterator.make_initializer(self.train_dataset)
        self._validation_iterator_op = self._iterator.make_initializer(self.validation_dataset)
        self._test_iterator_op = self._iterator.make_initializer(self.test_dataset)

    def make_batch(self):
        tokens_batch, labels_batch, size_batch = self._iterator.get_next()

        return tokens_batch, labels_batch, size_batch

    def build_pipeline(self):
        self.create_datasets()
        self.create_iterator()
