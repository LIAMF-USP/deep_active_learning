import tensorflow as tf


def parser(tfrecord):
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

    return tokens, label


def input_pipeline(tfrecord_files, batch_size, perform_shuffle, num_epochs=1):
    dataset = tf.data.TFRecordDataset(tfrecord_files).map(parser)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_tokens, batch_labels = iterator.get_next()

    return batch_tokens, batch_labels
