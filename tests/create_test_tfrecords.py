import random

import tensorflow as tf


def create_data(num_elements):
    data = []

    for i in range(num_elements):
        example = list(range(1, random.randint(2, 11)))
        random.shuffle(example)

        label = random.randint(0, 1)
        data.append((example, label))

    return data


def make_example(data, label):
    example = tf.train.SequenceExample()

    sentence_size = len(data)
    example.context.feature['size'].int64_list.value.append(sentence_size)
    example.context.feature['label'].int64_list.value.append(label)

    sentence_tokens = example.feature_lists.feature_list['tokens']

    for value in data:
        sentence_tokens.feature.add().int64_list.value.append(int(value))

    return example


def create_tfrecord(data, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)

    for example, label in data:
        example = make_example(example, label)

        writer.write(example.SerializeToString())

    writer.close()


train_data = create_data(50)
validation_data = create_data(10)
test_data = create_data(12)

print(train_data)
print()
print(validation_data)
print()
print(test_data)
print()

create_tfrecord(train_data, 'test_data/train_data.tfrecord')
create_tfrecord(validation_data, 'test_data/validation_data.tfrecord')
create_tfrecord(test_data, 'test_data/test_data.tfrecord')
