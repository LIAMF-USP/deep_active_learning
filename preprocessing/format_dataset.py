import re

import tensorflow as tf

from tensorflow.contrib import learn


class SentenceTFRecord():
    def __init__(self, reviews, output_path, progbar=None):
        self.reviews = reviews
        self.output_path = output_path
        self.progbar = progbar

    def parse_sentences(self):
        writer = tf.python_io.TFRecordWriter(self.output_path)

        for index, (sentence, label, size) in enumerate(self.reviews):
            example = self.make_example(sentence, label, size)

            writer.write(example.SerializeToString())

            if self.progbar:
                self.progbar.update(index + 1, [])

        writer.close()

    def make_example(self, sentence, label, size):
        example = tf.train.SequenceExample()

        example.context.feature['size'].int64_list.value.append(size)
        example.context.feature['label'].int64_list.value.append(label)

        sentence_tokens = example.feature_lists.feature_list['tokens']

        for token in sentence:
            sentence_tokens.feature.add().int64_list.value.append(int(token))

        return example


def sentence_to_id_list(sentence, word_index):
    return [word_index[word] for word in sentence.split()]


def add_space_between_characters(text):
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'m", " \'m", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"-", " - ", text)
    return re.sub(r"\?", " ? ", text)


def remove_url_from_text(text):
    return re.sub(r'https?\S+', ' ', text)


def remove_html_from_text(text):
    return re.sub(r'<br\s/><br\s/>', ' ', text)


def create_unique_apostrophe(text):
    return re.sub(r"\`", "\'", text)


def remove_special_characters_from_text(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\']", ' ', text)
    return re.sub(r'\s{2,}', ' ', text)


def to_lower(review_text):
    return review_text.lower()


def get_maximum_size_review(reviews_array):
    max_size = -1

    for review in reviews_array:
        review = review.split()
        if len(review) > max_size:
            max_size = len(review)

    return max_size


def get_vocab(reviews_array):
    max_size = get_maximum_size_review(reviews_array)

    vocabulary_processor = learn.preprocessing.VocabularyProcessor(max_size)
    vocabulary_processor.fit(reviews_array)

    vocab = vocabulary_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    return sorted_vocab
