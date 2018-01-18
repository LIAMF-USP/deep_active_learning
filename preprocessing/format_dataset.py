import re

import tensorflow as tf

from word_embedding.word_embedding import UNK_TOKEN


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


def find_and_replace_unknown_words(reviews, word_index, sentence_size, progbar=None):
    processed_reviews = []
    dynamic_sentence_size = False

    if not sentence_size:
        dynamic_sentence_size = True

    for review_index, review in enumerate(reviews):
        words = review.split()

        if dynamic_sentence_size:
            sentence_size = len(words)

        for index, word in enumerate(words[:sentence_size]):
            if word not in word_index:
                words[index] = UNK_TOKEN

        review = ' '.join(words)
        processed_reviews.append(review)

        if progbar:
            progbar.update(review_index + 1, [])

    return processed_reviews


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
