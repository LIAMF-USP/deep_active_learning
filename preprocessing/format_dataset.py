import os
import pickle
import re

import numpy as np
import tensorflow as tf


WORD_POS = 0
UNK_TOKEN = '<unk>'


class SentenceTFRecord():

    def __init__(self, reviews, output_path, progbar=None):
        self.reviews = reviews
        self.output_path = output_path
        self.progbar = progbar

    def parse_sentences(self):
        writer = tf.python_io.TFRecordWriter(self.output_path)

        for index, (sentence, label) in enumerate(self.reviews):
            example = self.make_example(sentence, label)

            writer.write(example.SerializeToString())

            if self.progbar:
                self.progbar.update(index + 1, [])

        writer.close()

    def make_example(self, sentence, label):
        example = tf.train.SequenceExample()

        sentence_size = len(sentence)
        example.context.feature['size'].int64_list.value.append(sentence_size)
        example.context.feature['label'].int64_list.value.append(label)

        sentence_tokens = example.feature_lists.feature_list['tokens']

        for token in sentence:
            sentence_tokens.feature.add().int64_list.value.append(int(token))

        return example


def sentence_to_id_list(sentence, vocabulary_processor):
    if type(sentence) is not list:
        sentence = [sentence]

    return vocabulary_processor.transform(sentence)


def create_vocab_parser(vocab, sentence_size):
    """
    The tensorflow method has a default tokenizer that
    removes some word from the glove vocab files, this function
    guarantees that no word is removed from it.
    """
    def tokenizer2(text):
        for value in text:
            yield value.split()

    vocabulary_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=sentence_size, tokenizer_fn=tokenizer2)
    vocabulary_processor.fit(vocab)

    return vocabulary_processor


def get_glove_matrix(save_path, glove_path, embed_size, progbar=None):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as glove_pkl:
            glove_matrix = pickle.load(glove_pkl)
    else:
        _, glove_matrix, _ = load_glove(glove_path, embed_size)

        with open(save_path, 'wb') as glove_pkl:
            pickle.dump(glove_matrix, glove_pkl)

    return glove_matrix


def find_and_replace_unknown_words(reviews, word_index, sentence_size, progbar=None):
    processed_reviews = []

    for review_index, review in enumerate(reviews):
        words = review.split()

        for index, word in enumerate(words[:sentence_size]):
            if word not in word_index:
                words[index] = UNK_TOKEN

        review = ' '.join(words)
        processed_reviews.append(review)

        if progbar:
            progbar.update(review_index + 1, [])

    return processed_reviews


def add_unknown_embedding(word_index, glove_matrix, vocab, embed_size):
    vocab.append(UNK_TOKEN)
    unknown_embedding = np.random.uniform(low=-1, high=1, size=embed_size)
    glove_matrix.append(unknown_embedding.tolist())
    word_index[UNK_TOKEN] = len(word_index) + 1


def load_glove(glove_path, embed_size, progbar=None):
    word_index = dict()
    glove_matrix = []
    vocab = []

    # Add an zero row on our glove matrix for padding purposes
    glove_matrix.append([float(0) for _ in range(embed_size)])

    with open(glove_path, 'r') as glove_file:
        for index, glove_line in enumerate(glove_file.readlines()):
            glove_features = glove_line.split()
            word_index[glove_features[WORD_POS]] = index + 1
            vocab.append(glove_features[WORD_POS])
            glove_matrix.append([float(value) for value in glove_features[1:]])

            if progbar:
                progbar.update(index + 1, [])

    add_unknown_embedding(word_index, glove_matrix, vocab, embed_size)

    return word_index, glove_matrix, vocab


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
    text = re.sub(r"[^A-Za-z0-9(),!?;-]", ' ', text)
    return re.sub(r'\s{2,}', ' ', text)


def to_lower(review_text):
    return review_text.lower()
