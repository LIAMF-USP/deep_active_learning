import re

import tensorflow as tf


WORD_POS = 0


def create_vocab_parser(vocab, sentence_size):
    """
    The tensorflow method has a default tokenizer that
    removes some word from the glove vocab files, this function
    guarantees that no word is removed from it.
    """
    def tokenizer(text):
        for value in text:
            yield value.split()

    return tf.contrib.learn.preprocessing.VocabularyParser(
        max_line_length=sentence_size, tokenizer_fn=tokenizer)


def load_glove(glove_path, progbar=None):
    word_index = dict()
    glove_matrix = []
    vocab = []

    with open(glove_path, 'r') as glove_file:
        for index, glove_line in enumerate(glove_file.readlines()):
            glove_features = glove_line.split()
            word_index[glove_features[WORD_POS]] = index + 1
            vocab.append(glove_features[WORD_POS])
            glove_matrix.append([float(value) for value in glove_features[1:]])

            if progbar:
                progbar.update(index + 1, [])

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
    return re.sub(r"\?", " ? ", text)


def remove_html_from_text(text):
    return re.sub(r'<br\s/><br\s/>', ' ', text)


def remove_special_characters_from_text(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`;-]", ' ', text)
    return re.sub(r'\s{2,}', ' ', text)


def to_lower(review_text):
    return review_text.lower()
