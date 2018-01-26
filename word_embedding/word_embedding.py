import fastText
import os
import pickle
import gensim

import numpy as np


WORD_POS = 0
UNK_TOKEN = '<unk>'
GLOVE_STR = 'glove'
FASTTEXT_STR = 'wiki'
WORD2VEC_STR = 'Google'


def get_embedding(embedding_path, embed_size, vocab, embedding_save_path=None,
                  word_index_save_path=None, progbar=None):
    word_embedding = None

    if GLOVE_STR in embedding_path:
        word_embedding = GloVeEmbedding(embedding_path, embed_size, vocab,
                                        embedding_save_path, word_index_save_path)
    elif FASTTEXT_STR in embedding_path:
        word_embedding = FastTextEmbedding(embedding_path, embed_size, vocab,
                                           embedding_save_path, word_index_save_path)
    elif WORD2VEC_STR in embedding_path:
        word_embedding = Word2VecEmbedding(embedding_path, embed_size, vocab,
                                           embedding_save_path, word_index_save_path)

    return word_embedding


class WordEmbedding:

    def __init__(self, embedding_path, embed_size, word_vocab,
                 embedding_save_path, word_index_save_path):
        self.embedding_path = embedding_path
        self.embed_size = embed_size
        self.word_vocab = word_vocab

        self.embedding_save_path = embedding_save_path
        self.word_index_save_path = word_index_save_path

        self.word_index = dict()
        self.embedding_matrix = []
        self.vocab = []

    def get_word_embedding(self, progbar=None, should_save=True, verbose=True):
        if self.embedding_save_path and os.path.exists(self.embedding_save_path):
            if verbose:
                print('Loading saved matrix embeddings...')

            with open(self.embedding_save_path, 'rb') as embedding_pkl:
                self.embedding_matrix = pickle.load(embedding_pkl)

            self.word_index = None
            if self.word_index_save_path and os.path.exists(self.word_index_save_path):
                with open(self.word_index_save_path, 'rb') as word_index_pkl:
                    self.word_index = pickle.load(word_index_pkl)

            self.vocab = None
        else:
            if verbose:
                print('Constructing embeddings...')

            self.prepare_embedding()

            if should_save:
                with open(self.embedding_save_path, 'wb') as embedding_pkl:
                    pickle.dump(self.embedding_matrix, embedding_pkl)

                with open(self.word_index_save_path, 'wb') as embedding_pkl:
                    pickle.dump(self.word_index, embedding_pkl)

        return self.word_index, self.embedding_matrix, self.vocab

    def add_zero_rows(self, embedding_matrix):
        # Add an zero row on to the matrix for padding purposes
        embedding_matrix.append([float(0) for _ in range(self.embed_size)])

    def add_unknown_embedding(self):
        self.vocab.append(UNK_TOKEN)
        unknown_embedding = np.random.uniform(low=-1, high=1, size=self.embed_size)
        self.embedding_matrix.append(unknown_embedding.tolist())
        self.word_index[UNK_TOKEN] = len(self.word_index) + 1

    def handle_unknown_words(self, reviews, sentence_size, progbar=None):
        processed_reviews = []
        dynamic_sentence_size = False

        if not sentence_size:
            dynamic_sentence_size = True

        for review_index, (review, label) in enumerate(reviews):
            words = review.split()

            if dynamic_sentence_size:
                sentence_size = len(words)

            for index, word in enumerate(words[:sentence_size]):
                if word not in self.word_index:
                    words[index] = UNK_TOKEN

            review = ' '.join(words)
            processed_reviews.append((review, label))

            if progbar:
                progbar.update(review_index + 1, [])

        return processed_reviews

    def prepare_embedding(self):
        word_index, matrix, embedding_vocab = self.load_embedding()

        self.add_zero_rows(self.embedding_matrix)
        self.add_unknown_embedding()

        for word, index in self.word_vocab[1:]:
            is_added = self.add_word_embedding(matrix, word_index, word)

            if is_added:
                self.word_index[word] = len(self.word_index) + 1
                self.vocab.append(word)

        return self.word_index, self.embedding_matrix, self.vocab

    def load_embedding(self):
        raise NotImplementedError

    def add_word_embedding(self, matrix, word_index, word):
        raise NotImplementedError


class GloVeEmbedding(WordEmbedding):

    def load_embedding(self, progbar=None):
        word_index = dict()
        embedding_matrix = []
        vocab = []

        self.add_zero_rows(embedding_matrix)

        with open(self.embedding_path, 'r') as glove_file:
            for index, glove_line in enumerate(glove_file.readlines()):
                glove_features = glove_line.split()
                word_index[glove_features[WORD_POS]] = index + 1
                vocab.append(glove_features[WORD_POS])
                embedding_matrix.append([float(value) for value in glove_features[1:]])

                if progbar:
                    progbar.update(index + 1, [])

        return word_index, embedding_matrix, vocab

    def add_word_embedding(self, matrix, word_index, word):
        if word in word_index:
            self.embedding_matrix.append(
                matrix[word_index[word]])

            return True

        return False


class FastTextEmbedding(WordEmbedding):

    def load_embedding(self, progbar=None):
        self.fasttext_model = fastText.load_model(self.embedding_path)

        return None, None, None

    def add_word_embedding(self, matrix, word_index, word):
        self.embedding_matrix.append(
            self.fasttext_model.get_word_vector(word).tolist())

        return True


class Word2VecEmbedding(WordEmbedding):

    def load_embedding(self, progbar=None):
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            self.embedding_path, binary=True)

        return None, None, None

    def add_word_embedding(self, matrix, word_index, word):
        if word in self.word2vec_model:
            self.embedding_matrix.append(
                self.word2vec_model[word].tolist())

            return True

        return False
