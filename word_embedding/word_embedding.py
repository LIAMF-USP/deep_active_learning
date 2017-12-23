import os
import pickle

import numpy as np


WORD_POS = 0
UNK_TOKEN = '<unk>'
GLOVE_STR = 'glove'


def get_embedding(embedding_path, embed_size, embedding_save_path=None,
                  only_embedding=False, progbar=None):
    word_embedding = None

    if GLOVE_STR in embedding_path:
        word_embedding = GloVeEmbedding(embedding_path, embed_size, embedding_save_path)

    return word_embedding.get_word_embedding(progbar, only_embedding)


class WordEmbedding:

    def __init__(self, embedding_path, embed_size, embedding_save_path):
        self.embedding_path = embedding_path
        self.embed_size = embed_size
        self.save_path = embedding_save_path

    def get_word_embedding(self, progbar=None, only_embedding=False):
        if self.save_path and os.path.exists(self.save_path) and only_embedding:
            with open(self.save_path, 'rb') as embedding_pkl:
                return pickle.load(embedding_pkl)
        else:
            self.load_embedding()

            if only_embedding:
                """
                When running the model, we only need to the embedding matrix. Since
                we are expected to run the model multiple times, it is easier ti pickle
                the embedding matrix, instead of parsing it every time.
                """
                with open(self.save_path, 'wb') as embedding_pkl:
                    pickle.dump(self.embedding_matrix, embedding_pkl)

        return self.word_index, self.embedding_matrix, self.vocab

    def load_embedding(self, progbar=None):
        raise NotImplementedError


class GloVeEmbedding(WordEmbedding):

    def add_unknown_embedding(self):
        self.vocab.append(UNK_TOKEN)
        unknown_embedding = np.random.uniform(low=-1, high=1, size=self.embed_size)
        self.embedding_matrix.append(unknown_embedding.tolist())
        self.word_index[UNK_TOKEN] = len(self.word_index) + 1

    def load_embedding(self, progbar=None):
        self.word_index = dict()
        self.embedding_matrix = []
        self.vocab = []

        # Add an zero row on our glove matrix for padding purposes
        self.embedding_matrix.append([float(0) for _ in range(self.embed_size)])

        with open(self.embedding_path, 'r') as glove_file:
            for index, glove_line in enumerate(glove_file.readlines()):
                glove_features = glove_line.split()
                self.word_index[glove_features[WORD_POS]] = index + 1
                self.vocab.append(glove_features[WORD_POS])
                self.embedding_matrix.append([float(value) for value in glove_features[1:]])

                if progbar:
                    progbar.update(index + 1, [])

        self.add_unknown_embedding()

        return self.word_index, self.embedding_matrix, self.vocab
