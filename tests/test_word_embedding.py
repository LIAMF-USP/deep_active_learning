import unittest

import numpy as np

from unittest.mock import patch

from word_embedding.word_embedding import GloVeEmbedding


class GloveEmbeddingTest(unittest.TestCase):

    def setUp(self):
        self.word_embedding = GloVeEmbedding(None, None, None, None, None)

    def test_handle_unknown_words(self):
        word_index = {'a': 1, 'b': 2, 'c': 3, '<unk>': 4}
        self.word_embedding.word_index = word_index
        reviews = [('a b c d', 1), ('e f g a', 0), ('3 5 c f', 1)]
        sentence_size = 3

        expected_reviews = [('a b c d', 1), ('<unk> <unk> <unk> a', 0),
                            ('<unk> <unk> c f', 1)]
        actual_reviews = self.word_embedding.handle_unknown_words(
            reviews, sentence_size)

        self.assertEquals(expected_reviews, actual_reviews)

        reviews = [('a b c d', 1), ('e f g a', 0), ('3 5 c f', 1)]
        sentence_size = None

        expected_reviews = [('a b c <unk>', 1), ('<unk> <unk> <unk> a', 0),
                            ('<unk> <unk> c <unk>', 1)]
        actual_reviews = self.word_embedding.handle_unknown_words(
            reviews, sentence_size)

        self.assertEquals(expected_reviews, actual_reviews)

    @patch('word_embedding.word_embedding.GloVeEmbedding.load_embedding')
    @patch('numpy.random.uniform')
    def test_prepare_embedding(self, mock_numpy, mock_embedding):
        self.word_embedding.embed_size = 3

        mock_numpy.return_value = np.array([15, 16, 17])
        mock_index = {'a': 1, 'b': 2, 'e': 3}
        mock_matrix = [[0, 0, 0],
                       [1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]]
        mock_vocab = ['<unk>', 'a', 'b', 'e']
        mock_embedding.return_value = (mock_index, mock_matrix, mock_vocab)

        vocab = [('<unk>', 0), ('a', 1), ('b', 2), ('c', 3)]
        self.word_embedding.word_vocab = vocab

        expected_word_index = {'<unk>': 1, 'a': 2, 'b': 3}
        expected_embedding_matrix = [[0, 0, 0],
                                     [15, 16, 17],
                                     [1, 2, 3],
                                     [4, 5, 6]]
        expected_vocab = ['<unk>', 'a', 'b']

        self.word_embedding.prepare_embedding()

        self.assertEqual(expected_word_index, self.word_embedding.word_index)
        self.assertEqual(expected_embedding_matrix,
                         self.word_embedding.embedding_matrix)
        self.assertEqual(expected_vocab, self.word_embedding.vocab)
