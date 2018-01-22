import unittest

from word_embedding.word_embedding import GloVeEmbedding


class GloveEmbeddingTest(unittest.TestCase):

    def setUp(self):
        self.word_embedding = GloVeEmbedding(None, None, None, None, None)

    def test_handle_unknown_words(self):
        word_index = {'a': 1, 'b': 2, 'c': 3, '<unk>': 4}
        self.word_embedding.word_index = word_index
        reviews = ['a b c d', 'e f g a', '3 5 c f']
        sentence_size = 3

        expected_reviews = ['a b c d', '<unk> <unk> <unk> a', '<unk> <unk> c f']
        actual_reviews = self.word_embedding.handle_unknown_words(
            reviews, sentence_size)

        self.assertEquals(expected_reviews, actual_reviews)

        reviews = ['a b c d', 'e f g a', '3 5 c f']
        sentence_size = None

        expected_reviews = ['a b c <unk>', '<unk> <unk> <unk> a', '<unk> <unk> c <unk>']
        actual_reviews = self.word_embedding.handle_unknown_words(
            reviews, sentence_size)

        self.assertEquals(expected_reviews, actual_reviews)
