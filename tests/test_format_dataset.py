import unittest

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_special_characters_from_text,
                                          load_glove)


class FormatDatasetTest(unittest.TestCase):

    def test_remove_html_from_text(self):
        text_string1 = 'This is my review.<br /><br />I liked the movie.'
        expected_string1 = 'This is my review. I liked the movie.'
        actual_string1 = remove_html_from_text(text_string1)

        self.assertEqual(expected_string1, actual_string1)

        text_string2 = ('This is my review.<br /><br />I liked the movie.<br /><br />' +
                        'I will watch it again')
        expected_string2 = 'This is my review. I liked the movie. I will watch it again'
        actual_string2 = remove_html_from_text(text_string2)

        self.assertEqual(expected_string2, actual_string2)

    def test_remove_special_characters_from_text(self):
        text_string1 = 'This movie is awesome!!!it is like a "poem"'
        expected_string1 = 'This movie is awesome!!!it is like a poem '
        actual_string1 = remove_special_characters_from_text(text_string1)

        self.assertEqual(expected_string1, actual_string1)

        text_string2 = 'I love this movie, but I ask, why?'
        expected_string2 = 'I love this movie, but I ask, why?'
        actual_string2 = remove_special_characters_from_text(text_string2)

        self.assertEqual(expected_string2, actual_string2)

        text_string3 = 'I love this movie ;-)'
        expected_string3 = 'I love this movie ;-)'
        actual_string3 = remove_special_characters_from_text(text_string3)

        self.assertEqual(expected_string3, actual_string3)

    def test_load_glove(self):
        glove_path = 'tests/test_data/glove_test_data.txt'

        expected_word_index = {'a': 1, 'b': 2, 'c': 3}
        expected_glove_matrix = [[0.1, 0.2, 0.3],
                                 [1, 2, 3],
                                 [4, 5, 6]]
        expected_vocab = ['a', 'b', 'c']

        actual_word_index, actual_glove_matrix, actual_vocab = load_glove(glove_path)

        self.assertEqual(expected_word_index, actual_word_index)
        self.assertEqual(expected_glove_matrix, actual_glove_matrix)
        self.assertEqual(expected_vocab, actual_vocab)
