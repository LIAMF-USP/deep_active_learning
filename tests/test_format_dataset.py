import unittest

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_special_characters_from_text)


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
