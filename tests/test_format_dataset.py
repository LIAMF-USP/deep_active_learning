import unittest

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_special_characters_from_text,
                                          add_space_between_characters,
                                          create_vocab_parser,
                                          to_lower,
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

    def test_add_space_between_characters(self):
        text_string1 = "The director's cut is excellent."
        expected_string1 = "The director 's cut is excellent."
        actual_string1 = add_space_between_characters(text_string1)

        self.assertEqual(expected_string1, actual_string1)

        text_string2 = "I'm really interested in this movie."
        expected_string2 = "I 'm really interested in this movie."
        actual_string2 = add_space_between_characters(text_string2)

        self.assertEqual(expected_string2, actual_string2)

        text_string3 = "They've have done a bad movie."
        expected_string3 = "They 've have done a bad movie."
        actual_string3 = add_space_between_characters(text_string3)

        self.assertEqual(expected_string3, actual_string3)

        text_string4 = "I don't like this movie"
        expected_string4 = "I do n't like this movie"
        actual_string4 = add_space_between_characters(text_string4)

        self.assertEqual(expected_string4, actual_string4)

        text_string5 = "They're a great team"
        expected_string5 = "They 're a great team"
        actual_string5 = add_space_between_characters(text_string5)

        self.assertEqual(expected_string5, actual_string5)

        text_string6 = "I'd love to change this movie"
        expected_string6 = "I 'd love to change this movie"
        actual_string6 = add_space_between_characters(text_string6)

        self.assertEqual(expected_string6, actual_string6)

        text_string7 = "I'll watch this movie"
        expected_string7 = "I 'll watch this movie"
        actual_string7 = add_space_between_characters(text_string7)

        self.assertEqual(expected_string7, actual_string7)

        text_string8 = "I liked the movie, but there are some problems"
        expected_string8 = "I liked the movie ,  but there are some problems"
        actual_string8 = add_space_between_characters(text_string8)

        self.assertEqual(expected_string8, actual_string8)

        text_string9 = "What a great movie!!!"
        expected_string9 = "What a great movie !  !  ! "
        actual_string9 = add_space_between_characters(text_string9)

        self.assertEqual(expected_string9, actual_string9)

        text_string10 = "What a great movie(You must see it)"
        expected_string10 = "What a great movie ( You must see it ) "
        actual_string10 = add_space_between_characters(text_string10)

        self.assertEqual(expected_string10, actual_string10)

        text_string11 = "Why have I watched this movie?"
        expected_string11 = "Why have I watched this movie ? "
        actual_string11 = add_space_between_characters(text_string11)

        self.assertEqual(expected_string11, actual_string11)

    def test_full_setence_preprocessing(self):
        text_string = "What a great movie!!!<br /><br />This was an \"crazy\" experience."
        expected_string = "what a great movie ! ! ! this was an crazy experience "

        text_string = remove_html_from_text(text_string)
        text_string = add_space_between_characters(text_string)
        text_string = remove_special_characters_from_text(text_string)
        actual_string = to_lower(text_string)

        self.assertEqual(expected_string, actual_string)

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

    def test_create_vocab_parser(self):
        word_index, glove_matrix, vocab = load_glove('data/glove.6B.50d.txt')
        vocabulary_processor = create_vocab_parser(vocab, 10)

        vp_size = len(vocabulary_processor.vocabulary_._mapping.keys())
        vocab_size = len(vocab)

        self.assertEqual(vp_size, vocab_size + 1)
    test_create_vocab_parser.slow = 1
