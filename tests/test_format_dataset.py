import unittest

import numpy as np

from unittest.mock import patch

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_url_from_text,
                                          remove_special_characters_from_text,
                                          create_unique_apostrophe,
                                          add_space_between_characters,
                                          sentence_to_id_list,
                                          to_lower)
from word_embedding.word_embedding import get_embedding


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
        expected_string3 = 'I love this movie )'
        actual_string3 = remove_special_characters_from_text(text_string3)

        self.assertEqual(expected_string3, actual_string3)

        text_string4 = "This movie is awesome!!!it is like a 'poem'"
        expected_string4 = "This movie is awesome!!!it is like a 'poem'"
        actual_string4 = remove_special_characters_from_text(text_string4)

        self.assertEqual(expected_string4, actual_string4)

        text_string5 = "I don't know about this movie"
        expected_string5 = "I don't know about this movie"
        actual_string5 = remove_special_characters_from_text(text_string5)

        self.assertEqual(expected_string5, actual_string5)

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

        text_string12 = "This is interesting;"
        expected_string12 = "This is interesting ; "
        actual_string12 = add_space_between_characters(text_string12)

        self.assertEqual(expected_string12, actual_string12)

        text_string13 = "This is an anti-civilisation movie"
        expected_string13 = "This is an anti - civilisation movie"
        actual_string13 = add_space_between_characters(text_string13)

        self.assertEqual(expected_string13, actual_string13)

    def test_remove_url_from_text(self):
        text_string1 = 'Well, both series are now available on DVD http://www.replaydvd.co.uk/joking_apart_S1.htm'   # noqa
        expected_string1 = 'Well, both series are now available on DVD  '
        actual_string1 = remove_url_from_text(text_string1)

        self.assertEqual(expected_string1, actual_string1)

        text_string2 = 'movie on http://www.imdb.com/title/tt0073891/ for more information)'   # noqa
        expected_string2 = 'movie on   for more information)'   # noqa
        actual_string2 = remove_url_from_text(text_string2)

        self.assertEqual(expected_string2, actual_string2)

    def test_create_unique_apostrophe(self):
        text_string = 'You`ve watched that movie ?'
        expected_string = "You've watched that movie ?"
        actual_string = create_unique_apostrophe(text_string)

        self.assertEqual(expected_string, actual_string)

    def test_full_setence_preprocessing(self):
        text_string = "What a great movie!!!<br /><br />This was a \"crazy\" experience."
        expected_string = "what a great movie ! ! ! this was a crazy experience "

        text_string = remove_html_from_text(text_string)
        text_string = add_space_between_characters(text_string)
        text_string = remove_special_characters_from_text(text_string)
        actual_string = to_lower(text_string)

        self.assertEqual(expected_string, actual_string)

    @patch('numpy.random.uniform')
    def test_load_glove(self, mock_numpy):
        mock_numpy.return_value = np.array([7, 8, 9])
        glove_path = 'tests/test_data/glove_test_data.txt'
        embed_size = 3

        expected_word_index = {'<unk>': 1, 'a': 2, 'b': 3, 'c': 4}
        expected_glove_matrix = [[0, 0, 0],
                                 [7, 8, 9],
                                 [0.1, 0.2, 0.3],
                                 [1, 2, 3],
                                 [4, 5, 6]]
        expected_vocab = ['<unk>', 'a', 'b', 'c']
        vocab = [('<unk>', 0), ('a', 1), ('b', 2), ('c', 3)]

        embedding = get_embedding(glove_path, embed_size, vocab)
        actual_word_index, actual_glove_matrix, actual_vocab = embedding.get_word_embedding(
            should_save=False, verbose=False)

        self.assertEqual(expected_word_index, actual_word_index)

        self.assertEqual(expected_glove_matrix, actual_glove_matrix)
        self.assertEqual(expected_vocab, actual_vocab)

    def test_sentence_to_id_list(self):
        embed_size = 50
        embedding = get_embedding('data/glove/glove.6B.50d.txt', embed_size, [])
        word_index, glove_matrix, vocab = embedding.load_embedding()

        sentence = 'i love you'
        parsed_sentence = sentence.split()
        id_sentence = sentence_to_id_list(sentence, word_index)

        self.assertEqual(len(parsed_sentence), len(id_sentence))

        for index, value in enumerate(parsed_sentence):
            self.assertEqual(word_index[parsed_sentence[index]], id_sentence[index])

        with open('tests/test_data/example_movie_review.txt', 'r') as sentence_file:
            sentence = sentence_file.read()

        id_sentence = sentence_to_id_list(sentence, word_index)
        parsed_sentence = sentence.split()

        self.assertEqual(len(parsed_sentence), len(id_sentence))

        for index, value in enumerate(parsed_sentence):
            self.assertEqual(word_index[parsed_sentence[index]], id_sentence[index])
    test_sentence_to_id_list.slow = 1
