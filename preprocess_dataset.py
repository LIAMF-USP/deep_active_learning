import argparse
import random
import os

import numpy as np

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_special_characters_from_text,
                                          add_space_between_characters,
                                          sentence_to_id_list,
                                          create_vocab_parser,
                                          SentenceTFRecord,
                                          load_glove,
                                          to_lower)
from utils.progress_bar import Progbar


def preprocess_review_text(review_text):
    formatted_text = remove_html_from_text(review_text)
    formatted_text = add_space_between_characters(formatted_text)
    formatted_text = remove_special_characters_from_text(formatted_text)
    return to_lower(formatted_text)


def preprocess_review_files(dataset_path, output_dataset_path, review_type):
    dataset_sentiment_type = os.path.join(dataset_path, review_type)
    output_sentiment_type = os.path.join(output_dataset_path, review_type)

    review_files = os.listdir(dataset_sentiment_type)
    num_review_files = len(review_files)
    formatted_review_texts = []

    print('Formatting {} texts'.format(review_type))
    progbar = Progbar(target=num_review_files)

    for index, review in enumerate(review_files):
        original_review = os.path.join(dataset_sentiment_type, review)
        with open(original_review, 'r') as review_file:
            review_text = review_file.read()

        formatted_text = preprocess_review_text(review_text)
        formatted_review_texts.append(formatted_text)

        output_review = os.path.join(output_sentiment_type, review)
        with open(output_review, 'w') as review_file:
            review_file.write(formatted_text)

        progbar.update(index + 1, [])
    print()

    return formatted_review_texts


def preprocess_files(dataset_path, output_dataset_path):
    pos_reviews = preprocess_review_files(dataset_path, output_dataset_path, 'pos')
    neg_reviews = preprocess_review_files(dataset_path, output_dataset_path, 'neg')

    return pos_reviews, neg_reviews


def make_output_dir(output_dataset_path):
    os.makedirs(output_dataset_path)

    output_pos = os.path.join(output_dataset_path, 'pos')
    os.makedirs(output_pos)

    output_neg = os.path.join(output_dataset_path, 'neg')
    os.makedirs(output_neg)


def apply_data_preprocessing(user_args):
    data_dir = user_args['data_dir']
    dataset_type = user_args['dataset_type']
    output_dir = user_args['output_dir']

    output_dataset_path = os.path.join(output_dir, dataset_type)
    dataset_path = os.path.join(data_dir, dataset_type)

    if not os.path.exists(output_dataset_path):
        make_output_dir(output_dataset_path)

    return preprocess_files(dataset_path, output_dataset_path)


def transform_sentences(vocabulary_processor, movie_reviews):
    for review in movie_reviews:
        yield sentence_to_id_list(review, vocabulary_processor)


def save_sentences_id_list(dataset_path, sentence_type, sentences_id_list,
                           num_sentences):
    sentences_type_path = os.path.join(dataset_path, sentence_type)
    filename = '{}_sentences_id_list.txt'.format(sentence_type)
    sentences_id_path = os.path.join(sentences_type_path, filename)

    progbar = Progbar(target=num_sentences)

    with open(sentences_id_path, 'wb') as sentences_file:
        for index, review in enumerate(sentences_id_list):
            review = list(review)[0]
            review = review.reshape(1, review.shape[0])
            np.savetxt(sentences_file, review, fmt='%u', delimiter=' ',
                       newline='\n', header='', footer='', comments='# ')

            progbar.update(index + 1, [])


def transform_data(pos_reviews, neg_reviews, user_args):
    glove_file = user_args['glove_file']
    sentence_size = user_args['sentence_size']
    data_dir = user_args['output_dir']
    dataset_type = user_args['dataset_type']

    print('Loading glove embeddings')
    word_index, glove_matrix, vocab = load_glove(glove_file)

    print('Creating Vocabulary Parser')
    vocabulary_processor = create_vocab_parser(vocab, sentence_size)
    print()

    dataset_path = os.path.join(data_dir, dataset_type)
    pos_sentences_id_list = transform_sentences(vocabulary_processor, pos_reviews)
    print('Saving positive sentences id lists')
    save_sentences_id_list(dataset_path, 'pos', pos_sentences_id_list, len(pos_reviews))

    neg_sentences_id_list = transform_sentences(vocabulary_processor, neg_reviews)
    print('Saving negative sentences id lists')
    save_sentences_id_list(dataset_path, 'neg', neg_sentences_id_list, len(neg_reviews))
    print()


def create_tf_record(sentences_id_path, output_path, label):
    label_str = 'pos' if label == 0 else 'neg'
    print('Creating {} TFRecords'.format(label_str))

    progbar = Progbar(target=0)
    sentence_tfrecord = SentenceTFRecord(sentences_id_path, output_path, label, progbar)
    sentence_tfrecord.parse_file()


def create_validation_set(pos_reviews, neg_reviews, percent=0.1):
    num_pos = int(len(pos_reviews) * percent)
    num_neg = int(len(neg_reviews) * percent)

    random.shuffle(pos_reviews)
    random.shuffle(neg_reviews)

    validation_pos = pos_reviews[0:num_pos]
    validation_neg = neg_reviews[0:num_neg]

    pos_reviews = pos_reviews[num_pos:]
    neg_reviews = neg_reviews[num_neg:]

    return pos_reviews, neg_reviews, validation_pos, validation_neg


def create_validation_dir(user_args):
    output_dir = user_args['output_dir']
    validation_dir = 'val'

    validation_path = os.path.join(output_dir, validation_dir)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data-dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')
    parser.add_argument('-dt',
                        '--dataset-type',
                        type=str,
                        help='The dataset that should be formatted: train or test')
    parser.add_argument('-gf',
                        '--glove-file',
                        type=str,
                        help='The location of the GloVe file')
    parser.add_argument('-s',
                        '--sentence-size',
                        type=int,
                        help=('The sentence size that will be used in the model.' +
                              'If a sentence in our dataset is larger than this variable' +
                              'It will be cropped to this size. Otherwise, it will be padded' +
                              'with an special character'))
    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        help='The path of the new formatted dataset')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    pos_reviews, neg_reviews = apply_data_preprocessing(user_args)
    transform_data(pos_reviews, neg_reviews, user_args)

    output_dir = user_args['output_dir']
    dataset_type = user_args['dataset_type']

    output_path = os.path.join(output_dir, dataset_type, 'pos', 'pos.tfrecord')
    sentences_id_path = os.path.join(output_dir, dataset_type, 'pos', 'pos_sentences_id_list.txt')
    create_tf_record(sentences_id_path, output_path, 0)

    output_path = os.path.join(output_dir, dataset_type, 'neg', 'neg.tfrecord')
    sentences_id_path = os.path.join(output_dir, dataset_type, 'neg', 'neg_sentences_id_list.txt')
    create_tf_record(sentences_id_path, output_path, 1)


if __name__ == '__main__':
    main()
