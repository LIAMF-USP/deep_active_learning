import argparse
import os
import subprocess

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_special_characters_from_text,
                                          load_glove,
                                          to_lower)
from utils.progress_bar import Progbar


def preprocess_review_text(review_text):
    formatted_text = remove_html_from_text(review_text)
    formatted_text = remove_special_characters_from_text(review_text)
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


def count_file_lines(text_path):
    # TO DO: Refactor this function
    count_line = subprocess.check_output(['wc', '-l', text_path])
    return int(count_line.decode('utf-8').split()[0])


def transform_data(pos_reviews, neg_reviews, user_args):
    glove_file = user_args['glove_file']

    file_size = count_file_lines(glove_file)
    progbar = Progbar(target=file_size)
    progbar = None

    print('Loading glove models')
    word_index, glove_matrix, vocab = load_glove(glove_file, progbar)


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


if __name__ == '__main__':
    main()
