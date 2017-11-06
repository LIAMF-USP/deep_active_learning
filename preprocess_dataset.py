import argparse
import os

from preprocessing.format_dataset import remove_html_from_text, to_lower
from utils.progress_bar import Progbar


def preprocess_review_files(dataset_path, output_dataset_path, review_type):
    dataset_sentiment_type = os.path.join(dataset_path, review_type)
    output_sentiment_type = os.path.join(output_dataset_path, review_type)

    review_files = os.listdir(dataset_sentiment_type)
    num_review_files = len(review_files)

    print('Formatting {} texts'.format(review_type))
    progbar = Progbar(target=num_review_files)

    for index, review in enumerate(review_files):
        original_review = os.path.join(dataset_sentiment_type, review)
        with open(original_review, 'r') as review_file:
            formatted_text = remove_html_from_text(review_file.read())
            formatted_text = to_lower(formatted_text)

        output_review = os.path.join(output_sentiment_type, review)
        with open(output_review, 'w') as review_file:
            review_file.write(formatted_text)

        progbar.update(index + 1, [])
    print()


def preprocess_files(dataset_path, output_dataset_path):
    preprocess_review_files(dataset_path, output_dataset_path, 'pos')
    preprocess_review_files(dataset_path, output_dataset_path, 'neg')


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

    preprocess_files(dataset_path, output_dataset_path)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')
    parser.add_argument('-dt',
                        '--dataset_type',
                        type=str,
                        help='The dataset that should be formatted: train or test')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='The path of the new formatted dataset')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    apply_data_preprocessing(user_args)


if __name__ == '__main__':
    main()