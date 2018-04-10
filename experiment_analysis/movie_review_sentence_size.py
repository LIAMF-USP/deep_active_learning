import argparse
import os
import re

from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def create_graph(sentences_size, sentence_mean, sentence_std, sentence_type,
                 dataset_type):
    n, bins, patches = plt.hist(
        sentences_size, 200, normed=1, facecolor='green', alpha=0.75,
        histtype='bar', ec='black')

    y = mlab.normpdf(bins, sentence_mean, sentence_std)
    plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Sentence Size')
    plt.ylabel('Probability')
    plt.title(
        'Histogram ({0} set) of {1} Sentence Sizes\n u={2:.2f}, sigma={3:.2f}'.format(
             dataset_type, sentence_type, sentence_mean, sentence_std))
    plt.axis([20, 1000, 0, 0.007])
    plt.savefig('dataset_analysis/{}_{}_graph'.format(
        dataset_type, sentence_type.lower()))
    plt.close()


def get_review_size(sentences_dir, review_file):
    review_file = os.path.join(sentences_dir, review_file)

    with open(review_file, 'r') as rf:
        sentence = rf.read()

    words = re.findall(r'\w+', sentence)
    return len(words)


def average_sentence_size(data_path, review_type):
    sentences_dir = os.path.join(data_path, review_type)
    sentences_size = []

    for review in os.scandir(sentences_dir):
        review_size = get_review_size(sentences_dir, review.name)
        sentences_size.append(review_size)

    mean, std = norm.fit(sentences_size)
    return (sentences_size, mean, std)


def perform_dataset_analysis(user_args):
    data_dir = user_args['data_dir']
    dataset_type = user_args['dataset_type']

    data_path = os.path.join(data_dir, dataset_type)

    pos_size, pos_mean, pos_std = average_sentence_size(data_path, 'pos')
    print('Creating {} set positive sentences graph...'.format(dataset_type))
    create_graph(pos_size, pos_mean, pos_std, 'Positive', dataset_type)

    neg_size, neg_mean, neg_std = average_sentence_size(data_path, 'neg')
    print('Creating {} set negative sentences graph...'.format(dataset_type))
    create_graph(neg_size, neg_mean, neg_std, 'Negative', dataset_type)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')

    parser.add_argument('-dt',
                        '--dataset_type',
                        type=str,
                        help='Run analysis on train or test set')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    perform_dataset_analysis(user_args)


if __name__ == '__main__':
    main()
