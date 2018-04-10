import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def create_graph(train_pos_count, train_neg_count,
                 test_pos_count, test_neg_count):
    N = 2
    train_values = (train_pos_count, train_neg_count)
    test_values = (test_pos_count, test_neg_count)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2         # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, train_values, width, color='xkcd:light green')
    rects2 = ax.bar(ind + width, test_values, width, color='xkcd:light red')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Number of examples')
    ax.set_title('Examples by type and dataset')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Train', 'Test'))

    ax.legend((rects1[0], rects2[0]), ('Positive', 'Negative'),
              loc='upper center')

    plt.savefig('dataset_analysis/example_count_graph')


def count_dir_examples(data_path, dir_name):
    dir_path = os.path.join(data_path, dir_name)
    return len([f for f in os.listdir(dir_path) if f.endswith('.txt')])


def count_all_examples(data_dir, dataset_type):
    data_path = os.path.join(data_dir, dataset_type)

    dataset_pos = count_dir_examples(data_path, 'pos')
    dataset_neg = count_dir_examples(data_path, 'neg')

    return dataset_pos, dataset_neg


def perform_dataset_analysis(user_args):
    data_dir = user_args['data_dir']

    print('Counting training set examples...')
    train_pos_count, train_neg_count = count_all_examples(data_dir, 'train')

    print('Counting test set examples...')
    test_pos_count, test_neg_count = count_all_examples(data_dir, 'test')

    print('Creating Train and Test sets graph...')
    create_graph(train_pos_count, train_neg_count,
                 test_pos_count, test_neg_count)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    perform_dataset_analysis(user_args)


if __name__ == '__main__':
    main()
