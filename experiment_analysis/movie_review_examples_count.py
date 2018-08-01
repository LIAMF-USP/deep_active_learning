import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


class DatasetCounter():

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def count_all_examples(self, dataset_type):
        raise NotImplementedError

    def perform_dataset_analysis(self):
        raise NotImplementedError


class LMRDCounter(DatasetCounter):
    def __init__(self, data_dir):
        super().__init__(data_dir)

        self.graph_name = 'lmdr_count_graph'
        self.labels = ('Positive', 'Negative')

    def create_graph(self, train_pos_count, train_neg_count, test_pos_count, test_neg_count):
        N = 2
        ind = np.arange(N)  # the x locations for the groups
        width = 0.2         # the width of the bars
        tick_labels = ['Train', 'Test']

        fig, ax = plt.subplots()

        train_values = (train_pos_count, train_neg_count)
        rects1 = ax.bar(ind, train_values, width, color='xkcd:light green')

        test_values = (test_pos_count, test_neg_count)
        rects2 = ax.bar(ind + width, test_values, width, color='xkcd:light red')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Number of examples')
        ax.set_title('Examples by type and dataset')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(tick_labels)

        ax.legend((rects1[0], rects2[0]), self.labels, loc='upper center')
        ax.legend(self.labels, loc='upper center')

        plt.savefig('experiment_analysis/' + self.graph_name)

    def count_dir_examples(self, data_path, dir_name):
        dir_path = os.path.join(data_path, dir_name)
        return len([f for f in os.listdir(dir_path) if f.endswith('.txt')])

    def count_all_examples(self, dataset_type):
        data_path = os.path.join(self.data_dir, dataset_type)

        dataset_pos = self.count_dir_examples(data_path, 'pos')
        dataset_neg = self.count_dir_examples(data_path, 'neg')

        return dataset_pos, dataset_neg

    def perform_dataset_analysis(self):
        print('Counting training set examples...')
        train_pos_count, train_neg_count = self.count_all_examples('train')

        print('Counting test set examples...')
        test_pos_count, test_neg_count = self.count_all_examples('test')

        print('Creating Train and Test sets graph...')
        self.create_graph(train_pos_count, train_neg_count, test_pos_count, test_neg_count)


class SubjectivityCounter(DatasetCounter):
    def __init__(self, data_dir):
        super().__init__(data_dir)

        self.graph_name = 'subjectivity_count_graph'
        self.labels = ('Subjective', 'Objective')

        self.subjective_filename = 'quote.tok.gt9.5000'
        self.objective_filename = 'plot.tok.gt9.5000'

    def create_graph(self, subjective_count, objective_count):
        ind = [1, 2]
        fig, ax = plt.subplots()

        pm, pc = plt.bar(ind, (subjective_count, objective_count))
        pm.set_facecolor('g')
        pc.set_facecolor('r')
        ax.set_xticks(ind)
        ax.set_xticklabels(['Subjective', 'Objective'])
        ax.set_ylim([0, 6000])
        ax.set_ylabel('Number of Examples')

        plt.savefig('experiment_analysis/' + self.graph_name)

    def count_file_lines(self, dataset_filename):
        dataset_file = os.path.join(self.data_dir, dataset_filename)
        with open(dataset_file, 'r', encoding='latin1') as sf:
            dataset_count = len(sf.readlines())

        return dataset_count

    def count_all_examples(self, dataset_type):
        subjective_count = self.count_file_lines(self.subjective_filename)
        objective_count = self.count_file_lines(self.objective_filename)

        return subjective_count, objective_count

    def perform_dataset_analysis(self):
        print('Counting examples...')
        subjective_count, objective_count = self.count_all_examples(None)
        print('Creating graph')
        self.create_graph(subjective_count, objective_count)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')

    parser.add_argument('-dn',
                        '--dataset_name',
                        type=str,
                        help='Dataset name')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    if user_args['dataset_name'] == 'lmrd':
        counter = LMRDCounter(user_args['data_dir'])
    else:
        counter = SubjectivityCounter(user_args['data_dir'])

    counter.perform_dataset_analysis()


if __name__ == '__main__':
    main()
