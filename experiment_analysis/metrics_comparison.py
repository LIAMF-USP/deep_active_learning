import argparse
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-df',
                        '--data-folder',
                        type=str,
                        help='Location of the metrics that will be compared')

    parser.add_argument('-ne',
                        '--num-experiments',
                        type=int,
                        help='The number of experiments performed for each metric')

    parser.add_argument('-mcf',
                        '--metrics-accuracy-file',
                        type=str,
                        help='Name of the accuracy file generated for each experiment')

    parser.add_argument('-ndf',
                        '--num-data-file',
                        type=str,
                        help='Name of the num data file generated for each experiment')

    parser.add_argument('-mf',
                        '--metrics-folder',
                        nargs='+',
                        help='List of folders containing the metrics data')

    parser.add_argument('-mn',
                        '--metric-names',
                        nargs='+',
                        help='The name of each metric to be compared')

    parser.add_argument('-gp',
                        '--graph-path',
                        type=str,
                        help='Location to save the graph')

    parser.add_argument('-gn',
                        '--graph-name',
                        type=str,
                        help='Graph name')

    parser.add_argument('-ustd',
                        '--use-standard-deviation',
                        type=int,
                        help='Use standard deviation in the graph')

    return parser


def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def final_accuracies(test_file, train_file, num_experiments):
    test_accuracies = []

    for i in range(1, num_experiments + 1):
        index = str(i)
        test_accuracies.append(load(test_file.format(index)))

    final_accuracies = []
    std_values = []

    for a in zip(*test_accuracies):
        value = sum(a) / len(a)
        std = np.std(a)

        final_accuracies.append(value)
        std_values.append(std)

    train_data = load(train_file.format('1'))

    return final_accuracies, train_data, std_values


def create_graph(metric_names, metric_accuracies, train_data, std_final, use_std, graph_save_path):

    for metric_name, metric_accuracy, std_values in zip(metric_names, metric_accuracies, std_final):
        plt.plot(train_data, metric_accuracy, label=metric_name)

        std_array = np.array(std_values)
        y = np.array(metric_accuracy)

        if use_std:
            plt.fill_between(train_data, y - std_array, y + std_array, alpha=0.5)

    plt.legend(loc='lower right')
    plt.savefig(graph_save_path)


def make_comparison(user_args):
    data_folder = user_args['data_folder']
    metrics_accuracy_file = user_args['metrics_accuracy_file']
    num_data_file = user_args['num_data_file']

    metric_names = user_args['metric_names']
    metrics_folder = user_args['metrics_folder']
    num_experiments = user_args['num_experiments']

    test_file = metrics_accuracy_file + '_{}.pkl'
    train_file = num_data_file + '_{}.pkl'

    use_std = True if user_args['use_standard_deviation'] == 1 else False

    metric_accuracies = []
    std_final = []

    for metric_name, metric_folder in zip(metric_names, metrics_folder):
        metric_file = os.path.join(data_folder, metric_folder, test_file)
        metric_train_file = os.path.join(data_folder, metric_folder, train_file)

        metric_accuracy, train_data, std_values = final_accuracies(
            metric_file, metric_train_file, num_experiments)

        metric_accuracies.append(metric_accuracy)
        std_final.append(std_values)

    graph_path = user_args['graph_path']
    graph_name = user_args['graph_name']
    graph_save_path = os.path.join(graph_path, graph_name)

    create_graph(metric_names, metric_accuracies, train_data, std_final,
                 use_std, graph_save_path)


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    make_comparison(user_args)


if __name__ == '__main__':
    main()
