import argparse
import os
import random

import numpy as np

from model.model_manager import ActiveLearningModelManager
from preprocessing.dataset import load


DEFAULT_BATCH_SIZE = 32
DEFAULT_PERFORM_SHUFFLE = True
DEFAULT_NUM_EPOCHS = 10000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_index(train_labels, label, size):
    # Initialize variable with a single 0
    label_indexes = np.zeros(shape=(1), dtype=np.int64)

    for index, review_label in enumerate(train_labels):
        if review_label == label:
            label_indexes = np.append(label_indexes, np.array([index], dtype=np.int64))

    # Remove initialization variable
    label_indexes = label_indexes[1:]
    np.random.shuffle(label_indexes)

    return label_indexes[:size]


def create_initial_dataset(train_file, test_file, train_initial_size=10):
    train_data = load(train_file)
    test_data = load(test_file)

    random.shuffle(train_data)

    data_ids, data_labels, data_sizes = [], [], []
    test_ids, test_labels, test_sizes = [], [], []

    for word_ids, label, size in train_data:
        data_ids.append(word_ids)
        data_labels.append(label)
        data_sizes.append(size)

    for word_ids, label, size in test_data:
        test_ids.append(word_ids)
        test_labels.append(label)
        test_sizes.append(size)

    train_ids = np.array(data_ids[:30])
    train_labels = np.array(data_labels[:30])
    train_sizes = np.array(data_sizes[:30])

    unlabeled_ids = np.array(data_ids[30:])
    unlabeled_labels = np.array(data_labels[30:])
    unlabeled_sizes = np.array(data_sizes[30:])

    size = int(train_initial_size / 2)
    negative_samples = get_index(train_labels, 0, size)
    positive_samples = get_index(train_labels, 1, size)
    train_indexes = np.concatenate([negative_samples, positive_samples])

    labeled_dataset = (train_ids[train_indexes], train_labels[train_indexes],
                       train_sizes[train_indexes])
    unlabeled_dataset = (unlabeled_ids, unlabeled_labels, unlabeled_sizes)
    test_dataset = (test_ids, test_labels, test_sizes)

    return labeled_dataset, unlabeled_dataset, test_dataset


def bool_arguments(value):
    return True if int(value) == 1 else False


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tr',
                        '--train-file',
                        type=str,
                        help='The location of the train file',
                        required=True)

    parser.add_argument('-ts',
                        '--test-file',
                        type=str,
                        help='The location of the test file',
                        required=True)

    parser.add_argument('-sm',
                        '--saved-model-folder',
                        type=str,
                        help='Location to search/save models. The model name variable will be used for searching')  # noqa

    parser.add_argument('-nt',
                        '--num-train',
                        type=int,
                        help='Number of training examples')

    parser.add_argument('-nte',
                        '--num-test',
                        type=int,
                        help='Number of test examples')

    parser.add_argument('-uv',
                        '--use-validation',
                        type=bool_arguments,
                        help='If the model should provide accuracy measurements using validation set')  # noqa

    parser.add_argument('-gd',
                        '--graphs-dir',
                        type=str,
                        help='The location of the graphs dir')

    parser.add_argument('-mn',
                        '--model-name',
                        type=str,
                        help='The model name that will be used to save tensorboard information')

    parser.add_argument('-td',
                        '--tensorboard-dir',
                        type=str,
                        help='Directory to save tensorboard information')

    parser.add_argument('-ef',
                        '--embedding-file',
                        type=str,
                        help='The path of the embedding file')

    parser.add_argument('-ekl',
                        '--embedding-pickle',
                        type=str,
                        help='The path of embedding matrix pickle file')

    parser.add_argument('-lr',
                        '--learning-rate',
                        type=float,
                        help='The learning rate to use during training')

    parser.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help='The batch size used for stochastic gradient descent')

    parser.add_argument('-np',
                        '--num-epochs',
                        type=int,
                        default=DEFAULT_NUM_EPOCHS,
                        help='Number of epochs to train the model')

    parser.add_argument('-ps',
                        '--perform-shuffle',
                        type=bool_arguments,
                        default=DEFAULT_PERFORM_SHUFFLE,
                        help='If the dataset should be shuffled before using it')

    parser.add_argument('-es',
                        '--embed-size',
                        type=int,
                        help='The embedding size of the embedding matrix')

    parser.add_argument('-nu',
                        '--num-units',
                        type=int,
                        help='The number of hidden units in the Recurrent layer')

    parser.add_argument('-nc',
                        '--num-classes',
                        type=int,
                        help='The number of classification classes')

    parser.add_argument('-lod',
                        '--recurrent-output-dropout',
                        type=float,
                        help='Dropout value for Recurrent output')

    parser.add_argument('-lsd',
                        '--recurrent-state-dropout',
                        type=float,
                        help='Dropout value for recurrent state (variational dropout)')

    parser.add_argument('-ed',
                        '--embedding-dropout',
                        type=float,
                        help='Dropout value for embedding layer')

    parser.add_argument('-cp',
                        '--clip-gradients',
                        type=bool_arguments,
                        help='If gradient clipping should be performed')

    parser.add_argument('-mxn',
                        '--max-norm',
                        type=int,
                        help='The max norm to clip the gradients, if --clip-gradients=True')

    parser.add_argument('-wd',
                        '--weight-decay',
                        type=float,
                        help='Weight Decay value for L2 regularizer')

    parser.add_argument('-bw',
                        '--bucket-width',
                        type=int,
                        help='The width use to define a bucket id for a given movie review')

    parser.add_argument('-nb',
                        '--num-buckets',
                        type=int,
                        help='The maximum number of buckets allowed')

    parser.add_argument('-ut',
                        '--use-test',
                        type=bool_arguments,
                        help='Define if the model should check accuracy on test dataset')

    parser.add_argument('-sg',
                        '--save-graph',
                        type=bool_arguments,
                        help='Define if an accuracy graph should be saved')

    return parser


def run_active_learning(**user_args):
    train_file = user_args['train_file']
    test_file = user_args['test_file']

    labeled_dataset, unlabeled_dataset, test_dataset = create_initial_dataset(
        train_file, test_file)

    user_args['train_data'] = labeled_dataset
    user_args['validation_data'] = unlabeled_dataset
    user_args['num_validation'] = len(unlabeled_dataset[0])
    user_args['test_data'] = test_dataset

    al_model_manager = ActiveLearningModelManager(user_args)
    _, _, _, test_accuracy = al_model_manager.run_model()

    print('Test accuracy for this round: {}'.format(test_accuracy))

    unlabeled_uncertainty = al_model_manager.unlabeled_uncertainty()
    print(unlabeled_uncertainty.shape)


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    return run_active_learning(**user_args)


if __name__ == '__main__':
    main()
