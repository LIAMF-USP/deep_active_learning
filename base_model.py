import argparse
import os

import tensorflow as tf

from model.input_pipeline import InputPipeline
from model.lstm_model import LSTMModel, LSTMConfig
from preprocessing.format_dataset import get_glove_matrix
from utils.tensorboard import create_unique_name

DEFAULT_BATCH_SIZE = 32
DEFAULT_PERFORM_SHUFFLE = True
DEFAULT_NUM_EPOCHS = 10000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_dataset(user_args):
    train_file = user_args['train_file']
    validation_file = user_args['validation_file']
    test_file = user_args['test_file']
    batch_size = user_args['batch_size']
    perform_shuffle = user_args['perform_shuffle']

    input_pipeline = InputPipeline(
        train_file, validation_file, test_file, batch_size, perform_shuffle)
    input_pipeline.build_pipeline()

    return input_pipeline


def initialize_tensorboard(user_args):
    model_name = user_args['model_name']
    tensorboard_save_name = create_unique_name(model_name)
    tensorboard_dir = user_args['tensorboard_dir']

    writer = tf.summary.FileWriter(
        os.path.join(tensorboard_dir, tensorboard_save_name))

    return writer


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tr',
                        '--train-file',
                        type=str,
                        help='The location of the train file',
                        required=True)

    parser.add_argument('-v',
                        '--validation-file',
                        type=str,
                        help='The location of the validation file',
                        required=True)

    parser.add_argument('-ts',
                        '--test-file',
                        type=str,
                        help='The location of the test file',
                        required=True)

    parser.add_argument('-nt',
                        '--num-train',
                        type=int,
                        help='Number of training examples')

    parser.add_argument('-nv',
                        '--num-validation',
                        type=int,
                        help='Number of validation examples')

    parser.add_argument('-nte',
                        '--num-test',
                        type=int,
                        help='Number of test examples')

    parser.add_argument('-mn',
                        '--model-name',
                        type=str,
                        help='The model name that will be used to save tensorboard information')

    parser.add_argument('-td',
                        '--tensorboard-dir',
                        type=str,
                        help='Directory to save tensorboard information')

    parser.add_argument('-gf',
                        '--glove-file',
                        type=str,
                        help='The path of the GloVe file')

    parser.add_argument('-gpkl',
                        '--glove-pickle',
                        type=str,
                        help='The path of GloVe matrix pickle file')

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
                        type=bool,
                        default=DEFAULT_PERFORM_SHUFFLE,
                        help='If the dataset should be shuffled before using it')

    parser.add_argument('-es',
                        '--embed-size',
                        type=int,
                        help='The embedding size of the glove matrix')

    parser.add_argument('-nu',
                        '--num-units',
                        type=int,
                        help='The number of hidden units in the LSTM layer')

    parser.add_argument('-ml',
                        '--max-length',
                        type=int,
                        help='The maximum size of the sentece to be parsed')

    parser.add_argument('-nc',
                        '--num-classes',
                        type=int,
                        help='The number of classification classes')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    print('Creating dataset...')
    input_pipeline = create_dataset(user_args)
    lstm_config = LSTMConfig(user_args)

    print('Loading glove file...')
    glove_file = user_args['glove_file']
    embed_size = user_args['embed_size']
    glove_pickle = user_args['glove_pickle']
    glove_matrix = get_glove_matrix(glove_pickle, glove_file, embed_size)

    print('Creating LSTM model...')
    lstm_model = LSTMModel(lstm_config, glove_matrix)

    with tf.Session() as sess:
        writer = initialize_tensorboard(user_args)
        writer.add_graph(sess.graph)

        lstm_model.prepare(sess, input_pipeline)

        init = tf.global_variables_initializer()
        sess.run(init)

        lstm_model.fit(sess, input_pipeline, writer)


if __name__ == '__main__':
    main()
