import argparse
import os

import tensorflow as tf

from model.input_pipeline import InputPipeline
from model.recurrent_model import RecurrentModel, RecurrentConfig
from word_embedding.word_embedding import get_embedding
from utils.tensorboard import create_unique_name
from utils.graphs import accuracy_graph

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
    bucket_width = user_args['bucket_width']
    num_buckets = user_args['num_buckets']

    input_pipeline = InputPipeline(
        train_file, validation_file, test_file, batch_size, perform_shuffle,
        bucket_width, num_buckets)
    input_pipeline.build_pipeline()

    return input_pipeline


def initialize_tensorboard(user_args):
    model_name = user_args['model_name']
    tensorboard_save_name = create_unique_name(model_name)
    tensorboard_dir = user_args['tensorboard_dir']

    writer = tf.summary.FileWriter(
        os.path.join(tensorboard_dir, tensorboard_save_name))

    return writer, tensorboard_save_name


def save_accuracy_graph(train_accuracies, val_accuracies, graphs_dir, save_name):
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    save_path = os.path.join(graphs_dir, save_name)
    accuracy_graph(train_accuracies, val_accuracies, save_path)


def bool_arguments(value):
    return True if int(value) == 1 else False


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

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    print('Creating dataset...')
    input_pipeline = create_dataset(user_args)
    print('Calculating number of batches...')
    input_pipeline.get_datasets_num_batches()

    print('Loading embedding file...')
    embedding_file = user_args['embedding_file']
    embed_size = user_args['embed_size']
    embedding_pickle = user_args['embedding_pickle']
    word_embedding = get_embedding(
        embedding_file, embed_size, None, embedding_pickle)
    _, embedding_matrix, _ = word_embedding.get_word_embedding()

    print('Creating Recurrent model...')
    recurrent_config = RecurrentConfig(user_args)
    recurrent_model = RecurrentModel(recurrent_config, embedding_matrix)

    with tf.Session() as sess:
        writer, save_name = initialize_tensorboard(user_args)
        writer.add_graph(sess.graph)

        recurrent_model.prepare(sess, input_pipeline)

        init = tf.global_variables_initializer()
        sess.run(init)

        train_accuracies, val_accuracies = recurrent_model.fit(sess, input_pipeline, writer)

    graphs_dir = user_args['graphs_dir']
    save_accuracy_graph(train_accuracies, val_accuracies, graphs_dir, save_name)


if __name__ == '__main__':
    main()
