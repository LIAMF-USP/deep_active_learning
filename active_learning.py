import argparse
import os

from model.model_manager import ActiveLearningModelManager


DEFAULT_BATCH_SIZE = 32
DEFAULT_PERFORM_SHUFFLE = True
DEFAULT_NUM_EPOCHS = 10000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    parser.add_argument('-nr',
                        '--num-rounds',
                        type=int,
                        help='Number of Active Learning cycles to run')

    parser.add_argument('-sz',
                        '--sample-size',
                        type=int,
                        help='The size of samples to evaluate in the unlabeled dataset')

    parser.add_argument('-nq',
                        '--num-queries',
                        type=int,
                        help='The number of data to be extracted from the unlabeled sample')

    parser.add_argument('-nfp',
                        '--num-passes',
                        type=int,
                        help='Number of forward passes for Monte Carlo Dropout')

    parser.add_argument('-its',
                        '--initial-training-size',
                        type=int,
                        help='Initial size of the training set')

    return parser


def run_active_learning(**user_args):
    active_learning_params = {
        'train_file': user_args['train_file'],
        'test_file': user_args['test_file'],
        'num_rounds': user_args['num_rounds'],
        'sample_size': user_args['sample_size'],
        'num_queries': user_args['num_queries'],
        'num_passes': user_args['num_passes'],
        'train_initial_size': user_args['initial_training_size']
    }

    model_params = {
        'embedding_file': user_args['embedding_file'],
        'embed_size': user_args['embed_size'],
        'embedding_pickle': user_args['embedding_pickle'],
        'saved_model_folder': user_args['saved_model_folder'],
        'perform_shuffle': user_args['perform_shuffle'],
        'model_name': user_args['model_name'],
        'tensorboard_dir': user_args['tensorboard_dir'],
        'graphs_dir': user_args['graphs_dir'],
        'save_graph': user_args['save_graph'],
        'learning_rate': user_args['learning_rate'],
        'batch_size': user_args['batch_size'],
        'num_epochs': user_args['num_epochs'],
        'num_classes': user_args['num_classes'],
        'num_train': user_args['num_train'],
        'num_test': user_args['num_test'],
        'use_test': user_args['use_test'],
        'use_validation': user_args['use_validation'],
        'num_units': user_args['num_units'],
        'recurrent_output_dropout': user_args['recurrent_output_dropout'],
        'recurrent_state_dropout': user_args['recurrent_state_dropout'],
        'embedding_dropout': user_args['embedding_dropout'],
        'weight_decay': user_args['weight_decay'],
        'clip_gradients': user_args['clip_gradients'],
        'max_norm': user_args['max_norm'],
        'bucket_width': user_args['bucket_width'],
        'num_buckets': user_args['num_buckets']
    }

    al_model_manager = ActiveLearningModelManager(model_params, active_learning_params)
    al_model_manager.run_cycle()


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    return run_active_learning(**user_args)


if __name__ == '__main__':
    main()
