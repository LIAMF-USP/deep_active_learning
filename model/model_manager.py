import os

import tensorflow as tf

from model.input_pipeline import InputPipeline
from model.recurrent_model import RecurrentModel, RecurrentConfig
from word_embedding.word_embedding import get_embedding
from utils.graphs import accuracy_graph
from utils.tensorboard import create_unique_name


class ModelManager:

    def __init__(self, model_params, verbose=True):
        self.model_params = model_params
        self.sess = None
        self.recurrent_model = None
        self.verbose = verbose

    def create_dataset(self):
        train_file = self.model_params['train_file']
        validation_file = self.model_params['validation_file']
        test_file = self.model_params['test_file']
        batch_size = self.model_params['batch_size']
        perform_shuffle = self.model_params['perform_shuffle']
        bucket_width = self.model_params['bucket_width']
        num_buckets = self.model_params['num_buckets']

        input_pipeline = InputPipeline(
            train_file, validation_file, test_file, batch_size, perform_shuffle,
            bucket_width, num_buckets)
        input_pipeline.build_pipeline()

        return input_pipeline

    def save_accuracy_graph(self, train_accuracies, val_accuracies):
        graphs_dir = self.model_params['graphs_dir']
        model_name = self.model_params['model_name']
        save_name = create_unique_name(model_name)

        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        save_path = os.path.join(graphs_dir, save_name)
        accuracy_graph(train_accuracies, val_accuracies, save_path)

    def create_model(self):
        print('Creating dataset...')
        self.input_pipeline = self.create_dataset()
        print('Calculating number of batches...')

        if self.verbose:
            self.input_pipeline.get_datasets_num_batches()

        print('Loading embedding file...')
        embedding_file = self.model_params['embedding_file']
        embed_size = self.model_params['embed_size']
        embedding_pickle = self.model_params['embedding_pickle']
        word_embedding = get_embedding(
            embedding_file, embed_size, None, embedding_pickle)
        _, embedding_matrix, _ = word_embedding.get_word_embedding()

        print('Creating Recurrent model...')
        recurrent_config = RecurrentConfig(self.model_params)
        self.recurrent_model = RecurrentModel(
            recurrent_config, embedding_matrix, self.verbose)

        self.saved_model_path = self.model_params['saved_model_folder']

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        self.recurrent_model.build_graph(self.input_pipeline)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def run_model(self):
        if self.recurrent_model is None:
            print('Creating model ...')
            self.create_model()

        try:
            (best_accuracy, train_accuracies,
                val_accuracies, test_accuracy) = self.recurrent_model.fit(
                    self.sess, self.input_pipeline, self.saved_model_path)
        except tf.errors.InvalidArgumentError:
            print('Invalid set of arguments ... ')
            train_accuracies, val_accuracies = [], []
            best_accuracy, test_accuracy = -1, -1

        save_graph = self.model_params['save_graph']

        if save_graph:
            self.save_accuracy_graph(train_accuracies, val_accuracies)

        return best_accuracy, train_accuracies, val_accuracies, test_accuracy

    def reset_graph(self):
        self.sess.close()
        tf.reset_default_graph()
        self.recurrent_model = None
