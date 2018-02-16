import os

import tensorflow as tf

from model.input_pipeline import InputPipeline
from model.recurrent_model import RecurrentModel, RecurrentConfig
from word_embedding.word_embedding import get_embedding
from utils.tensorboard import create_unique_name
from utils.graphs import accuracy_graph


class ModelManager:

    def __init__(self, model_params):
        self.model_params = model_params

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

    def initialize_tensorboard(self):
        model_name = self.model_params['model_name']
        self.save_name = create_unique_name(model_name)
        tensorboard_dir = self.model_params['tensorboard_dir']

        writer = tf.summary.FileWriter(
            os.path.join(tensorboard_dir, self.save_name))

        return writer

    def save_accuracy_graph(self, train_accuracies, val_accuracies):
        graphs_dir = self.model_params['graphs_dir']

        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        save_path = os.path.join(graphs_dir, self.save_name)
        accuracy_graph(train_accuracies, val_accuracies, save_path)

    def run_model(self):
        print('Creating dataset...')
        input_pipeline = self.create_dataset()
        print('Calculating number of batches...')
        input_pipeline.get_datasets_num_batches()

        print('Loading embedding file...')
        embedding_file = self.model_params['embedding_file']
        embed_size = self.model_params['embed_size']
        embedding_pickle = self.model_params['embedding_pickle']
        word_embedding = get_embedding(
            embedding_file, embed_size, None, embedding_pickle)
        _, embedding_matrix, _ = word_embedding.get_word_embedding()

        print('Creating Recurrent model...')
        recurrent_config = RecurrentConfig(self.model_params)
        recurrent_model = RecurrentModel(recurrent_config, embedding_matrix)

        saved_model_path = self.model_params['saved_model_folder']

        with tf.Session() as sess:
            writer = self.initialize_tensorboard()
            writer.add_graph(sess.graph)

            recurrent_model.prepare(sess, input_pipeline)

            init = tf.global_variables_initializer()
            sess.run(init)

            try:
                best_accuracy, train_accuracies, val_accuracies, test_accuracy = recurrent_model.fit(
                    sess, input_pipeline, saved_model_path, writer)
            except tf.errors.InvalidArgumentError:
                print('Invalid set of arguments ... ')
                best_accuracy = -1

        save_graph = self.model_params['save_graph']

        if save_graph:
            self.save_accuracy_graph(train_accuracies, val_accuracies)

        tf.reset_default_graph()

        return best_accuracy
