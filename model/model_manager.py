import os
import random

import numpy as np
import tensorflow as tf

from model.input_pipeline import InputPipeline, ALInputPipeline
from model.recurrent_model import RecurrentModel, RecurrentConfig
from word_embedding.word_embedding import get_embedding
from utils.tensorboard import create_unique_name
from utils.graphs import accuracy_graph
from utils.metrics import variation_ratio
from preprocessing.dataset import load


class ModelManager:

    def __init__(self, model_params):
        self.model_params = model_params
        self.sess = None

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
        self.input_pipeline = self.create_dataset()
        print('Calculating number of batches...')
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
        self.recurrent_model = RecurrentModel(recurrent_config, embedding_matrix)

        saved_model_path = self.model_params['saved_model_folder']

        self.sess = tf.Session()

        writer = self.initialize_tensorboard()
        writer.add_graph(self.sess.graph)

        self.recurrent_model.prepare(self.sess, self.input_pipeline)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        try:
            (best_accuracy, train_accuracies,
                val_accuracies, test_accuracy) = self.recurrent_model.fit(
                    self.sess, self.input_pipeline, saved_model_path, writer)
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


class ActiveLearningModelManager(ModelManager):

    def __init__(self, model_params, active_learning_params):
        self.active_learning_params = active_learning_params
        super().__init__(model_params)

    def create_dataset(self):
        train_data = self.model_params['train_data']
        validation_data = self.model_params['validation_data']
        test_data = self.model_params['test_data']
        batch_size = self.model_params['batch_size']
        perform_shuffle = self.model_params['perform_shuffle']
        bucket_width = self.model_params['bucket_width']
        num_buckets = self.model_params['num_buckets']

        input_pipeline = ALInputPipeline(
            train_data, validation_data, test_data, batch_size, perform_shuffle,
            bucket_width, num_buckets)
        input_pipeline.build_pipeline()

        return input_pipeline

    def get_index(self, train_labels, label, size):
        # Initialize variable with a single 0
        label_indexes = np.zeros(shape=(1), dtype=np.int64)

        for index, review_label in enumerate(train_labels):
            if review_label == label:
                label_indexes = np.append(label_indexes, np.array([index], dtype=np.int64))

        # Remove initialization variable
        label_indexes = label_indexes[1:]
        np.random.shuffle(label_indexes)

        return label_indexes[:size]

    def create_initial_dataset(self):
        train_file = self.active_learning_params['train_file']
        test_file = self.active_learning_params['test_file']
        train_initial_size = self.active_learning_params['train_initial_size']

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
        negative_samples = self.get_index(train_labels, 0, size)
        positive_samples = self.get_index(train_labels, 1, size)
        train_indexes = np.concatenate([negative_samples, positive_samples])

        labeled_dataset = (train_ids[train_indexes], train_labels[train_indexes],
                           train_sizes[train_indexes])
        unlabeled_dataset = (unlabeled_ids, unlabeled_labels, unlabeled_sizes)
        test_dataset = (test_ids, test_labels, test_sizes)

        return labeled_dataset, unlabeled_dataset, test_dataset

    def unlabeled_uncertainty(self, num_samples=10):
        all_preds, all_labels = self.recurrent_model.monte_carlo_samples(
            self.sess, self.input_pipeline.validation_iterator, num_samples=10)
        mc_counts = self.recurrent_model.monte_carlo_samples_count(all_preds)
        variation_ratios = np.array(variation_ratio(mc_counts))

        return variation_ratios

    def select_samples(self):
        sample_size = self.active_learning_params['sample_size']
        unlabeled_pool_indexes = np.array(
            random.sample(range(0, self.unlabeled_dataset_word_id.shape[0]), sample_size)
        )

        pool_unlabeled_ids = self.unlabeled_dataset_word_id[unlabeled_pool_indexes]
        pool_unlabeled_labels = self.unlabeled_dataset_labels[unlabeled_pool_indexes]
        pool_unlabeled_sizes = self.unlabeled_dataset_sizes[unlabeled_pool_indexes]

        return (pool_unlabeled_ids, pool_unlabeled_labels, pool_unlabeled_sizes,
                unlabeled_pool_indexes)

    def select_new_examples(self, pool_unlabeled_ids, pool_unlabeled_labels,
                            pool_unlabeled_sizes):
        num_passes = self.active_learning_params['num_passes']
        num_queries = self.active_learning_params['num_queries']

        unlabeled_uncertainty = self.unlabeled_uncertainty(num_passes)
        new_samples = unlabeled_uncertainty.argsort()[-num_queries:][::-1]
        self.reset_graph()

        pooled_word_id = pool_unlabeled_ids[new_samples]
        pooled_labels = pool_unlabeled_labels[new_samples]
        pooled_sizes = pool_unlabeled_sizes[new_samples]

        return pooled_word_id, pooled_labels, pooled_sizes, new_samples

    def update_labeled_dataset(self, pooled_word_id, pooled_labels, pooled_sizes):
        labeled_word_id = self.labeled_dataset[0]
        labeled_labels = self.labeled_dataset[1]
        labeled_sizes = self.labeled_dataset[2]

        labeled_word_id = np.concatenate([labeled_word_id, pooled_word_id], axis=0)
        labeled_labels = np.concatenate([labeled_labels, pooled_labels], axis=0)
        labeled_sizes = np.concatenate([labeled_sizes, pooled_sizes], axis=0)

        return labeled_word_id, labeled_labels, labeled_sizes

    def remove_data_from_dataset(self, ids, labels, sizes, data_index):
        deleted_ids = np.delete(ids, (data_index), axis=0)
        deleted_labels = np.delete(labels, (data_index), axis=0)
        deleted_sizes = np.delete(sizes, (data_index), axis=0)

        return deleted_ids, deleted_labels, deleted_sizes

    def update_unlabeled_dataset(self, delete_unlabeled_word_id, delete_unlabeled_word_id_sample,
                                 delete_unlabeled_labels, delete_unlabeled_labels_sample,
                                 delete_unlabeled_sizes, delete_unlabeled_sizes_sample):
        self.unlabeled_dataset_word_id = np.concatenate(
            [delete_unlabeled_word_id, delete_unlabeled_word_id_sample], axis=0)
        self.unlabeled_dataset_labels = np.concatenate(
            [delete_unlabeled_labels, delete_unlabeled_labels_sample], axis=0)
        self.unlabeled_dataset_sizes = np.concatenate(
            [delete_unlabeled_sizes, delete_unlabeled_sizes_sample], axis=0)

    def run_cycle(self):
        (self.labeled_dataset, unlabeled_dataset,
         test_dataset) = self.create_initial_dataset()

        self.unlabeled_dataset_word_id = unlabeled_dataset[0]
        self.unlabeled_dataset_labels = unlabeled_dataset[1]
        self.unlabeled_dataset_sizes = unlabeled_dataset[2]

        test_accuracies, train_data_sizes = [], []
        self.model_params['num_validation'] = self.active_learning_params['sample_size']
        self.model_params['test_data'] = test_dataset

        for i in range(self.active_learning_params['num_rounds']):

            (pool_unlabeled_ids, pool_unlabeled_labels,
             pool_unlabeled_sizes, unlabeled_pool_indexes) = self.select_samples()

            pool_unlabeled_dataset = (pool_unlabeled_ids, pool_unlabeled_labels,
                                      pool_unlabeled_sizes)

            self.model_params['train_data'] = self.labeled_dataset
            self.model_params['validation_data'] = pool_unlabeled_dataset

            _, _, _, test_accuracy = self.run_model()

            test_accuracies.append(test_accuracy)

            pooled_word_id, pooled_labels, pooled_sizes, sample_index = self.select_new_examples(
                pool_unlabeled_ids, pool_unlabeled_labels, pool_unlabeled_sizes)
            train_data_sizes.append(len(self.labeled_dataset[0]))

            labeled_word_id, labeled_labels, labeled_sizes = self.update_labeled_dataset(
                pooled_word_id, pooled_labels, pooled_sizes)
            self.labeled_dataset = (labeled_word_id, labeled_labels, labeled_sizes)

            (delete_unlabeled_word_id_sample, delete_unlabeled_labels_sample,
             delete_unlabeled_sizes_sample) = self.remove_data_from_dataset(
                    pool_unlabeled_ids, pool_unlabeled_labels, pool_unlabeled_sizes, sample_index)

            (delete_unlabeled_word_id, delete_unlabeled_labels,
             delete_unlabeled_sizes) = self.remove_data_from_dataset(
                self.unlabeled_dataset_word_id, self.unlabeled_dataset_labels,
                self.unlabeled_dataset_sizes, unlabeled_pool_indexes)

            self.unlabeled_dataset_word_id = np.concatenate(
                [delete_unlabeled_word_id, delete_unlabeled_word_id_sample], axis=0)
            self.unlabeled_dataset_labels = np.concatenate(
                [delete_unlabeled_labels, delete_unlabeled_labels_sample], axis=0)
            self.unlabeled_dataset_sizes = np.concatenate(
                [delete_unlabeled_sizes, delete_unlabeled_sizes_sample], axis=0)

            self.update_unlabeled_dataset(
                delete_unlabeled_word_id, delete_unlabeled_word_id_sample,
                delete_unlabeled_labels, delete_unlabeled_labels_sample,
                delete_unlabeled_sizes, delete_unlabeled_sizes_sample)

            print('End of round {}'.format(i))
            print('Size of pool {}'.format(self.unlabeled_dataset_word_id.shape[0]))
            print('Train data size: {}'.format(len(labeled_word_id)))
            print()