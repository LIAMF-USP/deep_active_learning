import os
import random

import numpy as np
import tensorflow as tf

from model.input_pipeline import InputPipeline, ALInputPipeline
from model.recurrent_model import RecurrentModel, RecurrentConfig
from model.monte_carlo_evaluation import get_monte_carlo_metric
from word_embedding.word_embedding import get_embedding
from utils.graphs import accuracy_graph
from utils.tensorboard import create_unique_name
from preprocessing.dataset import load


class ModelManager:

    def __init__(self, model_params, verbose=True):
        self.model_params = model_params
        self.sess = None
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
        self.recurrent_model = RecurrentModel(
            recurrent_config, embedding_matrix, self.verbose)

        saved_model_path = self.model_params['saved_model_folder']

        self.sess = tf.Session()

        self.recurrent_model.build_graph(self.input_pipeline)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        try:
            (best_accuracy, train_accuracies,
                val_accuracies, test_accuracy) = self.recurrent_model.fit(
                    self.sess, self.input_pipeline, saved_model_path)
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

    def __init__(self, model_params, active_learning_params, verbose):
        self.active_learning_params = active_learning_params
        super().__init__(model_params, verbose)

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
        np.random.shuffle(train_indexes)

        labeled_dataset = (train_ids[train_indexes], train_labels[train_indexes],
                           train_sizes[train_indexes])
        unlabeled_dataset = (unlabeled_ids, unlabeled_labels, unlabeled_sizes)
        test_dataset = (test_ids, test_labels, test_sizes)

        return labeled_dataset, unlabeled_dataset, test_dataset

    def unlabeled_uncertainty(self, uncertainty_metric, pool_ids, pool_sizes,
                              num_classes, num_samples=10):
        MonteCarlo = get_monte_carlo_metric(uncertainty_metric)

        monte_carlo_evaluation = MonteCarlo(
                sess=self.sess,
                model=self.recurrent_model,
                data_batch=pool_ids,
                sizes_batch=pool_sizes,
                labels_batch=None,
                num_classes=num_classes,
                num_samples=num_samples,
                max_len=self.active_learning_params['max_len'],
                verbose=True)

        variation_ratios = monte_carlo_evaluation.evaluate()

        return variation_ratios

    def select_samples(self):
        sample_size = self.active_learning_params['sample_size']
        pool_indexes = np.array(
            random.sample(range(0, self.unlabeled_dataset_id.shape[0]), sample_size)
        )

        pool_ids = self.unlabeled_dataset_id[pool_indexes]
        pool_labels = self.unlabeled_dataset_labels[pool_indexes]
        pool_sizes = self.unlabeled_dataset_sizes[pool_indexes]

        return (pool_ids, pool_labels, pool_sizes, pool_indexes)

    def select_new_examples(self, pool_ids, pool_labels, pool_sizes):

        num_passes = self.active_learning_params['num_passes']
        num_queries = self.active_learning_params['num_queries']
        uncertainty_metric = self.active_learning_params['uncertainty_metric']
        num_classes = self.model_params['num_classes']

        unlabeled_uncertainty = self.unlabeled_uncertainty(
            uncertainty_metric, pool_ids, pool_sizes, num_classes, num_passes)
        new_samples = unlabeled_uncertainty.argsort()[-num_queries:][::-1]
        np.random.shuffle(new_samples)
        self.reset_graph()

        pooled_id = pool_ids[new_samples]
        pooled_labels = pool_labels[new_samples]
        pooled_sizes = pool_sizes[new_samples]

        return pooled_id, pooled_labels, pooled_sizes, new_samples

    def update_labeled_dataset(self, pooled_id, pooled_labels, pooled_sizes):
        labeled_id = self.labeled_dataset[0]
        labeled_labels = self.labeled_dataset[1]
        labeled_sizes = self.labeled_dataset[2]

        labeled_id = np.concatenate([labeled_id, pooled_id], axis=0)
        labeled_labels = np.concatenate([labeled_labels, pooled_labels], axis=0)
        labeled_sizes = np.concatenate([labeled_sizes, pooled_sizes], axis=0)

        return labeled_id, labeled_labels, labeled_sizes

    def remove_data_from_dataset(self, ids, labels, sizes, data_index):
        deleted_ids = np.delete(ids, (data_index), axis=0)
        deleted_labels = np.delete(labels, (data_index), axis=0)
        deleted_sizes = np.delete(sizes, (data_index), axis=0)

        return deleted_ids, deleted_labels, deleted_sizes

    def update_unlabeled_dataset(self, delete_id, delete_id_sample, delete_labels,
                                 delete_labels_sample, delete_sizes, delete_sizes_sample):
        self.unlabeled_dataset_id = np.concatenate([delete_id, delete_id_sample], axis=0)
        self.unlabeled_dataset_labels = np.concatenate([delete_labels, delete_labels_sample],
                                                       axis=0)
        self.unlabeled_dataset_sizes = np.concatenate([delete_sizes, delete_sizes_sample], axis=0)

    def set_random_seeds(self):
        metric = self.active_learning_params['uncertainty_metric']

        if metric == 'variation_ratio':
            random.seed(9011)
        elif metric == 'entropy':
            random.seed(2001)
        elif metric == 'bald':
            random.seed(5001)
        elif metric == 'random':
            random.seed(9001)
        elif metric == 'softmax':
            return random.seed(7001)

    def run_cycle(self):
        self.set_random_seeds()

        (self.labeled_dataset, unlabeled_dataset,
         test_dataset) = self.create_initial_dataset()

        self.unlabeled_dataset_id = unlabeled_dataset[0]
        self.unlabeled_dataset_labels = unlabeled_dataset[1]
        self.unlabeled_dataset_sizes = unlabeled_dataset[2]

        test_accuracies, train_data_sizes = [], []
        self.model_params['num_validation'] = self.active_learning_params['sample_size']
        self.model_params['test_data'] = test_dataset

        uncertainty_metric = self.active_learning_params['uncertainty_metric']
        uncertainty_metric = uncertainty_metric.replace('_', ' ').title()

        print('Running Active Learning with {} metric'.format(uncertainty_metric))

        for i in range(self.active_learning_params['num_rounds']):
            print('Starting round {}'.format(i))

            pool_ids, pool_labels, pool_sizes, pool_indexes = self.select_samples()
            pool_dataset = (pool_ids, pool_labels, pool_sizes)
            self.model_params['train_data'] = self.labeled_dataset
            self.model_params['validation_data'] = pool_dataset

            _, _, _, test_accuracy = self.run_model()
            test_accuracies.append(test_accuracy)

            sample_id, sample_labels, sample_sizes, sample_index = self.select_new_examples(
                pool_ids, pool_labels, pool_sizes)
            train_data_sizes.append(len(self.labeled_dataset[0]))

            labeled_id, labeled_labels, labeled_sizes = self.update_labeled_dataset(
                sample_id, sample_labels, sample_sizes)
            self.labeled_dataset = (labeled_id, labeled_labels, labeled_sizes)

            (delete_id_sample, delete_labels_sample,
             delete_sizes_sample) = self.remove_data_from_dataset(pool_ids, pool_labels,
                                                                  pool_sizes, sample_index)

            delete_id, delete_labels, delete_sizes = self.remove_data_from_dataset(
                self.unlabeled_dataset_id, self.unlabeled_dataset_labels,
                self.unlabeled_dataset_sizes, pool_indexes)

            self.unlabeled_dataset_id = np.concatenate([delete_id, delete_id_sample], axis=0)
            self.unlabeled_dataset_labels = np.concatenate(
                [delete_labels, delete_labels_sample], axis=0)
            self.unlabeled_dataset_sizes = np.concatenate(
                [delete_sizes, delete_sizes_sample], axis=0)

            self.update_unlabeled_dataset(
                delete_id, delete_id_sample,
                delete_labels, delete_labels_sample,
                delete_sizes, delete_sizes_sample)

            print('End of round {}'.format(i))
            print('Size of pool {}'.format(self.unlabeled_dataset_id.shape[0]))
            print('Train data size: {}'.format(len(labeled_id)))
            print()

        return train_data_sizes, test_accuracies
