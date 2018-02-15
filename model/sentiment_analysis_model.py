import os

import numpy as np
import tensorflow as tf

from model.model import Model
from utils.progress_bar import Progbar
from utils.tensorboard import add_array_to_summary_writer


class Config:
    """
    Holds model hyperparameters and data information.

    The hyperparameters are defined by the user when running a model.
    Every model object receives a config object at instantiation.
    """

    def __init__(self, user_args):
        self.learning_rate = user_args['learning_rate']
        self.batch_size = user_args['batch_size']
        self.num_epochs = user_args['num_epochs']
        self.embed_size = user_args['embed_size']
        self.num_classes = user_args['num_classes']
        self.num_train = user_args['num_train']
        self.num_validation = user_args['num_validation']
        self.num_test = user_args['num_test']
        self.use_test = user_args['use_test']
        self.model_name = user_args['model_name']


class SentimentAnalysisModel(Model):
    """
    Implements basic functionalities for Sentiment Analysis Task.
    """

    def __init__(self, config, verbose=True):
        super().__init__()

        self.config = config
        self.verbose = verbose

    def batch_evaluate(self, sess, batch_data, batch_labels):
        """
        This method is used to calculate en evaluation metric over a batch of
        data.

        Args:
            sess: A Tensorflow session
            batch_data: The batch of data to be feed to the model
            batch_labels: The labels corresponding to each data in the batch
        Returns:
            metric_value: The value of the metric for the batch
            size: The size of the batch used to calculate this metric.
        """
        return NotImplementedError()

    def evaluate(self, sess, total_batch):
        """
        This method will be used to calculate the accuracy metric over
        a batch of examples from the validation or test set.

        In order for the accuracy to be right, an weight average must be used
        to calculate the final accuracy.

        Args:
            sess: A Tensorflow session
            dataset: A tf.data.Dataset object containing the validation or
                     test data
            total_batch: Variable indicating number of batches for the dataset
        Returns:
            accuracy: The mean accuracy for the dataset
        """
        ac_accuracy, ac_total = 0, 0

        if self.verbose:
            i = 0
            progbar = Progbar(target=total_batch)

        while True:
            try:
                accuracy, size = self.batch_evaluate(sess)

                ac_accuracy += accuracy
                ac_total += size

                if self.verbose:
                    i += 1
                    progbar.update(i, [])

            except tf.errors.OutOfRangeError:
                return ac_accuracy / ac_total

    def monte_carlo_samples(self, sess, dataset, num_samples):
        all_preds = np.zeros(shape=(self.config.num_validation, num_samples))
        all_labels = np.zeros(shape=(self.config.num_validation))

        for i in range(num_samples):
            batch_pos = 0
            sess.run(dataset.validation_iterator)

            if self.verbose:
                progbar = Progbar(target=num_samples)

            while True:
                try:
                    feed = self.create_feed_dict()
                    preds, labels = sess.run([self.pred, self.labels], feed_dict=feed)

                    preds = np.argmax(preds, axis=1)

                    batch_aux = batch_pos + preds.shape[0]
                    all_preds[batch_pos:batch_aux, i] = preds
                    all_labels[batch_pos:batch_aux] = labels

                    batch_pos = batch_aux
                except tf.errors.OutOfRangeError:
                    break

            if self.verbose:
                progbar.update(i + 1, [])

        return all_preds, all_labels

    def monte_carlo_samples_count(self, all_preds):
        mc_counts = []

        all_preds = all_preds.astype(dtype=np.int64)

        for row in all_preds:
            bincount = np.bincount(row)
            mc_counts.append((bincount, bincount.argmax()))

        return mc_counts

    def monte_carlo_dropout_evaluate(self, sess, dataset):
        all_preds, all_labels = self.monte_carlo_samples(sess, dataset, num_samples=10)
        mc_counts = self.monte_carlo_samples_count(all_preds)

        predictions = np.zeros(shape=(self.config.num_validation))
        for index, (bincount, value) in enumerate(mc_counts):
            predictions[index] = value

        correct_pred = np.equal(predictions, all_labels)

        return np.mean(correct_pred)

    def run_epoch(self, sess, dataset, writer, epoch, total_batch):
        if self.verbose:
            progbar = Progbar(target=total_batch)
            i = 0

        while True:
            try:
                loss, s = self.train_on_batch(sess)

                if i % 20 == 0:
                    index = (epoch * total_batch) + i
                    writer.add_summary(s, index)

                if self.verbose:
                    i += 1
                    progbar.update(i, [('train loss', loss)])

            except tf.errors.OutOfRangeError:
                break

    def prepare(self, sess, dataset):
        sess.run(dataset.train_iterator)
        data_batch, labels_batch, size_batch = dataset.make_batch()
        self.build_graph(data_batch, labels_batch, size_batch)

    def create_saver(self, saved_model_path):
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)

        self.saver = tf.train.Saver()

    def check_saved_model(self, sess, dataset, saved_model_path):
        if os.path.exists(saved_model_path + '.index'):
            print('Loading saved model ...')
            self.saver.restore(sess, saved_model_path)

            sess.run(dataset.validation_iterator)
            total_batch = dataset.validation_batches
            val_accuracy = self.evaluate(sess, total_batch)

            sess.run(dataset.train_iterator)
            total_batch = dataset.train_batches
            train_accuracy = self.evaluate(sess, total_batch)

            print('Train accuracy for saved model: {:.3f}'.format(train_accuracy))
            print('Validation accuracy for saved model: {:.3f}'.format(val_accuracy))

            return True

        return False

    def run_test_accuracy(self, sess, dataset):
        if self.config.use_test:
            sess.run(dataset.test_iterator)
            total_batch = dataset.test_batches
            accuracy = self.evaluate(sess, total_batch)
            print('Test Accuracy: {}'.format(accuracy))

            return accuracy

    def fit(self, sess, dataset, saved_model_path=None, writer=None):
        train_accuracies = []
        val_accuracies = []
        print('Training model...')

        self.create_saver(saved_model_path)
        saved_model_path = os.path.join(
            saved_model_path, self.config.model_name + '.ckpt')

        if self.check_saved_model(sess, dataset, saved_model_path):
            best_accuracy = self.run_test_accuracy(sess, dataset)
            return best_accuracy, train_accuracies, val_accuracies

        best_accuracy = -1

        for epoch in range(self.config.num_epochs):
            print('Running epoch {}'.format(epoch))
            total_batch = dataset.train_batches
            self.run_epoch(sess, dataset, writer, epoch, total_batch)

            sess.run(dataset.train_iterator)
            train_accuracy = self.evaluate(sess, total_batch)

            total_batch = dataset.validation_batches
            sess.run(dataset.validation_iterator)
            val_accuracy = self.evaluate(sess, total_batch)

            mc_accuracy = self.monte_carlo_dropout_evaluate(sess, dataset)

            val_accuracies.append(val_accuracy)
            train_accuracies.append(train_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.saver.save(sess, saved_model_path)

            print('Train Accuracy for epoch {}: {}'.format(epoch, train_accuracy))
            print('Validation Accuracy for epoch {}: {}'.format(epoch, val_accuracy))
            print('Validation Accuracy (MC Dropout) for epoch {}: {}'.format(epoch, mc_accuracy))
            print()

            sess.run(dataset.train_iterator)

        add_array_to_summary_writer(writer, val_accuracies, 'val_accuracy')
        add_array_to_summary_writer(writer, train_accuracies, 'train_accuracy')

        if self.config.use_test:
            self.run_test_accuracy(sess, dataset)

        return best_accuracy, train_accuracies, val_accuracies
