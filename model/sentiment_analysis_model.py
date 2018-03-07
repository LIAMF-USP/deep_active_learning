import os

import tensorflow as tf

from model.model import Model
from utils.progress_bar import Progbar


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
        self.num_validation = user_args.get('num_validation', 0)
        self.num_test = user_args['num_test']
        self.use_test = user_args['use_test']
        self.model_name = user_args['model_name']
        self.use_validation = user_args['use_validation']
        self.use_mc_dropout = user_args['use_mc_dropout']


class SentimentAnalysisModel(Model):
    """
    Implements basic functionalities for Sentiment Analysis Task.
    """

    def __init__(self, config, verbose=True):
        super().__init__()

        self.config = config
        self.verbose = verbose

    def batch_evaluate(self, sess, acc, acc_size):
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

    def evaluate(self, sess, total_batch, acc, acc_size):
        """
        This method will be used to calculate the accuracy metric over
        a batch of examples.

        In order for the accuracy to be right, an weight average must be used
        to calculate the final accuracy.
        """
        ac_accuracy, ac_total = 0, 0

        if self.verbose:
            i = 0
            progbar = Progbar(target=total_batch)

        while True:
            try:
                accuracy, size = self.batch_evaluate(sess, acc, acc_size)

                ac_accuracy += accuracy
                ac_total += size

                if self.verbose:
                    i += 1
                    progbar.update(i, [])

            except tf.errors.OutOfRangeError:
                return ac_accuracy / ac_total

    def run_epoch(self, sess, dataset, epoch, total_batch):
        if self.verbose:
            progbar = Progbar(target=total_batch)
            i = 0

        while True:
            try:
                loss = self.train_on_batch(sess)

                if self.verbose:
                    i += 1
                    progbar.update(i, [('train loss', loss)])

            except tf.errors.OutOfRangeError:
                break

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
            sess.run(dataset.test_iterator.initializer)
            total_batch = dataset.test_batches
            accuracy = self.evaluate(sess, total_batch, self.test_acc, self.test_acc_size)
            print('Test Accuracy: {}'.format(accuracy))

            return accuracy

    def fit(self, sess, dataset, saved_model_path=None):
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
            sess.run(dataset.train_iterator.initializer)
            self.run_epoch(sess, dataset, epoch, total_batch)

            if self.config.use_validation:
                sess.run(dataset.train_iterator.initializer)
                total_batch = dataset.train_batches
                train_accuracy = self.evaluate(sess, total_batch,
                                               self.train_acc, self.train_acc_size)
                train_accuracies.append(train_accuracy)

                total_batch = dataset.validation_batches
                sess.run(dataset.validation_iterator.initializer)

                val_accuracy = self.evaluate(sess, total_batch,
                                             self.validation_acc, self.validation_acc_size)

                val_accuracies.append(val_accuracy)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    self.saver.save(sess, saved_model_path)

                print('Train Accuracy for epoch {}: {}'.format(epoch, train_accuracy))
                print('Validation Accuracy for epoch {}: {}'.format(epoch, val_accuracy))
                print()

        test_accuracy = -1
        if self.config.use_test:
            test_accuracy = self.run_test_accuracy(sess, dataset)

        return best_accuracy, train_accuracies, val_accuracies, test_accuracy

    def build_graph(self, dataset):
        with tf.name_scope('placeholders'):
            self.add_placeholder()

        with tf.name_scope('iterators'):
            train_iterator = dataset.train_iterator
            validation_iterator = dataset.validation_iterator
            test_iterator = dataset.test_iterator

        with tf.name_scope('train_data'):
            train_data, train_labels, train_sizes = train_iterator.get_next()

        with tf.name_scope('train'):
            train_logits = self.get_logits(train_data, train_sizes)
            self.loss = self.add_loss_op(train_logits, train_labels)
            self.train = self.add_training_op(self.loss)

        with tf.name_scope('train_accuracy'):
            self.train_acc, self.train_acc_size = self.add_evaluation_op(
                train_logits, train_labels)

        with tf.name_scope('validation'):
            validation_data, validation_labels, validation_sizes = validation_iterator.get_next()
            validation_logits = self.get_logits(validation_data, validation_sizes, reuse=True)
            # validation_loss = self.add_loss_op(validation_loss)

            self.validation_acc, self.validation_acc_size = self.add_evaluation_op(
                validation_logits, validation_labels)

        with tf.name_scope('test_accuracy'):
            test_data, test_labels, test_sizes = test_iterator.get_next()
            test_logits = self.get_logits(test_data, test_sizes, reuse=True)

            self.test_acc, self.test_acc_size = self.add_evaluation_op(
                test_logits, test_labels)

        with tf.name_scope('prediction'):
            prediction_logits = self.get_logits(
                self.data_placeholder, self.sizes_placeholder, reuse=True)
            self.predictions_distribution = tf.nn.softmax(prediction_logits)
            self.predictions = tf.argmax(prediction_logits, axis=1)

        with tf.name_scope('summary'):
            self.summ = tf.summary.merge_all()
