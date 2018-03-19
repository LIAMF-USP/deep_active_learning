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
        self.should_save = user_args['should_save']


class SentimentAnalysisModel(Model):
    """
    Implements basic functionalities for Sentiment Analysis Task.
    """

    def __init__(self, config, verbose=True):
        super().__init__()

        self.config = config
        self.verbose = verbose

    def batch_evaluate(self, sess, metric_update):
        """
        This method is used to calculate en evaluation metric over a batch of
        data.

        Args:
            sess: A Tensorflow session
            metric_update: the update_op that allows the metric value to be
            updated over batches of data.
        """
        return NotImplementedError

    def initialize_embeddings(self, sess):
        return NotImplementedError

    def evaluate(self, sess, total_batch, acc, acc_update):
        if self.verbose:
            i = 0
            progbar = Progbar(target=total_batch)

        while True:
            try:
                self.batch_evaluate(sess, acc_update)

                if self.verbose:
                    i += 1
                    progbar.update(i, [])

            except tf.errors.OutOfRangeError:
                return sess.run(acc)

    def run_epoch(self, sess, dataset, epoch):
        sess.run(dataset.train_iterator.initializer)

        if self.verbose:
            total_batch = dataset.train_batches
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

            sess.run(dataset.validation_iterator.initializer)
            total_batch = dataset.validation_batches
            val_accuracy = self.evaluate(sess, total_batch,
                                         self.validation_acc, self.validation_acc_size)

            sess.run(dataset.train_iterator.initializer)
            total_batch = dataset.train_batches
            train_accuracy = self.evaluate(sess, total_batch,
                                           self.train_acc, self.train_acc_size)

            print('Train accuracy for saved model: {:.3f}'.format(train_accuracy))
            print('Validation accuracy for saved model: {:.3f}'.format(val_accuracy))

            return True

        return False

    def run_accuracy(self, sess, iterator, acc, acc_update_op,
                     acc_initializer, total_batch=0):
        sess.run([iterator, acc_initializer])
        accuracy = self.evaluate(sess, total_batch, acc, acc_update_op)

        return accuracy

    def fit(self, sess, dataset, saved_model_path=None):
        train_accuracies = []
        val_accuracies = []
        print('Training model...')

        self.create_saver(saved_model_path)
        saved_model_path = os.path.join(
            saved_model_path, self.config.model_name + '.ckpt')

        best_accuracy = -1
        test_accuracy = -1

        self.initialize_embeddings(sess)

        if self.config.should_save and self.check_saved_model(sess, dataset, saved_model_path):
            best_accuracy = self.run_test_accuracy(sess, dataset)
            return best_accuracy, train_accuracies, val_accuracies, best_accuracy

        for epoch in range(self.config.num_epochs):
            print('Running epoch {}'.format(epoch))
            self.run_epoch(sess, dataset, epoch)

            if self.config.use_validation:
                print('Evaluating model for epoch {} ...'.format(epoch))
                total_batch = dataset.train_batches
                train_accuracy = self.run_accuracy(
                    sess, dataset.train_iterator.initializer, self.train_acc, self.train_acc_op,
                    self.train_acc_initializer, total_batch)
                train_accuracies.append(train_accuracy)

                total_batch = dataset.validation_batches
                val_accuracy = self.run_accuracy(
                    sess, dataset.validation_iterator.initializer, self.validation_acc,
                    self.validation_acc_op, self.val_acc_initializer, total_batch)
                val_accuracies.append(val_accuracy)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy

                    if self.config.should_save:
                        self.saver.save(sess, saved_model_path)

                print('Train Accuracy for epoch {}: {:.3f}'.format(epoch, train_accuracy))
                print('Validation Accuracy for epoch {}: {:.3f}'.format(epoch, val_accuracy))
                print()

        if self.config.use_test:
            total_batch = dataset.test_batches
            test_accuracy = self.run_accuracy(
                sess, dataset.test_iterator.initializer, self.test_acc, self.test_acc_op,
                self.test_acc_initializer, total_batch)
            print('Test accuracy: {:.3f}'.format(test_accuracy))

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
            train_predictions = tf.cast(tf.argmax(train_logits, axis=1), tf.int32)
            (self.train_acc, self.train_acc_op,
             self.train_acc_initializer) = self.add_evaluation_op(
                train_predictions, train_labels, scope='train_accuracy', name='train_acc')

        with tf.name_scope('validation'):
            validation_data, validation_labels, validation_sizes = validation_iterator.get_next()
            validation_logits = self.get_logits(validation_data, validation_sizes, reuse=True)
            validation_predictions = tf.cast(tf.argmax(validation_logits, axis=1), tf.int32)

            (self.validation_acc, self.validation_acc_op,
             self.val_acc_initializer) = self.add_evaluation_op(
                validation_predictions, validation_labels, scope='validation', name='val_acc')

        with tf.name_scope('test_accuracy'):
            test_data, test_labels, test_sizes = test_iterator.get_next()
            test_logits = self.get_logits(test_data, test_sizes, reuse=True)
            test_predictions = tf.cast(tf.argmax(test_logits, axis=1), tf.int32)

            (self.test_acc, self.test_acc_op,
             self.test_acc_initializer) = self.add_evaluation_op(
                test_predictions, test_labels, scope='test_accuracy', name='test_acc')

        with tf.name_scope('prediction'):
            prediction_logits = self.get_logits(
                self.data_placeholder, self.sizes_placeholder, reuse=True)
            self.predictions_distribution = tf.nn.softmax(prediction_logits)
            self.predictions = tf.argmax(prediction_logits, axis=1)
