import tensorflow as tf

from math import ceil

from model.model import Model
from utils.progress_bar import Progbar


class Config:
    """
    Holds model hyperparameters and data information.

    The hyperparameters are defined by the user when running a model.
    Every model object receives a config object at instantiation.
    """

    def __init__(self, user_args):
        self.batch_size = user_args['batch_size']
        self.num_epochs = user_args['num_epochs']
        self.embed_size = user_args['embed_size']
        self.num_classes = user_args['num_classes']
        self.num_train = user_args['num_train']
        self.num_validation = user_args['num_validation']
        self.num_test = user_args['num_test']


class SentimentAnalysisModel(Model):
    """
    Implements basic functionalities for Sentiment Analysis Task.
    """

    def __init__(self, config, verbose=True):
        super().__init__()

        self.config = config
        self.verbose = verbose

    def evaluate(self, sess, dataset, dataset_type='val'):
        """
        This method will be used to calculate the accuracy metric over
        a batch of examples from the validation or test set.

        In order for the accuracy to be right, an weight average must be used
        to calculate the final accuracy.

        Args:
            sess: A Tensorflow session
            dataset: A tf.data.Dataset object containing the validation or
                     test data
            dataset_type: A variable indicating which dataset is being evaluated
        Returns:
            accuracy: The mean accuracy for the dataset
        """
        ac_accuracy, ac_total = 0, 0

        if dataset_type == 'val':
            target = self.config.num_validation

        if self.verbose:
            i = 0
            progbar = Progbar(target=ceil(target / self.config.batch_size))

        while True:
            try:
                batch_data, batch_labels = sess.run(dataset.get_batch())

                predictions = tf.argmax(self.pred, axis=1)
                weight = tf.cast(tf.shape(predictions)[0], tf.float32)
                correct_pred = tf.equal(predictions, batch_labels)

                _accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * weight
                _total = weight

                feed = self.create_feed_dict(batch_data, batch_labels)
                accuracy, total = sess.run([_accuracy, _total], feed_dict=feed)

                ac_accuracy += accuracy
                ac_total += total

                if self.verbose:
                    i += 1
                    progbar.update(i, [])

            except tf.errors.OutOfRangeError:
                return ac_accuracy / ac_total

    def run_epoch(self, sess, dataset):

        if self.verbose:
            target = ceil(self.config.num_train / self.config.batch_size)
            progbar = Progbar(target=target)
            i = 0

        while True:
            try:
                batch_data, batch_labels = sess.run(dataset.get_batch())
                loss = self.train_on_batch(sess, batch_data, batch_labels)

                if self.verbose:
                    i += 1
                    progbar.update(i, [('train loss', loss)])

            except tf.errors.OutOfRangeError:
                break

    def fit(self, sess, dataset):
        best_score = 0
        print('Training model...')

        for epoch in range(self.config.num_epochs):
            print('Running epoch {}'.format(epoch))
            sess.run(dataset.train_iterator)
            self.run_epoch(sess, dataset)

            sess.run(dataset.validation_iterator)
            accuracy = self.evaluate(sess, dataset)

            print('Accuracy for epoch {}: {}'.format(epoch, accuracy))
            print()

            if accuracy > best_score:
                best_score = accuracy

        return best_score
