import tensorflow as tf

from math import ceil

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
        self.batch_size = user_args['batch_size']
        self.num_epochs = user_args['num_epochs']
        self.embed_size = user_args['embed_size']
        self.num_classes = user_args['num_classes']
        self.num_train = user_args['num_train']
        self.num_validation = user_args['num_validation']
        self.num_test = user_args['num_test']
        self.use_test = user_args['use_test']


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

        if dataset_type == 'train':
            target = self.config.num_train
        elif dataset_type == 'val':
            target = self.config.num_validation
        else:
            target = self.config.num_test

        if self.verbose:
            i = 0
            progbar = Progbar(target=ceil(target / self.config.batch_size))

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

    def run_epoch(self, sess, dataset, writer, epoch):
        total_batch = ceil(self.config.num_train / self.config.batch_size)

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
        batch_data, batch_labels = dataset.make_batch()
        self.build_graph(batch_data, batch_labels)

    def fit(self, sess, dataset, writer=None):
        train_accuracies = []
        val_accuracies = []
        print('Training model...')

        for epoch in range(self.config.num_epochs):
            print('Running epoch {}'.format(epoch))
            self.run_epoch(sess, dataset, writer, epoch)

            sess.run(dataset.train_iterator)
            train_accuracy = self.evaluate(sess, dataset, 'train')

            sess.run(dataset.validation_iterator)
            val_accuracy = self.evaluate(sess, dataset, 'val')
            val_accuracies.append(val_accuracy)
            train_accuracies.append(train_accuracy)

            print('Train Accuracy for epoch {}: {}'.format(epoch, train_accuracy))
            print('Validation Accuracy for epoch {}: {}'.format(epoch, val_accuracy))
            print()

            sess.run(dataset.train_iterator)

        add_array_to_summary_writer(writer, val_accuracies, 'val_accuracy')
        add_array_to_summary_writer(writer, train_accuracies, 'train_accuracy')

        if self.config.use_test:
            sess.run(dataset.test_iterator)
            accuracy = self.evaluate(sess, dataset, 'test')
            print('Test Accuracy: {}'.format(accuracy))

        return train_accuracies, val_accuracies
