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

    def evaluate(self, sess, dataset, total_batch):
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

    def fit(self, sess, dataset, writer=None):
        train_accuracies = []
        val_accuracies = []
        print('Training model...')

        for epoch in range(self.config.num_epochs):
            print('Running epoch {}'.format(epoch))
            total_batch = dataset.train_batches
            self.run_epoch(sess, dataset, writer, epoch, total_batch)

            sess.run(dataset.train_iterator)
            train_accuracy = self.evaluate(sess, dataset, total_batch)

            sess.run(dataset.validation_iterator)
            total_batch = dataset.validation_batches
            val_accuracy = self.evaluate(sess, dataset, total_batch)
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
            total_batch = dataset.test_batches
            accuracy = self.evaluate(sess, dataset, total_batch)
            print('Test Accuracy: {}'.format(accuracy))

        return train_accuracies, val_accuracies
