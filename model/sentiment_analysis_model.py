import tensorflow as tf

from model.model import Model


class Config:
    """
    Holds model hyperparameters and data information.

    The hyperparameters are defined by the user when running a model.
    Every model object receives a config object at instantiation.
    """

    def __init__(self, user_args):
        self.batch_size = user_args['batch_size']
        self.num_epochs = user_args['num_epochs']
        self.dropout = user_args['dropout']
        self.embed_size = user_args['embed_size']
        self.learning_rate = user_args['learning_rate']


class SentimentAnalysisModel(Model):
    """
    Implements basic functionalities for Sentiment Analysis Task.
    """

    def __init__(self, config):
        self.config = config

    def preprocess_sequence_data(self, examples):
        raise NotImplementedError()

    def evaluate(self, dataset):
        """
        This method will be used to calculate the accuracy metric over
        a batch of examples from the validation or test set.

        In order for the accuracy to be right, an weight average must be used
        to calculate the final accuracy.

        Args:
            sess: A Tensorflow session
            dataset: A tf.data.Dataset object containing the validation or
                     test data
        Returns:
            accuracy: The mean accuracy for the dataset
        """
        accuracy, total = 0, 0

        while True:
            try:
                batch_data, batch_labels = dataset.get_batch()
                self.add_data_op(batch_data)

                predictions = tf.argmax(self.pred, axis=1)
                weight = tf.cast(tf.shape(predictions)[0], tf.float32)
                correct_pred = tf.equal(predictions, batch_labels)

                accuracy += tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * weight
                total += weight
            except tf.errors.OutOfRangeError:
                return tf.divide(accuracy, total)

    def run_epoch(self, sess, dataset):
        while True:
            try:
                batch_data, batch_labels = dataset.get_batch()
                loss = self.train_on_batch(sess, batch_data, batch_labels)  # noqa
            except tf.errors.OutOfRangeError:
                break
