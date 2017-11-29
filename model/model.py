import tensorflow as tf


class Model:
    """
    Abstracts a Tensorflow graph for a learning task.

    This base class assumes that the Dataset API is being used
    to handle data, since no placeholders are being used.
    """

    def __init__(self):
        self.batch_data = None
        self.batch_labels = None

    def add_placeholder(self):
        """
        Method responsible for creating placeholder for the model.
        """
        raise NotImplementedError()

    def create_feed_dict(self):
        """
        Method responsible for creating the feed_dict for the model.
        """
        raise NotImplementedError()

    def add_evaluation_op(self, labels):
        """
        Method responsible for creating a graph to evaluate the model.
        """
        raise NotImplementedError()

    def add_prediction_op(self, inputs):
        """"
        Method responsible for transforming a batch of data into predictions.

        Returns:
            pred: A tensor with shape (batch_size, num_classes)
        """
        raise NotImplementedError()

    def add_loss_op(self, pred, labels):
        """
        Method responsible for adding a loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError()

    def add_training_op(self, loss):
        """
        Method responsible for adding the training operation to the
        computational graph.

        Args:
            loss: Loss Tensor (a scalar)
        Returns:
            train_op: The operation for training the model
        """

        raise NotImplementedError()

    def train_on_batch(self, sess):
        """
        Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: A Tensorflow session
            batch_data: A tensor with shape (batch_size, num_features)
            batch_labels: A tensor with shape (batch_size, num_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """

        feed = self.create_feed_dict()
        _, loss, summary = sess.run([self.train, self.loss, self.summ], feed_dict=feed)
        return loss, summary

    def build_graph(self, inputs, labels):
        """
        Create the computational graph for the model.
        """
        self.add_placeholder()
        self.pred = self.add_prediction_op(inputs)  # Too slow, look into
        self.loss = self.add_loss_op(self.pred, labels)
        self.train = self.add_training_op(self.loss)

        self.add_evaluation_op(labels)
        self.summ = tf.summary.merge_all()
