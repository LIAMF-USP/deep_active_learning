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
        Add placeholder variables to tensorflow computational graph.
        """
        raise NotImplementedError()

    def create_feed_dict(self, data_batch, labels_batch=None):
        """
        Create the feed dict for one step of training.
        """
        raise NotImplementedError()

    def add_prediction_op(self):
        """"
        Method responsible for transforming a batch of data into predictions.

        Returns:
            pred: A tensor with shape (batch_size, num_classes)
        """
        raise NotImplementedError()

    def add_loss_op(self, pred):
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

    def train_on_batch(self, sess, batch_data, batch_labels):
        """
        Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: A Tensorflow session
            batch_data: A tensor with shape (batch_size, num_features)
            batch_labels: A tensor with shape (batch_size, num_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(batch_data, batch_labels)
        _, loss = sess.run([self.train, self.loss], feed_dict=feed)
        return loss

    def build_graph(self):
        """
        Create the computational graph for the model.
        """
        self.add_placeholder()
        self.pred = self.add_prediction_op()  # Too slow, look into
        self.loss = self.add_loss_op(self.pred)
        self.train = self.add_training_op(self.loss)
