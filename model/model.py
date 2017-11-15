class Model:
    """
    Abstracts a Tensorflow graph for a learning task.

    This base class assumes that the Dataset API is being used
    to handle data, since no placeholders are being used.
    """
    def add_data_op(self, batch_data, batch_labels=None):
        """
        Method responsible for handling the batch data.

        Args:
            batch_data: A tensor with shape (batch_size, num_features)
            batch_labels: A tensor with shape (batch_size, num_classes)
                          This argument can be None since it will not
                          be used to generate predictions, only to train
                          the model.
        """
        self.batch_data = batch_data
        self.batch_labels = batch_labels

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
        self.add_data_op(batch_data, batch_labels)
        _, loss = sess.run([self.train_op, self.loss])

        return loss

    def predict_on_batch(self, sess, batch_data):
        """
        Make predictions for the provided batch of data

        Args:
            sess: A Tensorflow session
            batch_data: A tensor with shape (batch_size, num_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        self.add_data_op(batch_data)
        predictions = sess.run(self.pred)

        return predictions

    def build_graph(self):
        """
        Create the computational graph for the model.
        """
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train = self.add_training_op(self.loss)
