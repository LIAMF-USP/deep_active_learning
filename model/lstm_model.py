import tensorflow as tf

from model.sentiment_analysis_model import SentimentAnalysisModel, Config


def sequence_length(x_hat):
    used = tf.sign(tf.reduce_max(tf.abs(x_hat), 2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)

    return length


class LSTMConfig(Config):

    def __init__(self, user_args):
        super().__init__(user_args)

        self.num_units = user_args['num_units']
        self.time_steps = user_args['time_steps']
        self.num_features = user_args['num_features']
        self.batch_size = user_args['batch_size']


class LSTMModel(SentimentAnalysisModel):

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        self.pretrained_embeddings = pretrained_embeddings

    def add_embedding(self):
        """
        Adds and embedding layer that map the sentences id list to word vectors.
        """

        base_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(base_embeddings, self.batch_data)

        return inputs

    def lstm_layer(self, x_hat, num_units):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)

        seqlen = sequence_length(x_hat)

        """
        The dynamic_rnn outputs a 0 for every output after it has reached
        the end of a sentence. When doing sequence classification,
        we need the last relevant value from the outputs matrix.

        Since we are using a single LSTM cell, this can be achieved by getting
        the output from the cell.h returned from the dynamic_rnn method.
        """
        _, cell = tf.nn.dynamic_rnn(lstm_cell, x_hat,
                                    sequence_length=seqlen,
                                    dtype=tf.float32)

        """
        This variable will have shape [batch_size, num_units]
        """
        self._lstm_output = cell.h

    def add_prediction_op(self):
        num_units = self.config.num_units
        num_classes = self.config.num_classes

        x_hat = self.add_embedding()
        self.lstm_layer(x_hat, num_units)

        weight = tf.Variable(tf.truncated_normal([num_units, num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        prediction = tf.matmul(self._lstm_output, weight) + bias

        return prediction

    def add_loss_op(self, pred):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pred,
                labels=self.batch_labels))

        return loss

    def add_training_op(self, loss):
        train = tf.train.AdamOptimizer().minimize(loss)

        return train
