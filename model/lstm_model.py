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
        self.batch_size = user_args['batch_size']
        self.max_length = user_args['max_length']


class LSTMModel(SentimentAnalysisModel):

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        self.pretrained_embeddings = pretrained_embeddings

        self.build_graph()
        print('Saiu aqui')

    def add_placeholder(self):
        max_length = self.config.max_length

        self.data_placeholder = tf.placeholder(
            dtype=tf.int32, shape=[None, max_length])
        self.labels_placeholder = tf.placeholder(
            dtype=tf.int32, shape=[None])

    def create_feed_dict(self, data_batch, labels_batch=None):
        feed_dict = {self.data_placeholder: data_batch}

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embedding(self):
        """
        Adds and embedding layer that map the sentences id list to word vectors.
        """

        base_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(base_embeddings, self.data_placeholder)

        return inputs

    def add_prediction_op(self):
        num_units = self.config.num_units
        num_classes = self.config.num_classes

        x_hat = self.add_embedding()
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)

        seqlen = sequence_length(x_hat)  # Slow graph creation, investigate

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
        lstm_output = cell.h

        weight = tf.Variable(tf.truncated_normal([num_units, num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        prediction = tf.matmul(lstm_output, weight) + bias

        return prediction

    def add_loss_op(self, pred):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pred,
                labels=self.labels_placeholder))

        return loss

    def add_training_op(self, loss):
        train = tf.train.AdamOptimizer().minimize(loss)

        return train
