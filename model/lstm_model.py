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

    def add_placeholder(self):
        max_length = self.config.max_length

        with tf.name_scope('placeholders'):
            self.data_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None, max_length],
                name='data')
            self.labels_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None],
                name='labels')

    def create_feed_dict(self, data_batch, labels_batch=None):
        feed_dict = {self.data_placeholder: data_batch}

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embedding(self):
        """
        Adds and embedding layer that map the sentences id list to word vectors.
        """

        with tf.name_scope('embeddings'):
            base_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(base_embeddings, self.data_placeholder)

        return inputs

    def add_prediction_op(self):
        num_units = self.config.num_units
        num_classes = self.config.num_classes

        x_hat = self.add_embedding()

        with tf.name_scope('lstm_layer'):
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

        with tf.name_scope('output_layer'):
            weight = tf.Variable(
                tf.truncated_normal([num_units, num_classes]),
                name='weight')
            bias = tf.Variable(
                tf.constant(0.1, shape=[num_classes]),
                name='bias')

            prediction = tf.matmul(lstm_output, weight) + bias

        return prediction

    def add_loss_op(self, pred):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=pred,
                    labels=self.labels_placeholder),
                name='cross_entropy')

        return loss

    def add_training_op(self, loss):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss)

        return train

    def add_evaluation_op(self):
        with tf.name_scope('validation'):
            predictions = tf.cast(tf.argmax(self.pred, axis=1), tf.int32)
            size = tf.cast(tf.shape(predictions)[0], tf.float32)
            correct_pred = tf.equal(predictions, self.labels_placeholder)

            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * size
            self._size = size

    def batch_evaluate(self, sess, batch_data, batch_labels):
        feed = self.create_feed_dict(batch_data, batch_labels)
        accuracy, total = sess.run([self._accuracy, self._size], feed_dict=feed)

        return accuracy, total
