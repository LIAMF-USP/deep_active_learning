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

    def add_embedding(self, inputs):
        """
        Adds and embedding layer that map the sentences id list to word vectors.
        """

        with tf.name_scope('embeddings'):
            base_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(base_embeddings, inputs)

        return inputs

    def add_prediction_op(self, inputs):
        num_units = self.config.num_units
        num_classes = self.config.num_classes

        x_hat = self.add_embedding(inputs)

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
            tf.summary.histogram('lstm_output', lstm_output)

        with tf.name_scope('output_layer'):
            weight = tf.Variable(
                tf.truncated_normal([num_units, num_classes]),
                name='weight')
            bias = tf.Variable(
                tf.constant(0.1, shape=[num_classes]),
                name='bias')

            prediction = tf.matmul(lstm_output, weight) + bias

            tf.summary.histogram('output_weight', weight)
            tf.summary.histogram('output_bias', bias)
            tf.summary.histogram('prediction', prediction)

        return prediction

    def add_loss_op(self, pred, labels):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=pred,
                    labels=labels),
                name='cross_entropy')

            tf.summary.scalar('loss', loss)

        return loss

    def add_training_op(self, loss):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss)

        return train

    def add_evaluation_op(self, labels):
        with tf.name_scope('validation'):
            predictions = tf.cast(tf.argmax(self.pred, axis=1), tf.int32)
            size = tf.cast(tf.shape(predictions)[0], tf.float32)
            correct_pred = tf.equal(predictions, tf.cast(labels, tf.int32))

            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * size
            self._size = size

    def batch_evaluate(self, sess):
        accuracy, total = sess.run([self._accuracy, self._size])

        return accuracy, total
