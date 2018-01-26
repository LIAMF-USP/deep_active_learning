import tensorflow as tf

from model.sentiment_analysis_model import SentimentAnalysisModel, Config


class RecurrentConfig(Config):

    def __init__(self, user_args):
        super().__init__(user_args)

        self.num_units = user_args['num_units']
        self.batch_size = user_args['batch_size']
        self.recurrent_output_dropout = user_args['recurrent_output_dropout']
        self.recurrent_state_dropout = user_args['recurrent_state_dropout']
        self.embedding_dropout = user_args['embedding_dropout']
        self.weight_decay = user_args['weight_decay']


class RecurrentModel(SentimentAnalysisModel):

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        self.pretrained_embeddings = pretrained_embeddings

    def add_placeholder(self):
        self.recurrent_output_dropout_placeholder = tf.placeholder(tf.float32)
        self.recurrent_state_dropout_placeholder = tf.placeholder(tf.float32)
        self.embedding_dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, recurrent_output_dropout=None, recurrent_state_dropout=None,
                         embedding_dropout=None):
        if recurrent_output_dropout is None:
            recurrent_output_dropout = self.config.recurrent_output_dropout

        if recurrent_state_dropout is None:
            recurrent_state_dropout = self.config.recurrent_state_dropout

        if embedding_dropout is None:
            embedding_dropout = self.config.embedding_dropout

        feed_dict = {self.recurrent_output_dropout_placeholder: recurrent_output_dropout,
                     self.recurrent_state_dropout_placeholder: recurrent_state_dropout,
                     self.embedding_dropout_placeholder: embedding_dropout}

        return feed_dict

    def add_embedding(self, inputs):
        """
        Adds and embedding layer that map the sentences id list to word vectors.
        """

        with tf.name_scope('embeddings'):
            base_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
            embeddings_dropout = tf.nn.dropout(base_embeddings, self.embedding_dropout_placeholder)
            inputs = tf.nn.embedding_lookup(embeddings_dropout, inputs)

        return inputs

    def add_prediction_op(self, inputs, size):
        num_units = self.config.num_units
        num_classes = self.config.num_classes

        x_hat = self.add_embedding(inputs)

        with tf.name_scope('recurrent_layer'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
            drop_recurrent_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell,
                    output_keep_prob=self.recurrent_output_dropout_placeholder,
                    state_keep_prob=self.recurrent_state_dropout_placeholder,
                    variational_recurrent=True,
                    dtype=tf.float32)

            """
            The dynamic_rnn outputs a 0 for every output after it has reached
            the end of a sentence. When doing sequence classification,
            we need the last relevant value from the outputs matrix.

            Since we are using a single recurrent cell, this can be achieved by getting
            the output from the cell.h returned from the dynamic_rnn method.
            """
            _, cell = tf.nn.dynamic_rnn(drop_recurrent_cell, x_hat,
                                        sequence_length=size,
                                        dtype=tf.float32)

            """
            This variable will have shape [batch_size, num_units]
            """
            recurrent_output = cell.h
            tf.summary.histogram('recurrent_output', recurrent_output)

        with tf.name_scope('output_layer'):
            initializer = tf.contrib.layers.xavier_initializer()
            weight = tf.Variable(
                initializer(shape=[num_units, num_classes]),
                name='weight')
            bias = tf.Variable(
                initializer(shape=[num_classes]),
                name='bias')

            prediction = tf.matmul(recurrent_output, weight) + bias

            tf.summary.histogram('output_weight', weight)
            tf.summary.histogram('output_bias', bias)
            tf.summary.histogram('prediction', prediction)

        return prediction

    def add_loss_op(self, pred, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=pred,
                    labels=labels),
                name='cross_entropy')

            l2_loss = self.config.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            loss = cross_entropy + l2_loss

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
        feed = self.create_feed_dict(recurrent_output_dropout=1.0, recurrent_state_dropout=1.0,
                                     embedding_dropout=1.0)
        accuracy, total = sess.run([self._accuracy, self._size], feed_dict=feed)

        return accuracy, total
