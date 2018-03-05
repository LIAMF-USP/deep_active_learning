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
        self.clip_gradients = user_args['clip_gradients']
        self.max_norm = user_args['max_norm']


class RecurrentModel(SentimentAnalysisModel):

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        self.pretrained_embeddings = pretrained_embeddings

    def add_placeholder(self):
        self.recurrent_output_dropout_placeholder = tf.placeholder(tf.float32)
        self.recurrent_state_dropout_placeholder = tf.placeholder(tf.float32)
        self.embedding_dropout_placeholder = tf.placeholder(tf.float32)

        self.data_placeholder = tf.placeholder(
            tf.int32, shape=[None, 500], name='data_placeholder')
        self.sizes_placeholder = tf.placeholder(
            tf.int32, shape=[None], name='sizes_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None], name='labels_placeholder')

    def create_feed_dict(self, recurrent_output_dropout=None, recurrent_state_dropout=None,
                         embedding_dropout=None, data_placeholder=None, sizes_placeholder=None,
                         labels_placeholder=None):
        if recurrent_output_dropout is None:
            recurrent_output_dropout = self.config.recurrent_output_dropout

        if recurrent_state_dropout is None:
            recurrent_state_dropout = self.config.recurrent_state_dropout

        if embedding_dropout is None:
            embedding_dropout = self.config.embedding_dropout

        feed_dict = {self.recurrent_output_dropout_placeholder: recurrent_output_dropout,
                     self.recurrent_state_dropout_placeholder: recurrent_state_dropout,
                     self.embedding_dropout_placeholder: embedding_dropout}

        if data_placeholder is not None:
            feed_dict[self.data_placeholder] = data_placeholder

        if sizes_placeholder is not None:
            feed_dict[self.sizes_placeholder] = sizes_placeholder

        if labels_placeholder is not None:
            feed_dict[self.labels_placeholder] = labels_placeholder

        return feed_dict

    def add_embedding(self, inputs):
        """
        Adds and embedding layer that map the sentences id list to word vectors.
        """

        with tf.name_scope('embeddings'):
            vocab_size = len(self.pretrained_embeddings)
            base_embeddings = tf.get_variable(
                'embeddings',
                shape=(vocab_size, self.config.embed_size),
                initializer=tf.constant_initializer(self.pretrained_embeddings),
                dtype=tf.float32)
            embeddings_dropout = tf.nn.dropout(base_embeddings, self.embedding_dropout_placeholder)
            inputs = tf.nn.embedding_lookup(embeddings_dropout, inputs)

        return inputs

    def get_logits(self, inputs, size, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            num_units = self.config.num_units
            num_classes = self.config.num_classes

            x_hat = self.add_embedding(inputs)

            with tf.name_scope('recurrent_layer'):
                cell = tf.nn.rnn_cell.LSTMCell(num_units)

                drop_recurrent_cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell,
                        output_keep_prob=self.recurrent_output_dropout_placeholder,
                        state_keep_prob=self.recurrent_state_dropout_placeholder,
                        variational_recurrent=True,
                        input_size=self.config.embed_size,
                        dtype=tf.float32)

                """
                The dynamic_rnn outputs a 0 for every output after it has reached
                the end of a sentence. When doing sequence classification,
                we need the last relevant value from the outputs matrix.
                """
                _, state = tf.nn.dynamic_rnn(drop_recurrent_cell, x_hat,
                                             sequence_length=size,
                                             dtype=tf.float32)

                """
                This variable will have shape [batch_size, num_units]
                """
                recurrent_output = state.h

            with tf.name_scope('output_layer'):
                initializer = tf.contrib.layers.xavier_initializer()
                weight = tf.get_variable(
                    'output_weights',
                    shape=[num_units, num_classes],
                    initializer=initializer,
                    dtype=tf.float32)
                bias = tf.get_variable(
                    'output_bias',
                    shape=[num_classes],
                    initializer=initializer,
                    dtype=tf.float32)

                prediction = tf.matmul(recurrent_output, weight) + bias

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

        return loss

    def add_training_op(self, loss):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))

            if self.config.clip_gradients:
                gradients, _ = tf.clip_by_global_norm(
                    gradients, clip_norm=self.config.max_norm)

            train_op = optimizer.apply_gradients(zip(gradients, variables))

            return train_op

    def add_evaluation_op(self, logits, labels):
        with tf.name_scope('validation'):
            predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            size = tf.cast(tf.shape(predictions)[0], tf.float32)
            correct_pred = tf.equal(predictions, tf.cast(labels, tf.int32))

            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * size

            return accuracy, size

    def batch_evaluate(self, sess, accuracy, size):
        feed = self.create_feed_dict(recurrent_output_dropout=1.0, recurrent_state_dropout=1.0,
                                     embedding_dropout=1.0)
        accuracy, total = sess.run([accuracy, size], feed_dict=feed)

        return accuracy, total
