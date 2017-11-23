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

        base_embeddings = tf.Variable(self.pretrained_embeddings)
        inputs = tf.nn.embedding_lookup(base_embeddings, self.batch_data)

        return inputs
