import numpy as np

from utils.metrics import variation_ratio, entropy, bald
from utils.progress_bar import Progbar


def get_monte_carlo_metric(metric):
    if metric == 'variation_ratio':
        return VariationRationMC
    elif metric == 'entropy':
        return EntropyMC
    elif metric == 'bald':
        return BaldMC
    elif metric == 'random':
        return Random
    elif metric == 'softmax':
        return Softmax


class MonteCarloEvaluation:

    def __init__(self, sess, model, data_batch, sizes_batch, labels_batch,
                 num_classes, num_samples, max_len, verbose):
        self.sess = sess
        self.model = model

        self.data_batch = data_batch
        self.sizes_batch = sizes_batch
        self.labels_batch = labels_batch
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.max_len = max_len

        self.verbose = verbose

    def preprocess_batch(self, data_batch, sizes_batch):
        preprocessed_batch = []
        preprocessed_sizes_batch = []

        for data, size in zip(data_batch, sizes_batch):
            if len(data) < self.max_len:
                data += [0] * (self.max_len - len(data))

            elif len(data) > self.max_len:
                data = data[:self.max_len]
                size = self.max_len

            preprocessed_batch.append(data)
            preprocessed_sizes_batch.append(size)

        return np.array(preprocessed_batch), np.array(preprocessed_sizes_batch)

    def initialize_predictions(self):
        raise NotImplementedError

    def update_predictions(self, predictions, index):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def create_feed_dict(self, data_batch, sizes_batch):
        self.feed_dict = self.model.create_feed_dict(
            data_placeholder=data_batch,
            sizes_placeholder=sizes_batch)

    def prediction_samples(self, preprocess_batch=True):
        predictions = self.initialize_predictions()

        if preprocess_batch:
            data_batch, sizes_batch = self.preprocess_batch(
                self.data_batch, self.sizes_batch)

        self.create_feed_dict(data_batch, sizes_batch)

        for i in range(self.num_samples):
            if self.verbose:
                progbar = Progbar(target=self.num_samples)

            self.update_predictions(predictions, i)

            if self.verbose:
                progbar.update(i + 1, [])

        return predictions

    def monte_carlo_samples_count(self, all_preds):
        mc_counts = []

        all_preds = all_preds.astype(dtype=np.int64)

        for row in all_preds:
            bincount = np.bincount(row)
            mc_counts.append((bincount, bincount.argmax()))

        return mc_counts

    def monte_carlo_dropout_evaluate(self, num_data):
        all_preds = self.monte_carlo_samples()
        mc_counts = self.monte_carlo_samples_count(all_preds)

        predictions = np.zeros(shape=(num_data))
        for index, (bincount, value) in enumerate(mc_counts):
            predictions[index] = value

        correct_pred = np.equal(predictions, self.labels_batch)

        return np.mean(correct_pred)


class VariationRationMC(MonteCarloEvaluation):

    def initialize_predictions(self):
        return np.zeros(shape=(self.data_batch.shape[0], self.num_samples))

    def update_predictions(self, predictions, index):
        prediction = self.sess.run(
            self.model.predictions,
            feed_dict=self.feed_dict)

        predictions[:, index] = prediction

    def evaluate(self):
        all_preds = self.prediction_samples()
        mc_counts = self.monte_carlo_samples_count(all_preds)
        variation_ratios = np.array(variation_ratio(mc_counts))

        return variation_ratios


class EntropyMC(MonteCarloEvaluation):

    def initialize_predictions(self):
        return np.zeros(shape=(self.data_batch.shape[0], self.num_classes))

    def update_predictions(self, predictions, index):
        prediction = self.sess.run(
            self.model.predictions_distribution,
            feed_dict=self.feed_dict)

        predictions += prediction

    def evaluate(self):
        all_preds = self.prediction_samples()
        entropy_values = entropy(all_preds, self.num_samples)

        return entropy_values


class BaldMC(MonteCarloEvaluation):

    def __init__(self, sess, model, data_batch, sizes_batch, labels_batch,
                 num_classes, num_samples, max_len, verbose):
        super().__init__(sess, model, data_batch, sizes_batch, labels_batch, num_classes,
                         num_samples, max_len, verbose)

        self.dropout_entropy = np.zeros(shape=(self.data_batch.shape[0]))

    def initialize_predictions(self):
        return np.zeros(shape=(self.data_batch.shape[0], self.num_classes))

    def update_predictions(self, predictions, index):
        prediction = self.sess.run(
            self.model.predictions_distribution,
            feed_dict=self.feed_dict)

        self.dropout_entropy += entropy(prediction, 1)
        predictions += prediction

    def evaluate(self):
        all_preds = self.prediction_samples()
        bald_values = bald(all_preds, self.dropout_entropy, self.num_samples)

        return bald_values


class Random(MonteCarloEvaluation):

    def __init__(self, sess, model, data_batch, sizes_batch, labels_batch,
                 num_classes, num_samples, max_len, verbose):
        super().__init__(sess, model, data_batch, sizes_batch, labels_batch, num_classes,
                         0, max_len, verbose)

    def create_feed_dict(self, data_batch, sizes_batch):
        recurrent_output_dropout = 1
        recurrent_state_dropout = 1
        embedding_dropout = 1

        self.feed_dict = self.model.create_feed_dict(
            recurrent_output_dropout=recurrent_output_dropout,
            recurrent_state_dropout=recurrent_state_dropout,
            embedding_dropout=embedding_dropout,
            data_placeholder=data_batch,
            sizes_placeholder=sizes_batch)

    def initialize_predictions(self):
        return None

    def evaluate(self):
        return np.random.uniform(size=(self.data_batch.shape[0]))


class Softmax(MonteCarloEvaluation):

    def __init__(self, sess, model, data_batch, sizes_batch, labels_batch,
                 num_classes, num_samples, max_len, verbose):
        super().__init__(sess, model, data_batch, sizes_batch, labels_batch, num_classes,
                         1, max_len, verbose)

    def create_feed_dict(self, data_batch, sizes_batch):
        recurrent_output_dropout = 1
        recurrent_state_dropout = 1
        embedding_dropout = 1

        self.feed_dict = self.model.create_feed_dict(
            recurrent_output_dropout=recurrent_output_dropout,
            recurrent_state_dropout=recurrent_state_dropout,
            embedding_dropout=embedding_dropout,
            data_placeholder=data_batch,
            sizes_placeholder=sizes_batch)

    def initialize_predictions(self):
        return np.zeros(shape=(self.data_batch.shape[0], self.num_classes))

    def update_predictions(self, predictions, index):
        prediction = self.sess.run(
            self.model.predictions_distribution,
            feed_dict=self.feed_dict)

        predictions += prediction

    def evaluate(self):
        all_preds = self.prediction_samples()

        return np.amax(all_preds, axis=1)
