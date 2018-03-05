import numpy as np
from utils.progress_bar import Progbar


class MonteCarloEvaluation:

    def __init__(self, sess, model, verbose):
        self.sess = sess
        self.model = model
        self.verbose = verbose

    def preprocess_batch(self, data_batch):
        preprocessed_batch = []
        max_len = 500

        for data in data_batch:
            if len(data) < max_len:
                data += [0] * (max_len - len(data))
            elif len(data) > max_len:
                data = data[:max_len]

            preprocessed_batch.append(data)

        return preprocessed_batch

    def prediction_samples(self, data_batch, sizes_batch, num_samples, preprocess_batch=True):
        predictions = np.zeros(shape=(data_batch.shape[0], num_samples))

        if preprocess_batch:
            data_batch = np.array(self.preprocess_batch(data_batch))

        feed_dict = self.model.create_feed_dict(
            data_placeholder=data_batch,
            sizes_placeholder=sizes_batch)

        for i in range(num_samples):
            if self.verbose:
                progbar = Progbar(target=num_samples)

                prediction = self.sess.run(
                    self.model.predictions,
                    feed_dict=feed_dict)

                predictions[:, i] = prediction

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

    def monte_carlo_dropout_evaluate(self, data_batch, sizes_batch, labels_batch, num_data):
        all_preds = self.monte_carlo_samples(data_batch, sizes_batch, num_samples=10)
        mc_counts = self.monte_carlo_samples_count(all_preds)

        predictions = np.zeros(shape=(num_data))
        for index, (bincount, value) in enumerate(mc_counts):
            predictions[index] = value

        correct_pred = np.equal(predictions, labels_batch)

        return np.mean(correct_pred)
