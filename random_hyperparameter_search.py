import argparse
import random

import numpy as np
import tensorflow as tf

from collections import namedtuple
from pathlib import Path

from base_model import run_model

WordEmbedding = namedtuple(
    'WordEmbedding',
    ['name', 'train_file', 'validation_file', 'test_file',
     'embedding_file', 'embedding_pickle', 'embed_size']
)

GloVe = WordEmbedding(
    name='GloVe',
    train_file='data/glove/aclImdb_formatted/train/train.tfrecord',
    validation_file='data/glove/aclImdb_formatted/val/val.tfrecord',
    test_file='data/glove/aclImdb_formatted/test/test.tfrecord',
    embedding_file='data/glove/glove.6B.100d.txt',
    embedding_pickle='data/glove/glove.pkl',
    embed_size=100
)


FastText = WordEmbedding(
    name='FastText',
    train_file='data/fasttext/aclImdb_formatted/train/train.tfrecord',
    validation_file='data/fasttext/aclImdb_formatted/val/val.tfrecord',
    test_file='data/fasttext/aclImdb_formatted/test/test.tfrecord',
    embedding_file='data/fasttext/wiki.en.vec',
    embedding_pickle='data/fasttext/fasttext.pkl',
    embed_size=300
)


Word2Vec = WordEmbedding(
    name='Word2Vec',
    train_file='data/word2vec/aclImdb_formatted/train/train.tfrecord',
    validation_file='data/word2vec/aclImdb_formatted/val/val.tfrecord',
    test_file='data/word2vec/aclImdb_formatted/test/test.tfrecord',
    embedding_file='data/word2vec/GoogleNews-vectors-negative300.bin',
    embedding_pickle='data/word2vec/word2vec.pkl',
    embed_size=300
)

word_embeddings = {
    0: GloVe,
    1: FastText,
    2: Word2Vec
}

BATCH_SIZES = [32, 64, 128]
NUM_EPOCHS = [4, 8, 10, 12, 14, 16, 18, 20]


class RandomParameterSearch:

    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.num_train = 22500
        self.num_validation = 2500
        self.num_test = 25000

        self.tensorboard_dir = 'tensorboard_logs'
        self.graphs_dir = 'graphs'

        self.perform_shuffle = 1
        self.num_classes = 2
        self.clip_gradients = 0
        self.max_norm = 5

        self.bucket_width = 30
        self.num_buckets = 30
        self.use_test = 0
        self.save_graph = 0

    def get_embedding(self):
        return word_embeddings[random.randint(0, 2)]

    def get_batch_size(self):
        return BATCH_SIZES[random.randint(0, len(BATCH_SIZES) - 1)]

    def get_num_epochs(self):
        return NUM_EPOCHS[random.randint(0, len(NUM_EPOCHS) - 1)]

    def get_dropout(self):
        return np.random.uniform(0.2, 1)

    def get_learning_rate(self):
        return 10 ** np.random.uniform(-4, -0.6)

    def exponential_draw(self, min_value, max_value):
        return np.exp(np.random.uniform(np.log(min_value), np.log(max_value)))

    def geometric_draw(self, min_value, max_value):
        return np.round(self.exponential_draw(min_value, max_value))

    def get_num_units(self, min_value=128, max_value=1024):
        return int(self.geometric_draw(min_value, max_value))

    def get_weight_decay(self, min_value=3.1e-7, max_value=3.1e-5):
        return self.exponential_draw(min_value, max_value)

    def sample_parameters(self):
        chosen_embedding = self.get_embedding()

        self.embedding_name = chosen_embedding.name
        self.train_file = chosen_embedding.train_file
        self.validation_file = chosen_embedding.validation_file
        self.test_file = chosen_embedding.test_file
        self.embedding_file = chosen_embedding.embedding_file
        self.embedding_pickle = chosen_embedding.embedding_pickle
        self.embed_size = chosen_embedding.embed_size

        self.learning_rate = self.get_learning_rate()

        self.recurrent_output_dropout = self.get_dropout()
        self.recurrent_state_dropout = self.get_dropout()
        self.embedding_dropout = self.get_dropout()

        self.batch_size = self.get_batch_size()

        self.num_epochs = self.get_num_epochs()
        self.num_units = self.get_num_units()
        self.weight_decay = self.get_weight_decay()


        self.model_name = 'Embedding:{0},lr:{1:.5f},out_drop:{2:.3f},var_drop:{3:.3f},emb_drop:{4:.3f},batch:{5},epoch:{6},units:{7},decay:{8:.10f}'.format( # noqa
            self.embedding_name, self.learning_rate, self.recurrent_output_dropout,
            self.recurrent_state_dropout, self.embedding_dropout, self.batch_size,
            self.num_epochs, self.num_units, self.weight_decay
        )

    def save_parameters(self, best_model, best_accuracy, save_path):
        save_path = Path(save_path)
        parameters = best_model.split(',')

        if not save_path.exists():
            save_path.mkdir()

        best_model_path = save_path / 'best_model_parameters.txt'

        with open(best_model_path, 'w') as best_model_file:
            best_model_file.write(best_model + '\n')
            for parameter in parameters:
                best_model_file.write(parameter + '\n')

            best_model_file.write('accuracy: {}'.format(best_accuracy))

    def find_best_parameters(self, save_path, save_graph=False, verbose=True):
        best_accuracy = -1
        best_model = None

        for sample in range(self.num_samples):
            self.sample_parameters()

            if verbose:
                print('Evaluating model:\n{}'.format(self.model_name))
            try:
                accuracy = run_model(
                    train_file=self.train_file,
                    validation_file=self.validation_file,
                    test_file=self.test_file,
                    num_train=self.num_train,
                    num_validation=self.num_validation,
                    num_test=self.num_test,
                    graphs_dir=self.graphs_dir,
                    model_name=self.model_name,
                    tensorboard_dir=self.tensorboard_dir,
                    embedding_file=self.embedding_file,
                    embedding_pickle=self.embedding_pickle,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    perform_shuffle=self.perform_shuffle,
                    embed_size=self.embed_size,
                    num_units=self.num_units,
                    num_classes=self.num_classes,
                    recurrent_output_dropout=self.recurrent_output_dropout,
                    recurrent_state_dropout=self.recurrent_state_dropout,
                    embedding_dropout=self.embedding_dropout,
                    clip_gradients=self.clip_gradients,
                    max_norm=self.max_norm,
                    weight_decay=self.weight_decay,
                    bucket_width=self.bucket_width,
                    num_buckets=self.num_buckets,
                    use_test=self.use_test,
                    save_graph=self.save_graph)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = self.model_name

            except tf.errors.InvalidArgumentError:
                print('Error running model ...')
                continue
            except tf.errors.ResourceExhaustedError:
                print('Error running model ...')
                continue

        if verbose:
            print('Best model:\n{}'.format(best_model))
            print('Accuracy on validation set: {}'.format(best_accuracy))

        if verbose:
            print('Saving best model parameters ...')

        if best_model:
            self.save_parameters(best_model, best_accuracy, save_path)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ns',
                        '--num-samples',
                        type=int,
                        help='The number of sample to use')

    parser.add_argument('-sf',
                        '--save-folder',
                        type=str,
                        help='The folder to save the Random Parameter Search results')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    num_samples = user_args['num_samples']
    save_folder = user_args['save_folder']

    random_search = RandomParameterSearch(num_samples)
    random_search.find_best_parameters(save_folder)


if __name__ == '__main__':
    main()
