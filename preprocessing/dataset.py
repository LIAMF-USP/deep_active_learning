import os

from preprocessing.format_dataset import (remove_html_from_text, remove_url_from_text,
                                          remove_special_characters_from_text, to_lower,
                                          create_unique_apostrophe, add_space_between_characters,
                                          SentenceTFRecord, sentence_to_id_list, get_vocab)
from word_embedding.word_embedding import get_embedding
from utils.progress_bar import Progbar
from utils.pickle import load, save


class Dataset:

    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        self.train_save_path = train_save_path
        self.validation_save_path = validation_save_path
        self.test_save_path = test_save_path

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.embedding_file = embedding_file
        self.embed_size = embed_size
        self.embedding_path = embedding_path
        self.embedding_wordindex_path = embedding_wordindex_path

        self.sentence_size = sentence_size

        self.train = None
        self.validation = None
        self.test = None
        self.format_data = False

        self.save_formatted_data = True
        self.should_load_datasets = True
        self.create_validation_dataset = True

    def preprocess_text_data(self, text_data):
        formatted_text = remove_html_from_text(text_data)
        formatted_text = remove_url_from_text(formatted_text)
        formatted_text = create_unique_apostrophe(formatted_text)
        formatted_text = add_space_between_characters(formatted_text)
        formatted_text = remove_special_characters_from_text(formatted_text)
        return to_lower(formatted_text)

    def load_datasets(self):
        if self.train_save_path and os.path.exists(self.train_save_path):
            print('Loading formatted train data ...')
            self.train = load(self.train_save_path)

        if self.validation_save_path and os.path.exists(self.validation_save_path):
            print('Loading formatted validation data ...')
            self.validation = load(self.validation_save_path)

        if self.test_save_path and os.path.exists(self.test_save_path):
            print('Loading formatted test data ...')
            self.test = load(self.test_save_path)

    def prepare_data(self, data, embedding, word_index, review_type):
        print('Find and replacing unknown words for {} data...'.format(review_type))
        progbar = Progbar(target=len(data))
        data = embedding.handle_unknown_words(
            data, sentence_size=self.sentence_size, progbar=progbar)

        print('Transforming {} data into list of ids'.format(review_type))
        data = self.transform_sentences(data, word_index)

        print('Saving {} formatted data using pickle'.format(review_type))
        output_path = os.path.join(self.output_dir, review_type, '{}.pkl'.format(review_type))
        save(data, output_path)

        print('Transforming {} data into tfrecords'.format(review_type))
        self.create_tfrecords(data, review_type)

        print()

    def create_train_dataset(self):
        raise NotImplementedError

    def create_test_dataset(self):
        raise NotImplementedError

    def get_data(self):
        if self.should_load_datasets:
            self.load_datasets()

        if not self.train:
            print('Creating train dataset ...')
            self.train = self.create_train_dataset()

            if not self.validation:
                print('Creating validation data')
                if self.create_validation_dataset:
                    self.train, self.validation = self.split_data(self.train)

            if self.save_formatted_data:
                print('Saving train data ...')
                save(self.train, self.train_save_path)

                if self.create_validation_dataset and self.validation:
                    print('Saving validation data ...')
                    save(self.validation, self.validation_save_path)

        if not self.test:
            print('Creating test data ...')
            self.test = self.create_test_dataset()

            if self.save_formatted_data:
                print('Saving test data ...')
                save(self.test, self.test_save_path)

    def create_dataset(self):
        if not self.train and not self.validation and not self.test:
            self.format_data = True

        self.make_dirs()
        self.get_data()

        vocab = self.get_vocabulary()

        embedding = self.load_embeddings(vocab)
        word_index, matrix, embedding_vocab = embedding.get_word_embedding()

        self.prepare_data(self.train, embedding, word_index, 'train')
        if self.create_validation_dataset:
            self.prepare_data(self.validation, embedding, word_index, 'val')
        self.prepare_data(self.test, embedding, word_index, 'test')

    def create_tfrecords(self, data, dataset_type):
        output_path = os.path.join(self.output_dir, dataset_type,
                                   '{}.tfrecord'.format(dataset_type))
        progbar = Progbar(target=len(data))

        sentence_tfrecord = SentenceTFRecord(data, output_path, progbar)
        sentence_tfrecord.parse_sentences()

    def transform_sentences(self, data, word_index):
        transformed_sentences = []
        progbar = Progbar(target=len(data))

        for index, (review, label) in enumerate(data):
            review_id_list = sentence_to_id_list(review, word_index)
            size = len(review_id_list)

            transformed_sentences.append((review_id_list, label, size))
            progbar.update(index + 1, [])

        return transformed_sentences

    def load_embeddings(self, vocab):
        return get_embedding(
            self.embedding_file,
            self.embed_size,
            vocab,
            self.embedding_path,
            self.embedding_wordindex_path)

    def get_vocabulary(self):
        print('Loading vocabulary...')
        vocab = get_vocab(self.train)
        return vocab

    def split_data(self, train, percent=0.1):
        num_data = int(len(train) * percent)

        validation = train[0:num_data]
        train = train[num_data:]

        return train, validation

    def make_dirs(self):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        train_path = os.path.join(self.output_dir, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        validation_path = os.path.join(self.output_dir, 'val')
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        test_path = os.path.join(self.output_dir, 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
