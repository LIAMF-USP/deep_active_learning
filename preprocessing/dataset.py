import os
import pickle
import random

from preprocessing.format_dataset import (remove_html_from_text, remove_url_from_text,
                                          remove_special_characters_from_text, to_lower,
                                          create_unique_apostrophe, add_space_between_characters,
                                          sentence_to_id_list, SentenceTFRecord, get_vocab)
from word_embedding.word_embedding import get_embedding
from utils.progress_bar import Progbar


POS_LABEL = 0
NEG_LABEL = 1


def load(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)


def save(save_data, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(save_data, f)


class MovieReviewDataset:

    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        self.train_save_path = train_save_path
        self.validation_save_path = validation_save_path
        self.test_save_path = test_save_path

        self.data_dir = data_dir
        self.data_output_dir = data_output_dir
        self.output_dir = output_dir

        self.embedding_file = embedding_file
        self.embed_size = embed_size
        self.embedding_path = embedding_path
        self.embedding_wordindex_path = embedding_wordindex_path

        self.sentence_size = sentence_size

        self.train_reviews = None
        self.validation_reviews = None
        self.test_reviews = None
        self.format_reviews = False

        self.load_datasets()

    def load_datasets(self):
        if self.train_save_path and os.path.exists(self.train_save_path):
            print('Loading formatted train reviews ...')
            self.train_reviews = load(self.train_save_path)

        if self.validation_save_path and os.path.exists(self.validation_save_path):
            print('Loading formatted validation reviews ...')
            self.validation_reviews = load(self.validation_save_path)

        if self.test_save_path and os.path.exists(self.test_save_path):
            print('Loading formatted test reviews ...')
            self.test_reviews = load(self.test_save_path)

    def prepare_reviews(self, reviews, embedding, word_index, review_type):
        print('Find and replacing unknown words for {} reviews...'.format(review_type))
        progbar = Progbar(target=len(reviews))
        reviews = embedding.handle_unknown_words(
            reviews, sentence_size=self.sentence_size, progbar=progbar)

        print('Transforming {} reviews into list of ids'.format(review_type))
        reviews = self.transform_sentences(reviews, word_index)

        print('Transforming {} reviews into tfrecords'.format(review_type))
        self.create_tfrecords(reviews, review_type)

    def create_dataset(self):
        if not self.train_reviews and not self.validation_reviews and not self.test_reviews:
            self.format_reviews = True

        self.make_dirs()

        if not self.train_reviews:
            pos_train_reviews, neg_train_reviews = self.apply_data_preprocessing('train')
            self.train_reviews = self.create_unified_dataset(pos_train_reviews, neg_train_reviews)

            if not self.validation_reviews:
                print('Creating validation set')
                self.train_reviews, self.validation_reviews = self.create_validation_set(
                        self.train_reviews)

                print('Saving train reviews ...')
                save(self.train_reviews, self.train_save_path)
                print('Saving validation reviews ...')
                save(self.validation_reviews, self.validation_save_path)

        if not self.test_reviews:
            pos_test_reviews, neg_test_reviews = self.apply_data_preprocessing('test')
            self.test_reviews = self.create_unified_dataset(pos_test_reviews, neg_test_reviews)

            print('Saving test reviews ...')
            save(self.test_reviews, self.test_save_path)

        vocab = self.get_vocabulary(self.train_reviews)

        embedding = self.load_embeddings(vocab)
        word_index, matrix, embedding_vocab = embedding.get_word_embedding()

        self.prepare_reviews(self.train_reviews, embedding, word_index, 'train')
        print()

        self.prepare_reviews(self.validation_reviews, embedding, word_index, 'val')
        print()

        self.prepare_reviews(self.test_reviews, embedding, word_index, 'test')
        print()

    def create_tfrecords(self, reviews, dataset_type):
        output_path = os.path.join(self.output_dir, dataset_type,
                                   '{}.tfrecord'.format(dataset_type))
        progbar = Progbar(target=len(reviews))

        sentence_tfrecord = SentenceTFRecord(reviews, output_path, progbar)
        sentence_tfrecord.parse_sentences()

    def transform_sentences(self, reviews, word_index):
        transformed_sentences = []
        progbar = Progbar(target=len(reviews))

        for index, (review, label) in enumerate(reviews):
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

    def get_vocabulary(self, train_reviews):
        vocab = None

        print('Loading vocabulary...')
        vocab = get_vocab(train_reviews)
        return vocab

    def create_validation_set(self, train_reviews, percent=0.1):
        num_reviews = int(len(train_reviews) * percent)

        validation_reviews = train_reviews[0:num_reviews]
        train_reviews = train_reviews[num_reviews:]

        return train_reviews, validation_reviews

    def add_label_to_dataset(self, dataset, label):
        return [(data, label) for data in dataset]

    def create_unified_dataset(self, pos_reviews, neg_reviews):
        pos_reviews = self.add_label_to_dataset(pos_reviews, POS_LABEL)
        neg_reviews = self.add_label_to_dataset(neg_reviews, NEG_LABEL)

        all_reviews = pos_reviews + neg_reviews

        random.shuffle(all_reviews)

        return all_reviews

    def preprocess_review_text(self, review_text):
        formatted_text = remove_html_from_text(review_text)
        formatted_text = remove_url_from_text(formatted_text)
        formatted_text = create_unique_apostrophe(formatted_text)
        formatted_text = add_space_between_characters(formatted_text)
        formatted_text = remove_special_characters_from_text(formatted_text)
        return to_lower(formatted_text)

    def preprocess_review_files(self, dataset_path, output_dataset_path, review_type):
        dataset_sentiment_type = os.path.join(dataset_path, review_type)
        output_sentiment_type = os.path.join(output_dataset_path, review_type)

        review_files = os.listdir(dataset_sentiment_type)
        num_review_files = len(review_files)
        formatted_review_texts = []

        print('Formatting {} texts'.format(review_type))
        progbar = Progbar(target=num_review_files)

        for index, review in enumerate(review_files):
            original_review = os.path.join(dataset_sentiment_type, review)
            with open(original_review, 'r') as review_file:
                review_text = review_file.read()

            formatted_text = self.preprocess_review_text(review_text)
            formatted_review_texts.append(formatted_text)

            output_review = os.path.join(output_sentiment_type, review)
            with open(output_review, 'w') as review_file:
                review_file.write(formatted_text)

            progbar.update(index + 1, [])
        print()

        return formatted_review_texts

    def preprocess_files(self, dataset_path, output_dataset_path):
        pos_reviews = self.preprocess_review_files(dataset_path, output_dataset_path, 'pos')
        neg_reviews = self.preprocess_review_files(dataset_path, output_dataset_path, 'neg')

        return pos_reviews, neg_reviews

    def make_output_dirs(self, dataset_type):
        output_dataset_path = os.path.join(self.data_output_dir, dataset_type)
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)

        output_pos = os.path.join(output_dataset_path, 'pos')
        if not os.path.exists(output_pos):
            os.makedirs(output_pos)

        output_neg = os.path.join(output_dataset_path, 'neg')
        if not os.path.exists(output_neg):
            os.makedirs(output_neg)

    def apply_data_preprocessing(self, dataset_type):
        dataset_path = os.path.join(self.data_dir, dataset_type)
        output_dataset_path = os.path.join(self.data_output_dir, dataset_type)

        return self.preprocess_files(dataset_path, output_dataset_path)

    def make_dirs(self):

        if self.format_reviews:
            self.make_output_dirs('train')
            self.make_output_dirs('test')

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
