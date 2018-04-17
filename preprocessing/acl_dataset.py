import os
import random

from preprocessing.dataset import Dataset
from utils.progress_bar import Progbar


POS_LABEL = 0
NEG_LABEL = 1


def get_acl_dataset(dataset_type):

    if dataset_type == 'debug':
        return DebugMovieReviewDataset
    elif dataset_type == 'active_learning':
        return ActiveLearningDataset

    return ACLMovieReviewDataset


class ACLMovieReviewDataset(Dataset):

    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        self.labels = ['pos', 'neg']
        self.data_output_dir = data_output_dir

        super().__init__(train_save_path, validation_save_path, test_save_path,
                         data_dir, output_dir, embedding_file, embed_size,
                         embedding_path, embedding_wordindex_path, sentence_size)

    def create_train_dataset(self):
        pos_train_reviews, neg_train_reviews = self.apply_data_preprocessing('train')
        return self.create_unified_dataset(pos_train_reviews, neg_train_reviews)

    def create_test_dataset(self):
        pos_test_reviews, neg_test_reviews = self.apply_data_preprocessing('test')
        return self.create_unified_dataset(pos_test_reviews, neg_test_reviews)

    def add_label_to_dataset(self, dataset, label):
        return [(data, label) for data in dataset]

    def apply_data_preprocessing(self, dataset_type):
        dataset_path = os.path.join(self.data_dir, dataset_type)
        output_dataset_path = os.path.join(self.data_output_dir, dataset_type)

        return self.preprocess_files(dataset_path, output_dataset_path)

    def create_unified_dataset(self, pos_reviews, neg_reviews):
        pos_reviews = self.add_label_to_dataset(pos_reviews, POS_LABEL)
        neg_reviews = self.add_label_to_dataset(neg_reviews, NEG_LABEL)

        all_reviews = pos_reviews + neg_reviews

        random.shuffle(all_reviews)

        return all_reviews

    def preprocess_files(self, dataset_path, output_dataset_path):
        pos_reviews = self.preprocess_review_files(dataset_path, output_dataset_path, 'pos')
        neg_reviews = self.preprocess_review_files(dataset_path, output_dataset_path, 'neg')

        return pos_reviews, neg_reviews

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

            formatted_text = self.preprocess_text_data(review_text)
            formatted_review_texts.append(formatted_text)

            output_review = os.path.join(output_sentiment_type, review)
            with open(output_review, 'w') as review_file:
                review_file.write(formatted_text)

            progbar.update(index + 1, [])
        print()

        return formatted_review_texts

    def make_output_dirs(self, dataset_type):
        output_dataset_path = os.path.join(self.data_output_dir, dataset_type)
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)

        for label in self.labels:
            output_path = os.path.join(output_dataset_path, label)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

    def make_dirs(self):
        if self.format_data:
            self.make_output_dirs('train')
            self.make_output_dirs('test')

        super().make_dirs()


class DebugMovieReviewDataset(ACLMovieReviewDataset):

    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        super().__init__(train_save_path, validation_save_path, test_save_path, data_dir,
                         data_output_dir, output_dir, embedding_file, embed_size,
                         embedding_path, embedding_wordindex_path, sentence_size)
        self.save_formatted_reviews = False

    def get_reviews(self):
        print('Creating DEBUG dataset')
        super().get_reviews()

        self.train_reviews = self.train_reviews[0:500]
        self.validation_reviews = self.validation_reviews[0:100]


class ActiveLearningDataset(ACLMovieReviewDataset):
    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        super().__init__(train_save_path, validation_save_path, test_save_path, data_dir,
                         data_output_dir, output_dir, embedding_file, embed_size,
                         embedding_path, embedding_wordindex_path, sentence_size)
        self.save_formatted_reviews = False
        self.should_load_datasets = False
        self.create_validation_dataset = False
