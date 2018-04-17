import os
import random

from preprocessing.dataset import Dataset


def get_subj_dataset(dataset_type):
    if dataset_type == 'active_learning':
        return ActiveLearningSubjDataset

    return SubjectiveDataset


class SubjectiveDataset(Dataset):

    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        self.labels = ['subj', 'obj']
        self.subjective_file = 'quote.tok.gt9.5000'
        self.objective_file = 'plot.tok.gt9.5000'

        super().__init__(train_save_path, validation_save_path, test_save_path,
                         data_dir, output_dir, embedding_file, embed_size,
                         embedding_path, embedding_wordindex_path, sentence_size)

        self.create_datasets()

    def create_datasets(self):
        subjective_data = self.parse_files(self.subjective_file, 0)
        objective_data = self.parse_files(self.objective_file, 1)

        train_subjective, test_subjective = self.split_data(subjective_data)
        train_objetive, test_objective = self.split_data(objective_data)

        self.train_dataset = train_subjective + train_objetive
        random.shuffle(self.train_dataset)

        self.test_dataset = test_subjective + test_objective
        random.shuffle(self.test_dataset)

    def create_train_dataset(self):
        return self.train_dataset

    def create_test_dataset(self):
        return self.test_dataset

    def parse_files(self, data_type, data_label):
        data_path = os.path.join(self.data_dir, data_type)
        text_data = self.load_text_data(data_path, data_label)

        return text_data

    def load_text_data(self, data_path, data_label):
        text_data = []

        with open(data_path, 'r', encoding='latin1') as data_file:

            for line in data_file.readlines():
                sentence = self.preprocess_text_data(line)
                text_data.append((sentence, data_label))

        return text_data


class ActiveLearningSubjDataset(SubjectiveDataset):
    def __init__(self, train_save_path, validation_save_path, test_save_path,
                 data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                 embedding_path, embedding_wordindex_path, sentence_size=None):
        super().__init__(train_save_path, validation_save_path, test_save_path,
                         data_dir, data_output_dir, output_dir, embedding_file, embed_size,
                         embedding_path, embedding_wordindex_path, sentence_size)
        self.save_formatted_reviews = False
        self.should_load_datasets = False
        self.create_validation_dataset = False
