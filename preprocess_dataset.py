import argparse
import random
import os

from preprocessing.format_dataset import (remove_html_from_text,
                                          remove_url_from_text,
                                          remove_special_characters_from_text,
                                          create_unique_apostrophe,
                                          add_space_between_characters,
                                          sentence_to_id_list,
                                          SentenceTFRecord,
                                          get_vocab,
                                          to_lower)
from word_embedding.word_embedding import get_embedding
from utils.progress_bar import Progbar


POS_LABEL = 0
NEG_LABEL = 1


def preprocess_review_text(review_text):
    formatted_text = remove_html_from_text(review_text)
    formatted_text = remove_url_from_text(formatted_text)
    formatted_text = create_unique_apostrophe(formatted_text)
    formatted_text = add_space_between_characters(formatted_text)
    formatted_text = remove_special_characters_from_text(formatted_text)
    return to_lower(formatted_text)


def preprocess_review_files(dataset_path, output_dataset_path, review_type):
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

        formatted_text = preprocess_review_text(review_text)
        formatted_review_texts.append(formatted_text)

        output_review = os.path.join(output_sentiment_type, review)
        with open(output_review, 'w') as review_file:
            review_file.write(formatted_text)

        progbar.update(index + 1, [])
    print()

    return formatted_review_texts


def preprocess_files(dataset_path, output_dataset_path):
    pos_reviews = preprocess_review_files(dataset_path, output_dataset_path, 'pos')
    neg_reviews = preprocess_review_files(dataset_path, output_dataset_path, 'neg')

    return pos_reviews, neg_reviews


def make_output_dir(output_dataset_path):
    os.makedirs(output_dataset_path)

    output_pos = os.path.join(output_dataset_path, 'pos')
    os.makedirs(output_pos)

    output_neg = os.path.join(output_dataset_path, 'neg')
    os.makedirs(output_neg)


def apply_data_preprocessing(user_args):
    data_dir = user_args['data_dir']
    dataset_type = user_args['dataset_type']
    output_dir = user_args['output_dir']

    output_dataset_path = os.path.join(output_dir, dataset_type)
    dataset_path = os.path.join(data_dir, dataset_type)

    if not os.path.exists(output_dataset_path):
        make_output_dir(output_dataset_path)

    return preprocess_files(dataset_path, output_dataset_path)


def add_label_to_dataset(dataset, label):
    return [(data, label) for data in dataset]


def create_unified_dataset(pos_reviews, neg_reviews):
    pos_reviews = add_label_to_dataset(pos_reviews, POS_LABEL)
    neg_reviews = add_label_to_dataset(neg_reviews, NEG_LABEL)

    all_reviews = pos_reviews + neg_reviews

    random.shuffle(all_reviews)

    return all_reviews


def transform_sentences(movie_reviews, sentence_size, word_index):
    transformed_sentences = []
    progbar = Progbar(target=len(movie_reviews))

    for index, (review, label) in enumerate(movie_reviews):
        review_id_list = sentence_to_id_list(review, word_index)
        size = len(review_id_list)

        transformed_sentences.append((review_id_list, label, size))
        progbar.update(index + 1, [])

    return transformed_sentences


def load_embeddings(user_args, vocab):
    embedding_file = user_args['embedding_file']
    embed_size = user_args['embed_size']
    embedding_path = user_args['embedding_path']
    embedding_wordindex_path = user_args['embedding_wordindex_path']

    return get_embedding(embedding_file, embed_size, vocab, embedding_path,
                         embedding_wordindex_path)


def create_tfrecords(reviews, output_path, dataset_type):
    output_path = os.path.join(output_path, dataset_type,
                               '{}.tfrecord'.format(dataset_type))
    progbar = Progbar(target=len(reviews))

    sentence_tfrecord = SentenceTFRecord(reviews, output_path, progbar)
    sentence_tfrecord.parse_sentences()


def create_validation_set(all_reviews, percent=0.1):
    num_reviews = int(len(all_reviews) * percent)

    validation_reviews = all_reviews[0:num_reviews]
    all_reviews = all_reviews[num_reviews:]

    return all_reviews, validation_reviews


def create_validation_dir(user_args):
    output_dir = user_args['output_dir']
    validation_dir = 'val'

    validation_path = os.path.join(output_dir, validation_dir)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data-dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')

    parser.add_argument('-dt',
                        '--dataset-type',
                        type=str,
                        help='The dataset that should be formatted: train or test')

    parser.add_argument('-ef',
                        '--embedding-file',
                        type=str,
                        help='The location of the embedding file')

    parser.add_argument('-ep',
                        '--embedding-path',
                        type=str,
                        help='Location of the embedding file (Testing Dataset Only)')

    parser.add_argument('-ewi',
                        '--embedding-wordindex-path',
                        type=str,
                        help='Location of the embedding word index file (Testing Dataset Only)')

    parser.add_argument('-es',
                        '--embed-size',
                        type=int,
                        help='The embedding size of the embedding file')

    parser.add_argument('-s',
                        '--sentence-size',
                        type=int,
                        help=('The sentence size that will be used in the model.' +
                              'If a sentence in our dataset is larger than this variable' +
                              'It will be cropped to this size. Otherwise, it will be padded' +
                              'with an special character'))

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        help='The path of the new formatted dataset')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    dataset_type = user_args['dataset_type']
    output_dir = user_args['output_dir']
    is_test = False if dataset_type == 'train' else True

    """
    This step is responsible for parsing the review texts, such as removing
    HTML tags, or separating string such as he's into he and 's.
    """
    pos_reviews, neg_reviews = apply_data_preprocessing(user_args)

    """
    This step will join both pos_reviews and neg_reviews into a single list
    and add a label to each review sentence. Finally, these reviews will be
    shuffled.
    """
    all_reviews = create_unified_dataset(pos_reviews, neg_reviews)

    vocab = None
    if not is_test:
        """
        If we are preprocessing the training data, we should also
        create the validation set for hyperparamenter tuning.
        """
        all_reviews, validation_reviews = create_validation_set(all_reviews)
        print('Creating validation set')
        create_validation_dir(user_args)

        print('Loading vocabulary...')
        vocab = get_vocab(all_reviews)

    """
    Load Embeddings.
    """
    embedding = load_embeddings(user_args, vocab)
    word_index, matrix, embedding_vocab = embedding.get_word_embedding()

    """
    Handle unknown words in the embedding
    """
    sentence_size = user_args['sentence_size']
    print('Find and replacing unknown words for reviews...')
    progbar = Progbar(target=len(all_reviews))
    all_reviews = embedding.handle_unknown_words(all_reviews, sentence_size=None, progbar=progbar)
    print()

    if not is_test:
        print('Find and replacing unknown words for validation reviews...')
        progbar = Progbar(target=len(validation_reviews))
        validation_reviews = embedding.handle_unknown_words(validation_reviews, sentence_size=None,
                                                            progbar=progbar)
        print()

    """
    The reviews are turned into a list of ids.
    """

    print('Transforming {} reviews into list of ids'.format(dataset_type))
    all_reviews = transform_sentences(all_reviews, sentence_size, word_index)
    print()

    if not is_test:
        print('Transforming validation reviews into list of ids')
        validation_reviews = transform_sentences(validation_reviews, sentence_size,
                                                 word_index)
        print()

    """
    Create the TFRecords file for our reviews.
    """
    print('Transforming {} reviews into tfrecords'.format(dataset_type))
    create_tfrecords(all_reviews, output_dir, dataset_type)
    print()

    if not is_test:
        print('Transforming validation reviews into tfrecords')
        create_tfrecords(validation_reviews, output_dir, 'val')
        print()


if __name__ == '__main__':
    main()
