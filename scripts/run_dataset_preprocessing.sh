#!/bin/bash

set -e

#usage
#./script/run_dataset_preprocessing.sh

DATA_DIR="data/aclImdb"
DATASET_TRAIN="train"
DATASET_TEST="test"

TRAIN_SAVE_PATH="data/aclImdb_formatted/train.pkl"
VALIDATION_SAVE_PATH="data/aclImdb_formatted/validation.pkl"
TEST_SAVE_PATH="data/aclImdb_formatted/test.pkl"

GLOVE_FILE="data/glove/glove.6B.100d.txt"

EMBEDDING_PATH="data/glove/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/glove/glove_word_index.pkl"

SENTENCE_SIZE=1000
GLOVE_OUTPUT_DIR="data/glove/aclImdb_formatted"
EMBED_SIZE=100

echo "Preprocessing GloVe training data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TRAIN} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${GLOVE_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${GLOVE_OUTPUT_DIR}

echo -e "\n\nPreprocessing GloVe test data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TEST} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${GLOVE_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${GLOVE_OUTPUT_DIR}

FASTTEXT_FILE="data/fasttext/wiki.en.bin"
EMBEDDING_PATH="data/fasttext/fasttext.pkl"
EMBEDDING_WORDINDEX_PATH="data/fasttext/fasttext_word_index.pkl"
FASTTEXT_OUTPUT_DIR="data/fasttext/aclImdb_formatted"
EMBED_SIZE=300

echo "Preprocessing FastText training data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TRAIN} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${FASTTEXT_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${FASTTEXT_OUTPUT_DIR}

echo -e "\n\nPreprocessing FastText test data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TEST} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${FASTTEXT_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${FASTTEXT_OUTPUT_DIR}

WORD2VEC_FILE="data/word2vec/GoogleNews-vectors-negative300.bin"
EMBEDDING_PATH="data/word2vec/word2vec.pkl"
EMBEDDING_WORDINDEX_PATH="data/word2vec/word2vec_word_index.pkl"
WORD2VEC_OUTPUT_DIR="data/word2vec/aclImdb_formatted"
EMBED_SIZE=300

echo "Preprocessing Word2Vec training data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TRAIN} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${WORD2VEC_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${WORD2VEC_OUTPUT_DIR}

echo -e "\n\nPreprocessing Word2Vec test data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TEST} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${WORD2VEC_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${WORD2VEC_OUTPUT_DIR}
