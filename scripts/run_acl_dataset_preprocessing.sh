#!/bin/bash

set -e

#usage
#./script/run_acl_dataset_preprocessing.sh

DATASET="acl"
DATA_DIR="data/aclImdb"
DATA_OUTPUT_DIR="data/aclImdb_formatted"

TRAIN_SAVE_PATH="data/aclImdb_formatted/train.pkl"
VALIDATION_SAVE_PATH="data/aclImdb_formatted/validation.pkl"
TEST_SAVE_PATH="data/aclImdb_formatted/test.pkl"

GLOVE_FILE="data/glove/glove.6B.100d.txt"

SENTENCE_SIZE=600
EMBED_SIZE=100

EMBEDDING_PATH="data/glove/acl/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/glove/acl/glove_word_index.pkl"
GLOVE_OUTPUT_DIR="data/glove/acl/data"
DATASET_TYPE='full'

echo "Preprocessing GloVe data..."
python preprocess_dataset.py \
    --dataset=${DATASET} \
    --data-dir=${DATA_DIR} \
    --data-output-dir=${DATA_OUTPUT_DIR} \
    --dataset-type=${DATASET_TYPE} \
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
EMBEDDING_PATH="data/fasttext/acl/fasttext.pkl"
EMBEDDING_WORDINDEX_PATH="data/fasttext/acl/fasttext_word_index.pkl"
FASTTEXT_OUTPUT_DIR="data/fasttext/acl/data"
EMBED_SIZE=300

echo "Preprocessing FastText data..."
python preprocess_dataset.py \
    --dataset=${DATASET} \
    --data-dir=${DATA_DIR} \
    --data-output-dir=${DATA_OUTPUT_DIR} \
    --dataset-type=${DATASET_TYPE} \
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
EMBEDDING_PATH="data/word2vec/acl/word2vec.pkl"
EMBEDDING_WORDINDEX_PATH="data/word2vec/acl/word2vec_word_index.pkl"
WORD2VEC_OUTPUT_DIR="data/word2vec/acl/data"
EMBED_SIZE=300

echo "Preprocessing Word2Vec data..."
python preprocess_dataset.py \
    --dataset=${DATASET} \
    --data-dir=${DATA_DIR} \
    --data-output-dir=${DATA_OUTPUT_DIR} \
    --dataset-type=${DATASET_TYPE} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${WORD2VEC_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${WORD2VEC_OUTPUT_DIR}

EMBEDDING_PATH="data/debug/acl/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/debug/acl/glove_word_index.pkl"
DEBUG_OUTPUT_DIR="data/debug/acl/data"
DATASET_TYPE='debug'

echo "Preprocessing DEBUG data..."
python preprocess_dataset.py \
    --dataset=${DATASET} \
    --data-dir=${DATA_DIR} \
    --data-output-dir=${DATA_OUTPUT_DIR} \
    --dataset-type=${DATASET_TYPE} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${GLOVE_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${DEBUG_OUTPUT_DIR}

EMBEDDING_PATH="data/active_learning/acl/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/active_learning/acl/glove_word_index.pkl"
ACTIVE_LEARNING_OUTPUT_DIR="data/active_learning/acl/data"
DATASET_TYPE='active_learning'

echo "Preprocessing Active Learning data..."
python preprocess_dataset.py \
    --dataset=${DATASET} \
    --data-dir=${DATA_DIR} \
    --data-output-dir=${DATA_OUTPUT_DIR} \
    --dataset-type=${DATASET_TYPE} \
    --train-save-path=${TRAIN_SAVE_PATH} \
    --validation-save-path=${VALIDATION_SAVE_PATH} \
    --test-save-path=${TEST_SAVE_PATH} \
    --embedding-file=${GLOVE_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${ACTIVE_LEARNING_OUTPUT_DIR}
