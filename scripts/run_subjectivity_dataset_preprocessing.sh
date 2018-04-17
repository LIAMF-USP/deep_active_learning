#!/bin/bash

set -e

#usage
#./script/run_subjectivity_dataset_preprocessing.sh

DATASET="subj"
DATA_DIR="data/subj_dataset"

TRAIN_SAVE_PATH="data/subj_dataset/train.pkl"
VALIDATION_SAVE_PATH="data/subj_dataset/validation.pkl"
TEST_SAVE_PATH="data/subj_dataset/test.pkl"

GLOVE_FILE="data/glove/glove.6B.100d.txt"

SENTENCE_SIZE=20
EMBED_SIZE=100

EMBEDDING_PATH="data/glove/subj/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/glove/subj/glove_word_index.pkl"
GLOVE_OUTPUT_DIR="data/glove/subj/data"
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
EMBEDDING_PATH="data/fasttext/subj/fasttext.pkl"
EMBEDDING_WORDINDEX_PATH="data/fasttext/subj/fasttext_word_index.pkl"
FASTTEXT_OUTPUT_DIR="data/fasttext/subj/data"
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
EMBEDDING_PATH="data/word2vec/subj/word2vec.pkl"
EMBEDDING_WORDINDEX_PATH="data/word2vec/subj/word2vec_word_index.pkl"
WORD2VEC_OUTPUT_DIR="data/word2vec/subj/data"
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

EMBEDDING_PATH="data/active_learning/subj/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/active_learning/subj/glove_word_index.pkl"
ACTIVE_LEARNING_OUTPUT_DIR="data/active_learning/subj/data"
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
