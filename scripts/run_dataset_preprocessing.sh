#!/bin/bash

set -e

#usage
#./script/run_dataset_preprocessing.sh

DATA_DIR="data/aclImdb"
DATASET_TRAIN="train"
DATASET_TEST="test"

GLOVE_FILE="data/glove.6B.100d.txt"
FASTTEXT_FILE="data/wiki.en.bin"

EMBEDDING_PATH="data/glove.pkl"
EMBEDDING_WORDINDEX_PATH="data/glove_word_index.pkl"

SENTENCE_SIZE=1000
OUTPUT_DIR="data/aclImdb_formatted"
EMBED_SIZE=100

echo "Preprocessing training data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TRAIN} \
    --embedding-file=${GLOVE_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${OUTPUT_DIR}

echo -e "\n\nPreprocessing test data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TEST} \
    --embedding-file=${GLOVE_FILE} \
    --embedding-path=${EMBEDDING_PATH} \
    --embedding-wordindex-path=${EMBEDDING_WORDINDEX_PATH} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${OUTPUT_DIR}
