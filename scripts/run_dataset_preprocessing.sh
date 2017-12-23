#!/bin/bash

set -e

#usage
#./script/run_dataset_preprocessing.sh

DATA_DIR="data/aclImdb"
DATASET_TRAIN="train"
DATASET_TEST="test"

EMBEDDING_FILE="data/glove.6B.100d.txt"
SENTENCE_SIZE=1000
OUTPUT_DIR="data/aclImdb_formatted"
EMBED_SIZE=100

echo "Preprocessing training data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TRAIN} \
    --embedding-file=${EMBEDDING_FILE} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${OUTPUT_DIR}

echo -e "\n\nPreprocessing test data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TEST} \
    --embedding-file=${EMBEDDING_FILE} \
    --embed-size=${EMBED_SIZE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${OUTPUT_DIR}
