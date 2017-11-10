#!/bin/bash

set -e

#usage
#./script/run_dataset_preprocessing.sh

DATA_DIR="data/aclImdb"
DATASET_TRAIN="train"
DATASET_TEST="test"

GLOVE_FILE="data/glove.6B.50d.txt"
SENTENCE_SIZE=250
OUTPUT_DIR="data/aclImdb_formatted"

echo "Preprocessing training data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TRAIN} \
    --glove-file=${GLOVE_FILE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${OUTPUT_DIR}

echo -e "\n\nPreprocessing test data..."
python preprocess_dataset.py \
    --data-dir=${DATA_DIR} \
    --dataset-type=${DATASET_TEST} \
    --glove-file=${GLOVE_FILE} \
    --sentence-size=${SENTENCE_SIZE} \
    --output-dir=${OUTPUT_DIR}
