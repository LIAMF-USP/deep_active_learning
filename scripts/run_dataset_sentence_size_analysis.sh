#!/bin/bash

set -e

#usage
#./script/run_dataset_sentence_size_analysis.sh

DATA_DIR="data/aclImdb"
DATASET_TRAIN="train"
DATASET_TEST="test"

python experiment_analysis/movie_review_sentence_size.py \
    --data_dir=${DATA_DIR} \
    --dataset_type=${DATASET_TRAIN}

python experiment_analysis/movie_review_sentence_size.py \
    --data_dir=${DATA_DIR} \
    --dataset_type=${DATASET_TEST}
