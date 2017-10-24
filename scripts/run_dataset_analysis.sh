#!/bin/bash

set -e

#usage
#./script/run_dataset_analysis.sh

DATA_DIR="data/aclImdb"
DATASET_TYPE="train"

python dataset_analysis/movie_review_dataset_analysis.py \
    --data_dir=${DATA_DIR} \
    --dataset_type=${DATASET_TYPE}
