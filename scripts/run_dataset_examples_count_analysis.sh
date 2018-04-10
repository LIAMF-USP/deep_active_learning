#!/bin/bash

set -e

#usage
#./script/run_dataset_examples_count_analysis.sh

DATA_DIR="data/aclImdb"

python experiment_analysis/movie_review_examples_count.py \
    --data_dir=${DATA_DIR}
