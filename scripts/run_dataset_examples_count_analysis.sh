#!/bin/bash

set -e

#usage
#./script/run_dataset_examples_count_analysis.sh

PARAM=${1:-lmrd}

if [ $PARAM == "lmrd" ]; then
  echo "Creating count graph for Large Movie Review Dataset"
  DATA_DIR="data/aclImdb"
  DATASET_NAME="lmrd"
elif [ $PARAM == "sd" ]; then
  echo "Creating count graph for Subjectivity Dataset"
  DATA_DIR="data/subj_dataset"
  DATASET_NAME="sd"
fi


python experiment_analysis/movie_review_examples_count.py \
    --data_dir=${DATA_DIR} \
    --dataset_name=${DATASET_NAME}
