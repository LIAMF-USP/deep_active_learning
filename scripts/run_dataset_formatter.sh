#!/bin/bash

set -e

#usage
#./script/run_dataset_formatter.sh

DATA_DIR="data/aclImdb"
DATASET_TRAIN="train"
DATASET_TEST="test"
OUTPUT_DIR="data/aclImdb_formatted"

python dataset_formatter/format_dataset.py \
    --data_dir=${DATA_DIR} \
    --dataset_type=${DATASET_TRAIN} \
    --output_dir=${OUTPUT_DIR}

