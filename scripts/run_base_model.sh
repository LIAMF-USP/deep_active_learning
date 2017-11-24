#!/bin/bash

set -e

#usage
#./script/run_base_model.sh

TRAIN_FILE="data/aclImdb_formatted/train/train.tfrecord"
VALIDATION_FILE="data/aclImdb_formatted/val/val.tfrecord"
TEST_FILE="data/aclImdb_formatted/test/test.tfrecord"

GLOVE_FILE="data/glove.6B.50d.txt"
GLOVE_PICKLE="data/glove.pkl"

BATCH_SIZE=128
NUM_EPOCHS=1000
EMBED_SIZE=50
NUM_UNITS=64
NUM_CLASSES=2
MAX_LENGTH=250

PERFORM_SHUFFLE=true


python base_model.py \
    --train-file=${TRAIN_FILE} \
    --validation-file=${VALIDATION_FILE} \
    --test-file=${TEST_FILE} \
    --batch-size=${BATCH_SIZE} \
    --num-epochs=${NUM_EPOCHS} \
    --perform-shuffle=${PERFORM_SHUFFLE} \
    --glove-file=${GLOVE_FILE} \
    --glove-pickle=${GLOVE_PICKLE} \
    --embed-size=${EMBED_SIZE} \
    --num-units=${NUM_UNITS} \
    --max-length=${MAX_LENGTH} \
    --num-classes=${NUM_CLASSES}
