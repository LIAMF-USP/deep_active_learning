#!/bin/bash

set -e

#usage
#./script/run_base_model.sh

TRAIN_POS_FILE="data/aclImdb_formatted/train/pos/pos.tfrecord"
TRAIN_NEG_FILE="data/aclImdb_formatted/train/neg/neg.tfrecord"

VALIDATION_POS_FILE="data/aclImdb_formatted/val/pos/pos.tfrecord"
VALIDATION_NEG_FILE="data/aclImdb_formatted/val/neg/neg.tfrecord"

TEST_POS_FILE="data/aclImdb_formatted/test/pos/pos.tfrecord"
TEST_NEG_FILE="data/aclImdb_formatted/test/neg/neg.tfrecord"

GLOVE_FILE="data/glove.6B.50d.txt"

BATCH_SIZE=128
NUM_EPOCHS=1000
EMBED_SIZE=50
NUM_UNITS=64
NUM_CLASSES=2
MAX_LENGTH=250

PERFORM_SHUFFLE=true


python base_model.py \
    --train-files ${TRAIN_POS_FILE} ${TRAIN_NEG_FILE} \
    --validation-files ${VALIDATION_POS_FILE} ${VALIDATION_NEG_FILE} \
    --test-files ${TEST_POS_FILE} ${TEST_NEG_FILE} \
    --batch-size=${BATCH_SIZE} \
    --num-epochs=${NUM_EPOCHS} \
    --perform-shuffle=${PERFORM_SHUFFLE} \
    --glove-file=${GLOVE_FILE} \
    --embed-size=${EMBED_SIZE} \
    --num-units=${NUM_UNITS} \
    --max-length=${MAX_LENGTH} \
    --num-classes=${NUM_CLASSES}
