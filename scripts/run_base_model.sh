#!/bin/bash

set -e

#usage
#./script/run_base_model.sh

TRAIN_FILE="data/aclImdb_formatted/train/train.tfrecord"
VALIDATION_FILE="data/aclImdb_formatted/val/val.tfrecord"
TEST_FILE="data/aclImdb_formatted/test/test.tfrecord"

NUM_TRAIN=22500
NUM_VALIDATION=2500
NUM_TEST=25000

MODEL_NAME='base_model'
TENSORBOARD_DIR='tensorboard_logs'

GLOVE_FILE="data/glove.6B.50d.txt"
GLOVE_PICKLE="data/glove.pkl"

GRAPHS_DIR='graphs'

BATCH_SIZE=128
NUM_EPOCHS=25
EMBED_SIZE=50
NUM_UNITS=64
NUM_CLASSES=2
MAX_LENGTH=250

PERFORM_SHUFFLE=true
LSTM_OUTPUT_DROPOUT=0.5
LSTM_STATE_DROPOUT=0.5
USE_TEST=false


python base_model.py \
    --train-file=${TRAIN_FILE} \
    --validation-file=${VALIDATION_FILE} \
    --test-file=${TEST_FILE} \
    --num-train=${NUM_TRAIN} \
    --num-validation=${NUM_VALIDATION} \
    --num-test=${NUM_TEST} \
    --graphs-dir=${GRAPHS_DIR} \
    --model-name=${MODEL_NAME} \
    --tensorboard-dir=${TENSORBOARD_DIR} \
    --glove-file=${GLOVE_FILE} \
    --glove-pickle=${GLOVE_PICKLE} \
    --batch-size=${BATCH_SIZE} \
    --num-epochs=${NUM_EPOCHS} \
    --perform-shuffle=${PERFORM_SHUFFLE} \
    --embed-size=${EMBED_SIZE} \
    --num-units=${NUM_UNITS} \
    --max-length=${MAX_LENGTH} \
    --num-classes=${NUM_CLASSES} \
    --lstm-output-dropout=${LSTM_OUTPUT_DROPOUT} \
    --lstm-state-dropout=${LSTM_STATE_DROPOUT} \
    --use-test=${USE_TEST}