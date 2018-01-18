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

EMBEDDING_FILE="data/glove.6B.100d.txt"
EMBEDDING_PICKLE="data/glove.pkl"

GRAPHS_DIR='graphs'

NUM_EPOCHS=18
EMBED_SIZE=100
NUM_CLASSES=2

PERFORM_SHUFFLE=1
USE_TEST=0

#Bucket paramets
BUCKET_WIDTH=30
NUM_BUCKETS=30

#Hyper-parameters
BATCH_SIZE=128
NUM_UNITS=128
LSTM_OUTPUT_DROPOUT=0.5
LSTM_STATE_DROPOUT=0.5
EMBEDDINF_DROPOUT=0.5
WEIGHT_DECAY=0.000001

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
    --embedding-file=${EMBEDDING_FILE} \
    --embedding-pickle=${EMBEDDING_PICKLE} \
    --batch-size=${BATCH_SIZE} \
    --num-epochs=${NUM_EPOCHS} \
    --perform-shuffle=${PERFORM_SHUFFLE} \
    --embed-size=${EMBED_SIZE} \
    --num-units=${NUM_UNITS} \
    --num-classes=${NUM_CLASSES} \
    --lstm-output-dropout=${LSTM_OUTPUT_DROPOUT} \
    --lstm-state-dropout=${LSTM_STATE_DROPOUT} \
    --embedding-dropout=${EMBEDDINF_DROPOUT} \
    --weight-decay=${WEIGHT_DECAY} \
    --bucket-width=${BUCKET_WIDTH} \
    --num-buckets=${NUM_BUCKETS} \
    --use-test=${USE_TEST}
