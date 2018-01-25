#!/bin/bash

set -e

#usage
#./script/run_base_model.sh

GLOVE_TRAIN_FILE="data/glove/aclImdb_formatted/train/train.tfrecord"
GLOVE_VALIDATION_FILE="data/glove/aclImdb_formatted/val/val.tfrecord"
GLOVE_TEST_FILE="data/glove/aclImdb_formatted/test/test.tfrecord"

FASTTEXT_TRAIN_FILE="data/fasttext/aclImdb_formatted/train/train.tfrecord"
FASTTEXT_VALIDATION_FILE="data/fasttext/aclImdb_formatted/val/val.tfrecord"
FASTTEXT_TEST_FILE="data/fasttext/aclImdb_formatted/test/test.tfrecord"

NUM_TRAIN=22500
NUM_VALIDATION=2500
NUM_TEST=25000

MODEL_NAME='base_model'
TENSORBOARD_DIR='tensorboard_logs'

GLOVE_EMBEDDING_FILE="data/glove/glove.6B.100d.txt"
GLOVE_EMBEDDING_PICKLE="data/glove/glove.pkl"
GLOVE_EMBED_SIZE=100

FASTTEXT_EMBEDDING_FILE="data/fasttext/wiki.en.vec"
FASTTEXT_EMBEDDING_PICKLE="data/fasttext/fasttext.pkl"
FASTTEXT_EMBED_SIZE=300

GRAPHS_DIR='graphs'

NUM_EPOCHS=18
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

TRAIN_FILE="$FASTTEXT_TRAIN_FILE"
VALIDATION_FILE="$FASTTEXT_VALIDATION_FILE"
TEST_FILE="$FASTTEXT_TEST_FILE"

EMBEDDING_FILE="$FASTTEXT_EMBEDDING_FILE"
EMBEDDING_PICKLE="$FASTTEXT_EMBEDDING_PICKLE"
EMBED_SIZE="$FASTTEXT_EMBED_SIZE"

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
