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

WORD2VEC_TRAIN_FILE="data/word2vec/aclImdb_formatted/train/train.tfrecord"
WORD2VEC_VALIDATION_FILE="data/word2vec/aclImdb_formatted/val/val.tfrecord"
WORD2VEC_TEST_FILE="data/word2vec/aclImdb_formatted/test/test.tfrecord"

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

WORD2VEC_EMBEDDING_FILE="data/word2vec/GoogleNews-vectors-negative300.bin"
WORD2VEC_EMBEDDING_PICKLE="data/word2vec/word2vec.pkl"
WORD2VEC_EMBED_SIZE=300

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
RECURRENT_OUTPUT_DROPOUT=0.5
RECURRENT_STATE_DROPOUT=0.5
EMBEDDING_DROPOUT=0.5
WEIGHT_DECAY=0.000001
CLIP_GRADIENTS=0
MAX_NORM=5

TRAIN_FILE="$GLOVE_TRAIN_FILE"
VALIDATION_FILE="$GLOVE_VALIDATION_FILE"
TEST_FILE="$GLOVE_TEST_FILE"

EMBEDDING_FILE="$GLOVE_EMBEDDING_FILE"
EMBEDDING_PICKLE="$GLOVE_EMBEDDING_PICKLE"
EMBED_SIZE="$GLOVE_EMBED_SIZE"

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
    --recurrent-output-dropout=${RECURRENT_OUTPUT_DROPOUT} \
    --recurrent-state-dropout=${RECURRENT_STATE_DROPOUT} \
    --embedding-dropout=${EMBEDDING_DROPOUT} \
    --clip-gradients=${CLIP_GRADIENTS} \
    --max-norm=${MAX_NORM} \
    --weight-decay=${WEIGHT_DECAY} \
    --bucket-width=${BUCKET_WIDTH} \
    --num-buckets=${NUM_BUCKETS} \
    --use-test=${USE_TEST}
