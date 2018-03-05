#!/bin/bash

set -e

#usage
#./script/run_active_learning.sh

AL_TRAIN_FILE="data/active_learning/aclImdb_formatted/train/train.pkl"
AL_TEST_FILE="data/active_learning/aclImdb_formatted/test/test.pkl"

SAVED_MODEL_FOLDER="saved_models"

NUM_TRAIN=25000
NUM_TEST=25000

MODEL_NAME='base_model'
TENSORBOARD_DIR='tensorboard_logs'

FASTTEXT_EMBEDDING_FILE="data/fasttext/wiki.en.vec"
FASTTEXT_EMBEDDING_PICKLE="data/fasttext/fasttext.pkl"
FASTTEXT_EMBED_SIZE=300

GRAPHS_DIR='graphs'

NUM_EPOCHS=16
NUM_CLASSES=2

USE_VALIDATION=0
PERFORM_SHUFFLE=1
SAVE_GRAPH=0
USE_TEST=1

#Bucket paramets
BUCKET_WIDTH=30
NUM_BUCKETS=30

#Hyper-parameters
LEARNING_RATE=0.00214
BATCH_SIZE=32
NUM_UNITS=161
RECURRENT_OUTPUT_DROPOUT=0.905
RECURRENT_STATE_DROPOUT=0.331
EMBEDDING_DROPOUT=0.759
WEIGHT_DECAY=0.0000713144
CLIP_GRADIENTS=0
MAX_NORM=5

EMBED_SIZE="$GLOVE_EMBED_SIZE"

#Active Learning parameters
NUM_ROUNDS=118
SAMPLE_SIZE=2000
NUM_QUERIES=50
NUM_PASSES=50
INITIAL_TRAINING_SIZE=10
SAVE_GRAPH_PATH=$GRAPHS_DIR'/al_var_ratio.png'
SAVE_DATA_FOLDER='data/active_learning'

python active_learning.py \
    --train-file=${AL_TRAIN_FILE} \
    --test-file=${AL_TEST_FILE} \
    --saved-model-folder=${SAVED_MODEL_FOLDER} \
    --num-train=${NUM_TRAIN} \
    --num-test=${NUM_TEST} \
    --graphs-dir=${GRAPHS_DIR} \
    --model-name=${MODEL_NAME} \
    --tensorboard-dir=${TENSORBOARD_DIR} \
    --embedding-file=${FASTTEXT_EMBEDDING_FILE} \
    --embedding-pickle=${FASTTEXT_EMBEDDING_PICKLE} \
    --learning-rate=${LEARNING_RATE} \
    --batch-size=${BATCH_SIZE} \
    --num-epochs=${NUM_EPOCHS} \
    --perform-shuffle=${PERFORM_SHUFFLE} \
    --embed-size=${FASTTEXT_EMBED_SIZE} \
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
    --use-test=${USE_TEST} \
    --save-graph=${SAVE_GRAPH} \
    --num-rounds=${NUM_ROUNDS} \
    --sample-size=${SAMPLE_SIZE} \
    --num-queries=${NUM_QUERIES} \
    --num-passes=${NUM_PASSES} \
    --initial-training-size=${INITIAL_TRAINING_SIZE} \
    --save-graph-path=${SAVE_GRAPH_PATH} \
    --save-data-folder=${SAVE_DATA_FOLDER}
