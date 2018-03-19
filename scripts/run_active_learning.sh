#!/bin/bash

set -e

#usage
#./script/run_active_learning.sh

AL_TRAIN_FILE="data/glove/aclImdb_formatted/train/train.pkl"
AL_TEST_FILE="data/glove/aclImdb_formatted/test/test.pkl"

SAVED_MODEL_FOLDER="saved_models"

NUM_TRAIN=25000
NUM_TEST=25000

MODEL_NAME='base_model'
TENSORBOARD_DIR='tensorboard_logs'

GLOVE_EMBEDDING_FILE="data/glove/glove.6B.100d.txt"
GLOVE_EMBEDDING_PICKLE="data/glove/glove.pkl"
GLOVE_EMBED_SIZE=100

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
LEARNING_RATE=0.00410
BATCH_SIZE=64
NUM_UNITS=285
RECURRENT_OUTPUT_DROPOUT=0.694
RECURRENT_STATE_DROPOUT=0.855
EMBEDDING_DROPOUT=0.267
WEIGHT_DECAY=0.0003097047
CLIP_GRADIENTS=0
MAX_NORM=5

EMBED_SIZE="$GLOVE_EMBED_SIZE"

#Active Learning parameters
NUM_ROUNDS=100
SAMPLE_SIZE=2000
NUM_QUERIES=50
NUM_PASSES=100
INITIAL_TRAINING_SIZE=10
SAVE_DATA_FOLDER='data/active_learning'
MAX_LEN=600

UNCERTAINTY_METRIC="bald"

for i in {1..3}
do
  SAVE_GRAPH_PATH=$GRAPHS_DIR'/'$UNCERTAINTY_METRIC'_'$i'.png'
  TRAIN_DATA_NAME='train_data_'$i'.pkl'
  TEST_ACC_NAME='test_accuracies_'$i'.pkl'

  python active_learning.py \
      --train-file=${AL_TRAIN_FILE} \
      --test-file=${AL_TEST_FILE} \
      --saved-model-folder=${SAVED_MODEL_FOLDER} \
      --num-train=${NUM_TRAIN} \
      --num-test=${NUM_TEST} \
      --graphs-dir=${GRAPHS_DIR} \
      --model-name=${MODEL_NAME} \
      --tensorboard-dir=${TENSORBOARD_DIR} \
      --embedding-file=${GLOVE_EMBEDDING_FILE} \
      --embedding-pickle=${GLOVE_EMBEDDING_PICKLE} \
      --learning-rate=${LEARNING_RATE} \
      --batch-size=${BATCH_SIZE} \
      --num-epochs=${NUM_EPOCHS} \
      --perform-shuffle=${PERFORM_SHUFFLE} \
      --embed-size=${GLOVE_EMBED_SIZE} \
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
      --uncertainty-metric=${UNCERTAINTY_METRIC} \
      --num-rounds=${NUM_ROUNDS} \
      --sample-size=${SAMPLE_SIZE} \
      --num-queries=${NUM_QUERIES} \
      --num-passes=${NUM_PASSES} \
      --max-len=${MAX_LEN} \
      --initial-training-size=${INITIAL_TRAINING_SIZE} \
      --save-graph-path=${SAVE_GRAPH_PATH} \
      --save-data-folder=${SAVE_DATA_FOLDER} \
      --train-data-name=${TRAIN_DATA_NAME} \
      --test-acc-name=${TEST_ACC_NAME}
done
