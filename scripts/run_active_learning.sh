#!/bin/bash

set -e

#usage
#./script/run_active_learning.sh

AL_TRAIN_FILE="data/glove/subj/data/train/train.pkl"
AL_TEST_FILE="data/glove/subj/data/test/test.pkl"

SAVED_MODEL_FOLDER="saved_models"

NUM_TRAIN=25000
NUM_TEST=25000

MODEL_NAME='base_model'
TENSORBOARD_DIR='tensorboard_logs'

GLOVE_EMBEDDING_FILE="data/glove/glove.6B.100d.txt"
GLOVE_EMBEDDING_PICKLE="data/glove/subj/glove.pkl"
GLOVE_EMBED_SIZE=100

GRAPHS_DIR='graphs'

NUM_CLASSES=2

USE_VALIDATION=0
PERFORM_SHUFFLE=1
SAVE_GRAPH=0
USE_TEST=1

#Bucket paramets
BUCKET_WIDTH=30
NUM_BUCKETS=30

#Hyper-parameters
LEARNING_RATE=0.001
BATCH_SIZE=64
RECURRENT_INPUT_DROPOUT=0.5
RECURRENT_OUTPUT_DROPOUT=0.5
RECURRENT_STATE_DROPOUT=0.5
EMBEDDING_DROPOUT=0.5
WEIGHT_DECAY=0.0003097047
CLIP_GRADIENTS=1
MAX_NORM=5

EMBED_SIZE="$GLOVE_EMBED_SIZE"
MAX_LEN=20

#Active Learning parameters
NUM_UNITS=728
NUM_EPOCHS=150
NUM_ROUNDS=200
SAMPLE_SIZE=2000
NUM_QUERIES=10
NUM_PASSES=400
INITIAL_TRAINING_SIZE=10
ACTIVE_LEARNING_TYPE="common"
UNCERTAINTY_METRIC="bald"

SAVE_DATA_FOLDER='data/active_learning/bald_subj_n150_q10_u728_np400'

for i in {1..3}
do
  SAVE_GRAPH_PATH=$GRAPHS_DIR'/'$UNCERTAINTY_METRIC'_'$i'.png'
  TRAIN_DATA_NAME='train_data_'$i'.pkl'
  TEST_ACC_NAME='test_accuracies_'$i'.pkl'

  python -u active_learning.py \
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
      --recurrent-input-dropout=${RECURRENT_INPUT_DROPOUT} \
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
      --active-learning-type=${ACTIVE_LEARNING_TYPE} \
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
