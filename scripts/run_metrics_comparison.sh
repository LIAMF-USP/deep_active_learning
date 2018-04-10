#!/bin/bash

set -e

#usage
#./script/run_metrics_comparison.sh

DATA_FOLDER="data/active_learning"
NUM_EXPERIMENTS=3
METRICS_ACCURACY_FILE="test_accuracies"
NUM_DATA_FILE="train_data"

BALD_N100="bald_n100"
ENTROPY_N100="entropy_n100"
RANDOM_N100="random_n100"

BALD_N100_Q10="bald_n100_q10"
RANDOM_N100_Q10="random_n100_q10"

BALD="bald"
RAND="random"
ENTROPY="entropy"

GRAPH_PATH="graphs"
GRAPH_NAME="n100_q10_comparison.png"


python experiment_analysis/metrics_comparison.py \
  --data-folder=${DATA_FOLDER} \
  --num-experiments=${NUM_EXPERIMENTS} \
  --metrics-accuracy-file=${METRICS_ACCURACY_FILE} \
  --num-data-file=${NUM_DATA_FILE} \
  --metrics-folder ${BALD_N100_Q10} ${RANDOM_N100_Q10} \
  --metric-names ${BALD} ${RAND} \
  --graph-path=${GRAPH_PATH} \
  --graph-name=${GRAPH_NAME}
