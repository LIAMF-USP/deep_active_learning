#!/bin/sh

#usage: ./scripts/run_random_hyperparameter_search.sh

set -e

NUM_SAMPLES=1
SAVE_FOLDER='best_model'

python random_hyperparameter_search.py \
  --num-samples=${NUM_SAMPLES} \
  --save-folder=${SAVE_FOLDER}
