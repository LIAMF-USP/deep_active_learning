#!/bin/sh

#usage
#./scrips/download_data.sh

set -e

DATA_DIR="data/"
DATASET_URL="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_DIR="aclImdb/"

if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
fi

cd "$DATA_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    wget "$DATASET_URL"
    tar xvf aclImdb_v1.tar.gz
    rm aclImdb_v1.tar.gz
else
    echo "Dataset already downloaded!"
fi

cd ..
