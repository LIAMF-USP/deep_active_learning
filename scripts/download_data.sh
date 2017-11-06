#!/bin/sh

#usage
#./scrips/download_data.sh

set -e

DATA_DIR="data/"
DATASET_URL="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_DIR="aclImdb/"
DATASET_TARBALL="aclImdb_v1.tar.gz"

GLOVE_URL="http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP="glove.6B.zip"
GLOVE_FILE="glove.6B.50d.txt"

if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
fi

cd "$DATA_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    wget "$DATASET_URL"
    tar xvf "$DATASET_TARBALL"
    rm "$DATASET_TARBALL"
else
    echo "Dataset already downloaded!"
fi

if [ ! -e "$GLOVE_FILE" ]; then
    wget "$GLOVE_URL"
    unzip "$GLOVE_ZIP" 
    rm "$GLOVE_ZIP"
else
    echo "GloVe file already downloaded!"
fi

cd ..
