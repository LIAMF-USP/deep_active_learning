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

WORD2VEC_URL="https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
WORD2VEC_ZIP="GoogleNews-vectors-negative300.bin.gz"
COOKIES_FILE="cookies.txt"
WORD2VEC_FILE="GoogleNews-vectors-negative300.bin"


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

# Based on the following script:
# https://gist.github.com/yanaiela/cfef50380de8a5bfc8c272bb0c91d6e1
if [ ! -e "$WORD2VEC_FILE" ]; then
    OUTPUT=$( wget --save-cookies "$COOKIES_FILE" --keep-session-cookies --no-check-certificate "$WORD2VEC_URL" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p' )
    CODE=${OUTPUT##*Code: }

    wget --load-cookies "$COOKIES_FILE" 'https://docs.google.com/uc?export=download&confirm='$CODE'&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O "$WORD2VEC_ZIP"
    gunzip "$WORD2VEC_FILE"
    rm "$COOKIES_FILE"
else
    echo "Word2vec file already downloaded!"
fi

cd ..
