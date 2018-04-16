#!/bin/sh

#usage
#./scrips/download_data.sh

set -e

DATA_DIR="data/"

ACL_DATASET_URL="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ACL_DATASET_DIR="aclImdb/"
ACL_DATASET_TARBALL="aclImdb_v1.tar.gz"

SUBJ_DATASET_URL="http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"
SUBJ_DATASET_DIR="subj_dataset"
SUBJ_DATASET_TARBALL="rotten_imdb.tar.gz"

GLOVE_URL="http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP="glove.6B.zip"
GLOVE_FILE="glove.6B.50d.txt"
GLOVE_DIR="glove"

WORD2VEC_URL="https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
WORD2VEC_ZIP="GoogleNews-vectors-negative300.bin.gz"
COOKIES_FILE="cookies.txt"
WORD2VEC_FILE="GoogleNews-vectors-negative300.bin"
WORD2VEC_DIR="word2vec"

FASTTEXT_URL="https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip"
FASTEXT_ZIP="wiki.en.zip"
FASTTEXT_FILE="wiki.en.bin"
FASTTEXT_DIR="fasttext"

if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
fi

cd "$DATA_DIR"

if [ ! -d "$ACL_DATASET_DIR" ]; then
    wget "$ACL_DATASET_URL"
    tar xvf "$ACL_DATASET_TARBALL"
    rm "$ACL_DATASET_TARBALL"
else
    echo "ACL Dataset already downloaded!"
fi

if [ ! -d "$SUBJ_DATASET_DIR" ]; then
    mkdir "$SUBJ_DATASET_DIR"
    cd "$SUBJ_DATASET_DIR"

    wget "$SUBJ_DATASET_URL"
    tar xvf "$SUBJ_DATASET_TARBALL"
    rm "$SUBJ_DATASET_TARBALL"
    cd ..
else
    echo "Subjective Dataset already downloaded!"
fi

if [ ! -d "$GLOVE_DIR" ]; then
    mkdir "$GLOVE_DIR"
fi

cd "$GLOVE_DIR"

if [ ! -e "$GLOVE_FILE" ]; then
    wget "$GLOVE_URL"
    unzip "$GLOVE_ZIP" 
    rm "$GLOVE_ZIP"
else
    echo "GloVe file already downloaded!"
fi

cd ..

if [ ! -d "$WORD2VEC_DIR" ]; then
    mkdir "$WORD2VEC_DIR"
fi

cd "$WORD2VEC_DIR"

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

if [ ! -d "$FASTTEXT_DIR" ]; then
    mkdir "$FASTTEXT_DIR"
fi

cd "$FASTTEXT_DIR"

if [ ! -e "$FASTTEXT_FILE" ]; then
    wget "$FASTTEXT_URL"
    unzip "$FASTEXT_ZIP"
    rm "$FASTEXT_ZIP"
else
    echo "FastText file already downloaded!"
fi

cd ..

cd ..
