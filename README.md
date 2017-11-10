# Deep Active Learning

[![Build Status](https://travis-ci.org/LIAMF-USP/deep_active_learning.svg?branch=master)](https://travis-ci.org/LIAMF-USP/deep_active_learning)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/LIAMF-USP/deep_active_learning/blob/master/LICENSE)

This respository will contain the code for the master's thesis "Deep Active
Learning for Sentiment Analysis". It will store not only the Deep Learning
models created, but also both experiments and dataset analysis code.

## Download data

To downlod the data used in this project, run the following command:

```sh
$ ./scripts/download_data.sh
```

## Preprocessing the dataset

In order to apply preprocessing to the dataset, run the following command:

```sh
$ ./scripts/run_dataset_preprocessing.sh
```

This script will perform the following steps:

* Read both positive and negative reviews from the train directory of the dataset
* Format the string for both positive and negative reviews (i.e. remove HTML tags)
* Create a validation set
* Create vocabulary using the GloVe embeddings
* Turn the reviews into a list of ids (Every id represent the row associated with the word in the GloVe matrix) 
* Save the reviews into TFRecord format

This steps are also applied for the test data, but without the validation set part.
