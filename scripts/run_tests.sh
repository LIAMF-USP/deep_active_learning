#!/bin/bash

set -e

#usage
#./script/run_tests.sh $param
#param=all  - Run all tests
#param=slow - Run all slow tests
#param=fast - Run fast tests (Default argument)

PARAM=${1:-fast}
export TF_CPP_MIN_LOG_LEVEL=3

if [ $PARAM == "fast" ]; then
    echo "Running fast tests"
    nose2 -A "!slow"
elif [ $PARAM == "slow" ]; then
    echo "Running slow tests"
    nose2 -A slow
elif [ $PARAM == "all" ]; then
    echo "Running all tests"
    nose2
fi
