#!/bin/bash
set -e

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/../build

if [ ! -d "$BUILD_PATH" ]; then
    echo "Test cases are not built."
fi

if [ ! -f "${BUILD_PATH}/tests" ]; then
    echo "example test file is missing."
    exit 1
fi

usage () {
    echo "runExampleTests.sh need one param"
    echo "Supported param can be [fma_float32, fma_float16, fma_bfloat16, fma_int8]"
    echo "such as ./runExampleTests.sh fma_float32"
}


pushd ${BUILD_PATH} > /dev/null
    if [ "$#" -eq "0" ]; then
        usage
    elif [ "$#" -eq "1" ]; then
        echo "-- Running cambricon release performance test."
        ./tests --task $1;
    else
        echo "argv num should not be greater than 1"
        exit 1
    fi
popd > /dev/null
