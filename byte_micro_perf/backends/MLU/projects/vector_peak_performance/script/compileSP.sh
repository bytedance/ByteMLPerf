#!/bin/bash
set -e

SCRIPT_DIR=`dirname $0`
BUILD_PATH=${SCRIPT_DIR}/../build
CMAKE=cmake

if [ ! -d "$BUILD_PATH" ]; then
  mkdir "$BUILD_PATH"
fi

if [ ! -z "${NEUWARE_HOME}" ]; then
  echo "-- using NEUWARE_HOME = ${NEUWARE_HOME}"
else
  echo "-- NEUWARE_HOME is null, refer README.md to prepare NEUWARE_HOME environment."
  exit -1
fi

check_dep() {
  which cmake3 >/dev/null && CMAKE=cmake3 \
    || which cmake >/dev/null \
    || echo "CMake not found, please install cmake >= 3.5 or set CMAKE env to your cmake command path"
   echo "CMAKE: $(which ${CMAKE}), version: $(${CMAKE} --version)"
}

check_dep

pushd ${BUILD_PATH} > /dev/null
  rm -rf *
  echo "-- Build cambricon release test cases."
  ${CMAKE}  ../ -DNEUWARE_HOME="${NEUWARE_HOME}"
  make -j
popd > /dev/null
