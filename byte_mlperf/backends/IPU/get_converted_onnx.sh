# Copyright 2023 Graphcore Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
echo "************************** Downloading converted ONNX model... **************************"

if [ $1 != "albert-torch-fp32" -a $1 != "bert-torch-fp32" -a $1 != "resnet50-torch-fp32" ] ; then
    printf "Skip downloading, converted ONNX only available for: albert-torch-fp32, bert-torch-fp32, resnet50-torch-fp32\n"
    exit 0
fi

echo "The model name is: $1"
TAR_NAME="$1-onnx.tar"
ONNX_NAME="$1.onnx"
if [ ! -d "byte_mlperf/backends/IPU/pre_optimized_models/" ]; then
    mkdir -p "byte_mlperf/backends/IPU/pre_optimized_models/"
fi

if [ ! -f "byte_mlperf/backends/IPU/pre_optimized_models/$ONNX_NAME" ] ; then
    wget -O byte_mlperf/download/$TAR_NAME https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/$TAR_NAME
    tar xf byte_mlperf/download/$TAR_NAME -C byte_mlperf/backends/IPU/pre_optimized_models/
    mv byte_mlperf/backends/IPU/pre_optimized_models/converted_models/$ONNX_NAME byte_mlperf/backends/IPU/pre_optimized_models/
    rm -r byte_mlperf/backends/IPU/pre_optimized_models/converted_models
    printf "$ONNX_NAME has been downloaded\n"
else
    printf "$ONNX_NAME is already existing, skip downloading\n"
fi
