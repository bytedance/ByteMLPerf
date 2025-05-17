#!/bin/bash


HDFS_trainingdata=/mnt/hdfs/training_data

## /home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM
WORK_PATH=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM

cd /home/tiger/
git clone git@code.byted.org:data/ByteMLPerf.git
cd ByteMLPerf 
git fetch 
git checkout hht/hwj_training

cd $WORK_PATH
echo "=== check 0 ===, $WORK_PATH, $HDFS_trainingdata"




export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export no_proxy="*.byted.org"
pip install -e . 
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#MAX_JOBS=4  pip install flash_attn==2.4.2  --no-build-isolatio
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention 
git submodule update --init --recursive
git checkout v2.4.2
python3  setup.py install --user
pip3 install transformer_engine[pytorch]==1.11.0
cd $WORK_PATH
git clone https://github.com/NVIDIA/apex.git
cd apex
NVCC_APPEND_FLAGS="--threads 4" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" ./

# source /home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/venv/bin/activate
cd $WORK_PATH

export OMPI_COMM_WORLD_SIZE=$2
export OMPI_COMM_WORLD_RANK=$3

PRETRAIN_PATH=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/examples/ds_like/node2ckpt
CHECKPOINT_PATH=/mnt/hdfs/training_data/node${OMPI_COMM_WORLD_SIZE}/moe_ckpt
TENSORBOARD_LOGS_PATH=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/examples/ds_like/node2ckpt/tensorboard
VOCAB_FILE=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/gpt2-vocab.json
MERGE_FILE=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/gpt2-merges.txt
DATA_PATH=$1
TOKENIZER_MODEL=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/examples/ds_like/Llama2tokenizer.model
#bash examples/mixtral/train_moe.sh $CHECKPOINT_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH $TOKENIZER_MODEL 

#source /home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/venv/bin/activate


export OMPI_COMM_WORLD_SIZE=$2
export OMPI_COMM_WORLD_RANK=$3

MAX_TRAIN_SAMPLES=15360000

bash ./examples/ds_like/train_ds_like_4node_profile.sh $PRETRAIN_PATH $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $DATA_PATH $TOKENIZER_MODEL $MAX_TRAIN_SAMPLES $4 $5

