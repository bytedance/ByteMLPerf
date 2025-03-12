#!/bin/bash

# Runs the "345M" parameter model

PRETRAIN_PATH=/data03/huanghantao/download.cambricon.com:8821/upload/bytemlperf/ckpt/moe_ckpt/moe_ckpt
CHECKPOINT_PATH=/data02/huanghantao/ByteMLPerf/byte_train_perf/Megatron-LM/examples/mixtral/moe_ckpt
TENSORBOARD_LOGS_PATH=/data02/huanghantao/Megatron-LM-v0/examples/mixtral/moe_resume/tensorboard
VOCAB_FILE=/data02/huanghantao/Megatron-LM-v0/gpt2-vocab.json
MERGE_FILE=/data02/huanghantao/Megatron-LM-v0/gpt2-merges.txt
DATA_PATH=/data03/huanghantao/megatron-data/moe_test_text_document
TOKENIZER_MODEL=/data02/huanghantao/Megatron-LM-v0/examples/mixtral/Llama2tokenizer.model
#bash examples/mixtral/train_moe.sh $CHECKPOINT_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH $TOKENIZER_MODEL 
bash examples/mixtral/train_moe_resume.sh $PRETRAIN_PATH $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $DATA_PATH $TOKENIZER_MODEL 10000


# PRETRAIN_MODEL_PATH=$1 #<Specify path>
# CHECKPOINT_SAVE_PATH=$2 #<Specify path>
# TENSORBOARD_LOGS_PATH=$3 #<Specify path>
# # VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# # MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$4 #<Specify path and file prefix>_text_document
# TOKENIZER_MODEL=$5  #tokenizer path
# MAX_TRAIN_STEPS=$6 #train iters