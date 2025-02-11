#!/bin/bash

# Runs the "345M" parameter model

CHECKPOINT_PATH=/data02/huanghantao/Megatron-LM/examples/gpt3/ckpt_345M/ckpt
TENSORBOARD_LOGS_PATH=/data02/huanghantao/Megatron-LM/examples/gpt3/ckpt_345M/tensorboard
VOCAB_FILE=/data02/huanghantao/Megatron-LM/gpt2-vocab.json
MERGE_FILE=/data02/huanghantao/Megatron-LM/gpt2-merges.txt
DATA_PATH=/data02/huanghantao/Megatron-LM/my-gpt2_text_document
bash examples/gpt3/train_gpt3_345M_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH
