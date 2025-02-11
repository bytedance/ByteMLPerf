#!/bin/bash

# Runs the "345M" parameter model

CHECKPOINT_PATH=/data02/huanghantao/Megatron-LM/examples/mixtral/moe/ckpt
TENSORBOARD_LOGS_PATH=/data02/huanghantao/Megatron-LM/examples/mixtral/moe/tensorboard
VOCAB_FILE=/data02/huanghantao/Megatron-LM/gpt2-vocab.json
MERGE_FILE=/data02/huanghantao/Megatron-LM/gpt2-merges.txt
DATA_PATH=/data02/huanghantao/Megatron-LM/moe_test_text_document
TOKENIZER_MODEL=/data02/huanghantao/Megatron-LM/examples/mixtral/Llama2tokenizer.model
bash examples/mixtral/train_moe.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH $TOKENIZER_MODEL 
