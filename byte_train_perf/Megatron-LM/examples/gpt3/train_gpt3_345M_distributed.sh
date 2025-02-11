#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

export NCCL_ALGO=Ring # for deterministic
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0  # for deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for deterministic

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 32 
    --global-batch-size 512 #1536 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --deterministic-mode # for deterministic
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --wandb-project benchmark_training
    --wandb-exp-name gpt3_345M_deterministic
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
