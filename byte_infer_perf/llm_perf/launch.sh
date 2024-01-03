#!/bin/bash
set -xe

TASK=${TASK:-"chatglm"}
HARDWARE_TYPE=${HARDWARE_TYPE:-"GPU"}
PORT=${PORT:-"50051"}
NPROC=${NPROC:-1}

torchrun --nproc-per-node $NPROC llm_perf/launch.py \
    --task=${TASK} \
    --hardware_type=${HARDWARE_TYPE} \
    --port=${PORT}