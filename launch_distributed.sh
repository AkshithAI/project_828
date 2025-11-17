#!/bin/bash

# DeepSpeed Distributed Training Launch Script
# Usage: ./launch_distributed.sh [num_gpus]

NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified

echo "Launching distributed training on $NUM_GPUS GPUs..."

deepspeed --num_gpus=$NUM_GPUS \
    src/scripts/distributed_training.py \
    --deepspeed \
    --deepspeed_config src/scripts/ds-config.json

echo "Training completed!"
