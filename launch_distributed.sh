#!/bin/bash

# Installing all the requirements
pip install -r requirements.txt

# Wandb Login
wandb login

# Launch Training
PYTHONPATH=$(pwd) 
deepspeed --num_gpus=2 \
    src/scripts/distributed_training.py 
    --deepspeed
    --deepspeed_config src/scripts/ds-config.json 
    --batch_size 8

