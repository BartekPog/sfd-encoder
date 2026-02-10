#!/bin/bash

source .venv-sfd/bin/activate
CONFIG_PATH=$1
SEMANTIC_FEAT_TYPE=${2:-''}
REPA_DINO_MODEL_NAME=${3:-''}
IMAGE_SIZE=${4:-''}


GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}

cmd_args="--config $CONFIG_PATH"

if [ -n "$SEMANTIC_FEAT_TYPE" ]; then
    cmd_args="$cmd_args --semantic_feat_type $SEMANTIC_FEAT_TYPE"
fi

if [ -n "$REPA_DINO_MODEL_NAME" ]; then
    cmd_args="$cmd_args --repa_dino_model_name $REPA_DINO_MODEL_NAME"
fi

if [ -n "$IMAGE_SIZE" ]; then
    cmd_args="$cmd_args --image_size $IMAGE_SIZE"
fi

echo "execute: $cmd_args"

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    extract_features.py \
    $cmd_args