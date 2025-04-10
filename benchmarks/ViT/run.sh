#!/bin/bash
export NNODES=${NNODES:-2}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export RDVZ_ID=${RDVZ_ID:-125}
torchrun  \
    --nnodes=$NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=$RANK \
    --rdzv-id=$RDVZ_ID \
    --rdzv-endpoint=$ENDPOINT \
    train.py \
    $@ \
