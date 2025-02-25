#!/bin/bash
HOSTNAME=$(hostname)
#GPUS=1
#CUDA_VISIBLE_DEVICES=0
JOBID=${HOSTNAME#j-}
JOBID=${JOBID%-job-*}
RANK=${HOSTNAME##*-}
ENDPOINT=$1
shift
export GPUS=$1
shift
export NNODES=$1
shift
OLMO_SHARED_FS=1
echo "HOSTNAME=$HOSTNAME, NNODES=$NNODES, GPUS=$GPUS, JOBID=$JOBID, RANK=$RANK, ENDPOINT=$ENDPOINT, OLMO_SHARED_FS=$OLMO_SHARED_FS"
. ~/miniconda3/etc/profile.d/conda.sh
conda activate flexdemo
conda info
cd /work/flexdemo/DeToNATION/benchmarks/t5/
pwd
torchrun --nnodes=$NNODES --nproc_per_node=$GPUS --rdzv-id=$JOBID --rdzv-backend=c10d --rdzv-endpoint=$ENDPOINT --node_rank=$RANK train.py $@
echo "=== DONE OUT ==="
echo "=== DONE ERR ===" >&2
