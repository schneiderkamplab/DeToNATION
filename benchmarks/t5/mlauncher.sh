#!/bin/bash

# Calc params
MODELS=('google-t5/t5-large' 'google-t5/t5-base' 'google-t5/t5-large')
TOPKS=( 1 2 4 8 16 32 )
CHUNKS=( 32 64 128 )

for MODEL in ${MODELS[@]}; do
    for CHUNK in ${CHUNKS[@]}; do
        for TOPK in ${TOPKS[@]}; do
            PID=$(/work/flexdemo/DeToNATION/benchmarks/t5/multi-launch.sh --optim=deto-demo --model=$MODEL  --compression-topk=$TOPK --compression-chunk=$CHUNK --batch-size=8)
            echo $(date +"%Y-%m-%d %H:%M:%S"): Launched training with PID: $PID and params: optim: deto-demo, model: $MODEL, compression-topk: $TOPK, compression-chunk: $CHUNK
            while ps -p $PID > /dev/null; do sleep 5; done;
            echo $(date +"%Y-%m-%d %H:%M:%S"): Finished process $PID.
            echo Wating 10s before starting next...
            sleep 10
        done
    done
done
