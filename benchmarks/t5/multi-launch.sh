#!/bin/bash
HOSTNAME=$(hostname)
HOSTIP=$(hostname -I)
_NNODES=$(jq '.request.replicas' /work/JobParameters.json)
NNODES=${NNODES:-$_NNODES}
_GPUS=$(jq '.request.product.id' /work/JobParameters.json)
_GPUS=${_GPUS##*-}
_GPUS=${_GPUS%\"}
GPUS=${GPUS:-$_GPUS}
JOBID=${HOSTNAME#j-}
JOBID=${JOBID%-job-*}
CWD=$(pwd)
for RANK in $(seq $((NNODES-1)) -1 0)
do
    TARGET=j-${JOBID}-job-${RANK}
    LOGOUT=$CWD/$TARGET.out
    LOGERR=$CWD/$TARGET.err
    PID=$(ssh $TARGET nohup /work/flexdemo/DeToNATION/benchmarks/t5/start.sh $HOSTIP $GPUS $NNODES $@ > $LOGOUT 2> $LOGERR & echo $!)
done

echo $PID