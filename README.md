# Decoupled Torch Network-Aware Training on Interlinked Online Nodes (DeToNATION)

This code currently implements the results described in [FlexDeMo: Decoupled Momentum Optimization for Fully and Hybrid Sharded Training](https://arxiv.org/abs/2502.06728). An implementation to run all experiments from the paper is found in the benchmarks folder.

## Installation
Installation from PyPI:
```
pip install detonation
```

Installation from source:
```
git clone https://github.com/schneiderkamplab/DeToNATION
cd DeToNATION
pip install .
```

## Example
There is a a full example for language model training using FlexDeMo in the example folder. Please refer to the documentation:
```
examples/t5/README.md
```
This example demonstrates the use of the `prepare_detonation` function for obtaining a distributed model and optimizer.

## Benchmarks
There is a a full benchmarking example for language model training using FlexDeMo in the benchmarks folder. Please refer to the documentation:
```
benchmarks/t5/README.md
```
This benchmarking example demonstrates the use of the `prepare_detonation` function for obtaining a distributed model and optimizer, and uses aim and mltiming to track model parameters and performance.

## Usage
The direct usage of DeToNATION without using `prepare_detonation` requires three elements as exemplified below for the FlexDeMo optimizer, i.e., DeToNATION with node-based hybrid sharding using DeMo replication.

First, you need to wrap your model with FSDP and the hybrid sharding strategy:
```
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
)
```

Then, you can import and instantiate the FlexDeMo optimizer:
```
from detonation import DeMo
optim = DeMo(
    compression_topk=16,
    compression_chunk=128,
    sharding_parallel_group=model.process_group,
    replication_parallel_group=model._inter_node_pg,
)
```

Third and last, you need to wrap the forward and backward pass using a
`no_sync` context manager to avoid automatic full gradient synchronization:
```
    with model.no_sync(): # Disable gradient synchronizations across FSDP instances.
        loss = model(input_ids=batch["input_ids"],labels=batch["labels"])["loss"]
        loss.backward()
```
