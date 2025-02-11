# Decoupled Torch Network-Aware Training on Interlinked Online Nodes (DeToNATION)

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

## Usage
The usage requires three elements as exemplified below for using the FlexDeMo optimizer.

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
