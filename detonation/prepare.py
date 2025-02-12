import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from typing import Optional, Tuple

from .optim import DeToNATION
from .repl import Replicator

__all__ = ["prepare_detonation"]

def prepare_detonation(
    model: torch.nn.Module,
    replicator: Replicator,
    sharding_group_size: Optional[int] = None,
    replication_group_size: Optional[int] = None,
    detonation_kwargs: dict = {},
    **fsdp_kwargs,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    if sharding_group_size is None:
        if replication_group_size is None:
            sharding_group_size = local_world_size
            replication_group_size = world_size // local_world_size
        else:
            sharding_group_size = world_size // replication_group_size
    elif replication_group_size is None:
        replication_group_size = world_size // sharding_group_size
    assert world_size == sharding_group_size * replication_group_size
    assert local_world_size % sharding_group_size == 0
    mesh_2d = init_device_mesh(
        device_type="cuda",
        mesh_shape=(sharding_group_size, replication_group_size),
    )
    model = FSDP(
        model,
        device_mesh=mesh_2d,
        device_id=int(os.environ['LOCAL_RANK']),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        **fsdp_kwargs,
    )
    optim = DeToNATION(
        model.parameters(),
        replicator=replicator,
        sharding_parallel_group=model.process_group,
        replication_parallel_group=model._inter_node_pg,
        **detonation_kwargs,
    )
    return model, optim
