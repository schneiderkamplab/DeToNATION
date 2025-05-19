import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from typing import Optional, Tuple

from .optim import DeToSGD, DeToAdamW, Optimizers
from .repl import Replicator, DeMoReplicator

__all__ = ["prepare_detonation"]

def prepare_detonation(
    model: torch.nn.Module,
    optimizer: Optimizers,
    replicator: Optional[Replicator] = DeMoReplicator(),
    sharding_group_size: Optional[int] = None,
    replication_group_size: Optional[int] = None,
    sharding_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    replication_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    fsdp_kwargs: dict = {},
    **detonation_kwargs,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    if (sharding_parallel_group is None) ^ (replication_parallel_group is None):
        raise ValueError("Cannot specify only one of replication_parallel_group and sharding_parallel_group")
    if sharding_parallel_group is not None:
        if replication_group_size is not None or sharding_group_size is not None:
            raise ValueError("Cannot specify both group sizes and parallel groups")
        sharding_group_size = sharding_parallel_group.size()
        replication_group_size = replication_parallel_group.size()
    else:
        if sharding_group_size is None:
            if replication_group_size is None:
                sharding_group_size = local_world_size
                replication_group_size = world_size // local_world_size
            else:
                sharding_group_size = world_size // replication_group_size
        elif replication_group_size is None:
            replication_group_size = world_size // sharding_group_size
        mesh_2d = init_device_mesh(
            device_type="cuda",
            mesh_shape=(replication_group_size, sharding_group_size),
        )
        sharding_parallel_group = mesh_2d.get_group(1)
        replication_parallel_group = mesh_2d.get_group(0)
    assert world_size == sharding_group_size * replication_group_size
    assert local_world_size % sharding_group_size == 0
    model = FSDP(
        model,
        process_group=(sharding_parallel_group, replication_parallel_group),
        device_id=int(os.environ['LOCAL_RANK']),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        **fsdp_kwargs,
    )
    match optimizer:
        case Optimizers.SGD:
            optim = DeToSGD(
                model.parameters(),
                replicator=replicator,
                sharding_parallel_group=sharding_parallel_group,
                replication_parallel_group=replication_parallel_group,
                **detonation_kwargs,
            )
        case Optimizers.AdamW:
            optim = DeToAdamW(
                model.parameters(),
                replicator=replicator,
                sharding_parallel_group=sharding_parallel_group,
                replication_parallel_group=replication_parallel_group,
                **detonation_kwargs,
            )
        case _:
            raise Exception("Optimizer not supported.")

    return model, optim
