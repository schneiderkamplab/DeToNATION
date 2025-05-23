import torch
import torch.distributed as dist
from torch.optim import SGD
from typing import Callable, List

from ..detonatiomixin import DeToNATIONMixin
from ..repl import DeMoReplicator, Replicator

__all__ = ["DeToSGD"]

class DeToSGD(SGD, DeToNATIONMixin):

    def __init__(
        self,
        params,
        detonation_weight_decay: float = 0.0,
        detonation_sign: bool = True,
        sharding_parallel_group: dist.ProcessGroup | None = None,
        replication_parallel_group: dist.ProcessGroup | List[dist.ProcessGroup] | None = None,
        replicator: Replicator | List[Replicator] = DeMoReplicator(),
        replicate_every: int | List[int] = 1,
        skip_every: int | List[int] | None = None,
        *args,
        **kwargs,
    ):
        SGD.__init__(
            self,
            params,
            *args,
            **kwargs,
        )
        DeToNATIONMixin.__init__(
            self,
            detonation_weight_decay=detonation_weight_decay,
            detonation_sign=detonation_sign,
            sharding_parallel_group=sharding_parallel_group,
            replication_parallel_group=replication_parallel_group,
            replicator=replicator,
            replicate_every=replicate_every,
            skip_every=skip_every,
        )

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        return DeToNATIONMixin.step(self, closure=closure, base_step=SGD.step)
