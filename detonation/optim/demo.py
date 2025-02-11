"""DeMo: Decoupled Momentum Optimization

This implements DeMo replication for DeToNATION.
"""

from einops import rearrange
import math
import torch
import torch.fft
import torch.distributed as dist
from typing import Any, Dict, Optional

from . import DeToNATION
from ..repl import DeMoReplicator

__all__ = ["DeMo"]

class DeMo(DeToNATION):

    def __init__(
        self,
        params,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        sign: bool = True,
        sharding_parallel_group: Optional[dist.ProcessGroup] = None,
        replication_parallel_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        super().__init__(
            params,
            weight_decay=weight_decay,
            sign=sign,
            sharding_parallel_group=sharding_parallel_group,
            replication_parallel_group=replication_parallel_group,
            replicator=DeMoReplicator(
                compression_decay=compression_decay,
                compression_topk=compression_topk,
                compression_chunk=compression_chunk,
            ),
            **kwargs,
        )
