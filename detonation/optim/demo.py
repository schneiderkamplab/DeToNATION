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
        **kwargs,
    ):
        super().__init__(
            params,
            replicator=DeMoReplicator(
                compression_decay=compression_decay,
                compression_topk=compression_topk,
                compression_chunk=compression_chunk,
            ),
            **kwargs,
        )
