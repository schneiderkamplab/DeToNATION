"""DeMo: Decoupled Momentum Optimization

This implements DeMo replication for DeToNATION.
"""

from .DeToSGD import DeToSGD
from ..repl import DeMoReplicator

__all__ = ["DeMo"]

class DeMo(DeToSGD):

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
