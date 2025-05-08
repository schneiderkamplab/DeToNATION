from mltiming import timing
import torch
import torch.distributed as dist
import torch.fft
from typing import Dict, Any

from .replicator import Replicator
from ..util import DCTCompress, DCTTransform

__all__ = ["DeMoReplicator"]

class DeMoReplicator(Replicator):

    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
    ):
        self.compression_decay = compression_decay
        self.compression_chunk = compression_chunk
        self.compression_topk = compression_topk

        if self.compression_topk <= 0:
            raise ValueError("topk_size has to be positive")
        if self.compression_chunk <= 0:
            raise ValueError("chunk_size has to be positive")
        if self.compression_decay < 0:
            raise ValueError("Negative compression_decay is currently not supported")
        if self.compression_decay >= 1:
            raise ValueError("Values of compression_decay bigger or equal to 1.0 is currently not supported")

    def init(
            self,
            optim: torch.optim.Optimizer,
            replication_parallel_group: dist.ProcessGroup | None = None,
        ):
        for group in optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    optim.state[p]["demo_delta"] = torch.zeros_like(p)
        self.transform = DCTTransform(optim.param_groups, self.compression_chunk)
        print('Actual chunk size in DeMo replication:', self.transform.shape_dict)    # Print actual chunk sizes
        self.replication_parallel_group = optim.replication_parallel_group if replication_parallel_group is None else replication_parallel_group
        self._replication_world_size = self.replication_parallel_group.size()
        self.data_transmitted = []
        self.data_received = []

    def pre_step(self):
        self.data_transmit = 0
        self.data_receive = 0

    def post_step(self):
        self.data_transmitted.append(self.data_transmit)
        self.data_received.append(self.data_receive)

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        # Decay delta and add the gradient
        delta = param_state_dict["demo_delta"]
        if self.compression_decay != 1:
            delta.mul_(self.compression_decay)
        delta.add_(sharded_grad, alpha=param_group["lr"])

        # Replicating delta only if needed
        if self._replication_world_size == 1:
            new_grad = delta.clone()
            delta.zero_()
            return new_grad

        # Compress delta
        sparse_idx, sparse_val, xshape = DCTCompress.compress(
            self.transform.encode(delta), self.compression_topk
        )
        sparse_idx = sparse_idx.to(torch.int32)

        # Estimate transmitted delta
        transmit_grad = self.transform.decode(
            DCTCompress.decompress(sparse_idx.to(torch.int64), sparse_val, xshape, param.device, param.dtype)
        )

        # Remove transmitted from delta
        delta.sub_(transmit_grad)

        # Prepare and gather the indices and values
        sparse_idx_gather = [torch.zeros_like(sparse_idx) for _ in range(self._replication_world_size)]
        sparse_val_gather = [torch.zeros_like(sparse_val) for _ in range(self._replication_world_size)]
        sparse_idx_handle = dist.all_gather(sparse_idx_gather, sparse_idx, group=self.replication_parallel_group, async_op=True)
        sparse_val_handle = dist.all_gather(sparse_val_gather, sparse_val, group=self.replication_parallel_group, async_op=True)
        sparse_idx_handle.wait()
        sparse_val_handle.wait()

        # Log I/O data size
        self.data_transmit += sparse_idx.nbytes + sparse_val.nbytes
        for si, v in zip(sparse_idx_gather, sparse_val_gather):
            self.data_receive += si.nbytes + v.nbytes

        # Decode new gradient from all nodes
        sparse_idx_gather = [x.to(torch.int64) for x in sparse_idx_gather]
        new_grad = self.transform.decode(
            DCTCompress.batch_decompress(sparse_idx_gather, sparse_val_gather, xshape, param.device, param.dtype)
        )
        return new_grad
