import torch
import torch.distributed as dist
from typing import Any, Dict

from .replicator import Replicator

__all__ = ["FullReplicator"]

class FullReplicator(Replicator):

    def init(
            self,
            optim: torch.optim.Optimizer,
            replication_parallel_group: dist.ProcessGroup | None = None,
        ):
        self.replication_parallel_group = optim.replication_parallel_group if replication_parallel_group is None else replication_parallel_group
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
        dist.all_reduce(sharded_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group)
        self.data_receive += sharded_grad.nbytes
        self.data_transmit += sharded_grad.nbytes
        return sharded_grad.to(device=param.device, dtype=param.dtype)
