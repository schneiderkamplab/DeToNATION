import torch
import torch.distributed as dist
from typing import Any, Dict

from . import Replicator

__all__ = ["FullReplicator"]

class FullReplicator(Replicator):

    def init(self, optim: torch.optim.Optimizer):
        self.replication_parallel_group = optim.replication_parallel_group

    def step(self):
        pass

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        dist.all_reduce(sharded_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group)
        return sharded_grad.to(param.device).to(param.dtype)
