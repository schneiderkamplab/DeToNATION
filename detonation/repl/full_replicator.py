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

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        dist.all_reduce(sharded_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group)
        return sharded_grad.to(param.device).to(param.dtype)
