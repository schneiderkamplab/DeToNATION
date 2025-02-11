import torch
from typing import Any, Dict

from . import Replicator

__all__ = ["NoReplicator"]

class NoReplicator(Replicator):

    def init(self, optim: torch.optim.Optimizer):
        pass

    def step(self):
        pass

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        return sharded_grad.to(param.device).to(param.dtype)
