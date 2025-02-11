import torch
from typing import Any, Dict

from .replicator import Replicator

__all__ = ["NoReplicator"]

class NoReplicator(Replicator):

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        return sharded_grad.to(param.device).to(param.dtype)
