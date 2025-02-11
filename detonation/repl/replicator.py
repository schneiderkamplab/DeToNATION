from abc import ABC, abstractmethod
import torch
from typing import Any, Dict

__all__ = ["Replicator"]

class Replicator(ABC):
    """
    This provides the functions to initialize and execute replicators to be used by the
    :class:`DeToNATION` optimizer for replicating gradients across replication groups (typically nodes).
    """

    @abstractmethod
    def init(self, optim: torch.optim.Optimizer):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ):
        """
        Replicate the gradient across the replication group.

        Args:
            sharded_grad (torch.Tensor): The sharded gradient tensor.
            replication_parallel_group (dist.ProcessGroup): The replication parallel group.
            param_state_dict (dict): The state dictionary of the parameter.
            param_group (Dict[str, Any]): The parameter group.

        Returns:
            torch.Tensor: The replicated gradient tensor.
        """
        pass
