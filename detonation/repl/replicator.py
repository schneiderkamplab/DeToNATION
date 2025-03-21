from abc import ABC, abstractmethod
import torch
from typing import Any, Dict

__all__ = ["Replicator"]

class Replicator(ABC):
    """
    This provides the functions to initialize and execute replicators to be used by the
    :class:`DeToNATION` optimizer for replicating gradients across replication groups (typically nodes).
    """

    def init(
        self,
        optim: torch.optim.Optimizer,
        replication_parallel_group: torch.distributed.ProcessGroup | None = None,
    ):
        """
        Initialize the replicator.
        """
        pass

    def pre_step(self):
        """
        Pre-step hook.
        """
        pass

    def post_step(self):
        """
        Post-step hook.
        """
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
            param (torch.nn.Parameter): The current parameter.
            param_state_dict (dict): The state dictionary of the parameter.
            param_group (Dict[str, Any]): The current parameter group.

        Returns:
            torch.Tensor: The replicated gradient tensor.
        """
        pass
