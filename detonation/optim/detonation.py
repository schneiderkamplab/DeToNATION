"""DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes

This implements the DeToNATION optimizer for DDP, FSDP, and HSDP. For gradient synchronization, it offers
a number of approaches:
* no gradient replication
* Decouplee Momentum (DeMo) repliation - https://arxiv.org/abs/2106.11447 with the implementation
  adapted from https://github.com/bloc97/DeMo
* full gradient replication
In an exisiting codebase that uses PyTorch and one of these sharding strategies, wrap your forward-backward in 
`torch.distributed.DistributedDataParallel.no_sync` to disable external gradient synchronization.
See https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
TODO:
* grab replication_parallel_group at init time
* make sharding strategy a parameter
* automatically detect sharding strategy from model
* automatically detect sharding and replication groups from torch.dist or model
* refactor Replicator and Full/No to own file
"""

from abc import ABC, abstractmethod
import torch
import torch.fft
import torch.distributed as dist
from typing import Any, Callable, Dict, Optional

__all__ = ["DeToNATION", "Replicator"]

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
        sharded_grad: torch.Tensor,
        replication_parallel_group: dist.ProcessGroup,
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

class FullReplicator(Replicator):

    def init(self, optim: torch.optim.Optimizer):
        pass

    def step(self):
        pass

    def replicate(
        sharded_grad: torch.Tensor,
        replication_parallel_group: dist.ProcessGroup,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        dist.all_reduce(sharded_grad, dist.ReduceOp.AVG, group=replication_parallel_group)
        return sharded_grad

class NoReplicator(Replicator):

    def init(self, optim: torch.optim.Optimizer):
        pass

    def step(self):
        pass

    def replicate(
        sharded_grad: torch.Tensor,
        replication_parallel_group: dist.ProcessGroup,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        return sharded_grad

class DeToNATION(torch.optim.SGD):
    def __init__(
        self,
        params,
        weight_decay: float = 0.0,
        sign: bool = True,
        sharding_parallel_group: Optional[dist.ProcessGroup] = None,
        replication_parallel_group: Optional[dist.ProcessGroup] = None,
        replicator: Replicator = FullReplicator(),
        **kwargs,
    ):
        super().__init__(
            params,
            foreach=False,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            weight_decay=0.0,
            **kwargs,
        )

        self.weight_decay = weight_decay
        self.sign = sign
        self.sharding_parallel_group = sharding_parallel_group # intra-node communication 
        self.replication_parallel_group = replication_parallel_group # inter-node communication
        self.replicator = replicator

        self._sharding_world_size = dist.get_world_size(self.sharding_parallel_group)
        self._replication_world_size = dist.get_world_size(self.replication_parallel_group)

        if self._sharding_world_size == 0:
            raise ValueError("Sharding world size cannot be zero")
        if self._replication_world_size == 0:
            raise ValueError("Replication world size cannot be zero")

        self.replicator.init(self)

    def _grad_reduce_scatter(self, grad: torch.Tensor):
        # Do not reduce_scatter if the gradient is not sharded
        if self._sharding_world_size == 1:
            return grad

        # Chunk and pad the unsharded gradient
        chunks = list(grad.chunk(self._sharding_world_size))
        numel_to_pad = self._sharding_world_size * chunks[0].numel() - grad.numel()
        padded_unsharded_grad = F.pad(grad, [0, numel_to_pad]) if numel_to_pad > 0 else grad

        # Prepare and scatter the sharded gradient
        sharded_grad = torch.empty_like(chunks[0])
        dist.reduce_scatter_tensor(
            sharded_grad,
            padded_unsharded_grad,
            op=dist.ReduceOp.AVG,
            group=self.sharding_parallel_group,
        )
        return sharded_grad

    @torch.no_grad()
    def step(self, closure: Callable | None = None):

        # Any step-wise initialization needed by the replicator
        self.replicator.step()

        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if not param.requires_grad:
                    continue

                # Sharding gradient if needed
                unsharded_grad = param.grad.data
                param.grad = None
                sharded_grad = self._grad_reduce_scatter(unsharded_grad)

                # Step-Weight decay
                if self.weight_decay != 0.0:
                    param.data.mul_(1.0 - lr * self.weight_decay)

                # Replicating the gradient if needed
                new_grad = self.replicator.replicate(
                    sharded_grad=sharded_grad,
                    replication_parallel_group=self.replication_parallel_group,
                    param=param,
                    param_state_dict=self.state[param],
                    param_group=group,
                )
                param.grad = new_grad

                # Sign-SGD
                if self.sign:
                    param.grad.sign_()

        # SGD step
        return super().step(closure)
