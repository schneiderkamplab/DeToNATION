"""DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes

This implements the DeToNATION optimizer that works in DDP, FSDP, and HSDP settings. For gradient
synchronization in DDP and HSDP settings, it offers a number of approaches:
* no gradient replication
* Decoupled Momentum (DeMo) repliation - https://arxiv.org/abs/2106.11447 with the implementation based on https://github.com/bloc97/DeMo
* full gradient replication
In an exisiting codebase that uses PyTorch and one of these sharding strategies, wrap your forward-backward in 
`torch.distributed.DistributedDataParallel.no_sync` to disable external gradient synchronization.
See https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
TODO:
* auto-detect policy if none is given in prepare_detonation according to the model's structure
* compute transmitted data for full and no replication (and make that a part of the DeToNATION?)
* profiling of GPU, CPU, network, and wall time usage (optimize DeMoReplicator? stream replications instead of waiting?)
"""

import torch
import torch.distributed as dist
import torch.fft
import torch.nn.functional as F
from typing import Callable, Optional

from ..repl import DeMoReplicator, Replicator

__all__ = ["DeToNATION"]

class DeToNATION(torch.optim.SGD):

    def __init__(
        self,
        params,
        weight_decay: float = 0.0,
        sign: bool = True,
        sharding_parallel_group: Optional[dist.ProcessGroup] = None,
        replication_parallel_group: Optional[dist.ProcessGroup] = None,
        replicator: Replicator = DeMoReplicator(),
        replicate_every: int = 1,
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
        self.replicate_every = replicate_every

        self._sharding_world_size = dist.get_world_size(self.sharding_parallel_group)
        self._replication_world_size = dist.get_world_size(self.replication_parallel_group)

        if self._sharding_world_size == 0:
            raise ValueError("Sharding world size cannot be zero")
        if self._replication_world_size == 0:
            raise ValueError("Replication world size cannot be zero")

        self.state["detonation_step"] = 0
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
        self.state["detonation_step"] += 1

        # Any step-wise initialization needed by the replicator
        self.replicator.pre_step()

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
                if self.state["detonation_step"] % self.replicate_every == 0:
                    new_grad = self.replicator.replicate(
                        sharded_grad=sharded_grad,
                        param=param,
                        param_state_dict=self.state[param],
                        param_group=group,
                    )
                else:
                    new_grad = sharded_grad.to(param.device).to(param.dtype)
                param.grad = new_grad

                # Sign-SGD
                if self.sign:
                    param.grad.sign_()

        # SGD step
        result = super().step(closure)

        # Any step-wise finalization needed by the replicator
        self.replicator.post_step()

        return result
