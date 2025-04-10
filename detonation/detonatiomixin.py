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
import torch.nn.functional as F
from typing import Callable, List

from .repl import DeMoReplicator, Replicator

__all__ = ["DeToNATIONMixin"]

class DeToNATIONMixin():

    def __init__(
        self,
        detonation_weight_decay: float = 0.0,
        detonation_sign: bool = True,
        sharding_parallel_group: dist.ProcessGroup | None = None,
        replication_parallel_group: dist.ProcessGroup | List[dist.ProcessGroup] | None = None,
        replicator: Replicator | List[Replicator] = DeMoReplicator(),
        replicate_every: int | List[int] = 1,
        skip_every: int | List[int] | None = None,
    ):
        self.detonation_weight_decay = detonation_weight_decay
        self.detonation_sign = detonation_sign
        self.sharding_parallel_group = sharding_parallel_group # intra-node communication 
        self.replication_parallel_groups = replication_parallel_group if isinstance(replication_parallel_group, list) else [replication_parallel_group] # inter-node communication
        self.replicators = replicator if isinstance(replicator, list) else [replicator]
        self.replicate_everys = replicate_every if isinstance(replicate_every, list) else [replicate_every]*len(self.replicators)
        self.skip_everys = [None]*len(self.replicators) if skip_every is None else (skip_every if isinstance(skip_every, list) else [skip_every])

        self._sharding_world_size = dist.get_world_size(self.sharding_parallel_group)
        if self._sharding_world_size == 0:
            raise ValueError("Sharding world size cannot be zero")
        for replication_parallel_group in self.replication_parallel_groups:
            if dist.get_world_size(replication_parallel_group) == 0:
                raise ValueError("Replication world size cannot be zero")
        self.state["detonation_step"] = 0
        for replicator, replication_parallel_group in zip(self.replicators, self.replication_parallel_groups):
            replicator.init(self, replication_parallel_group=replication_parallel_group)

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

    def step(self, closure: Callable | None = None, base_step: torch.optim.Optimizer.step = None):
        self.state["detonation_step"] += 1

        # Any step-wise initialization needed by the replicator
        for replicator in self.replicators:
            replicator.pre_step()

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
                if self.detonation_weight_decay != 0.0:
                    param.data.mul_(1.0 - lr * self.weight_decay)

                # Replicating the gradient if needed
                for replicate_every, skip_every, replicator in zip(self.replicate_everys, self.skip_everys, self.replicators):
                    if (
                        (self.state["detonation_step"] % replicate_every == 0) and
                        (skip_every is None or (self.state["detonation_step"] % skip_every != 0))
                    ):
                        new_grad = replicator.replicate(
                            sharded_grad=sharded_grad,
                            param=param,
                            param_state_dict=self.state[param],
                            param_group=group,
                        )
                    else:
                        new_grad = sharded_grad.to(param.device).to(param.dtype)
                param.grad = new_grad

                # Sign-SGD
                if self.detonation_sign:
                    param.grad.sign_()

        # SGD step
        result = base_step(self, closure)

        # Any step-wise finalization needed by the replicator
        for replicator in self.replicators:
            replicator.post_step()

        return result
