from mltiming import timing
import torch
import torch.distributed as dist
import torch.fft
from typing import Dict, Any

from .replicator import Replicator

__all__ = ["RandomReplicator"]

class RandomReplicator(Replicator):

    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_rate: float = 0.1,
        seed: int = 42,
    ):
        self.compression_decay = compression_decay
        self.compression_rate = compression_rate
        self.seed = seed

        if self.compression_rate <= 0:
            raise ValueError("sample_factor has to be positive")
        if self.compression_rate > 1:
            raise ValueError("Values of compression_rate greater than 1.0 are not supported")

    def init(
            self,
            optim: torch.optim.Optimizer,
            replication_parallel_group: dist.ProcessGroup | None = None,
        ):
        device = optim.param_groups[0]["params"][0].device
        self.random_state = torch.Generator(device=device).manual_seed(self.seed)
        self.sizes = set()
        for group in optim.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                self.sizes.add(p.size(0))
                optim.state[p]["demo_delta"] = torch.zeros_like(p)
        self.replication_parallel_group = optim.replication_parallel_group if replication_parallel_group is None else replication_parallel_group
        self._replication_world_size = self.replication_parallel_group.size()
        self.data_transmitted = []
        self.data_received = []
        self
        self.val_queue = {}

    def pre_step(self):
        self.data_transmit = 0
        self.data_receive = 0
        rand_score = torch.rand(max(self.sizes), generator=self.random_state, device=self.random_state.device)
        self.permutations = {size: torch.topk(rand_score[:size], int(self.compression_rate * size), largest=False).indices for size in self.sizes}

    def post_step(self):
        self.data_transmitted.append(self.data_transmit)
        self.data_received.append(self.data_receive)

    def post_communication(self, compressed_grad, param, selected_rows):
        # Average the compressed gradient
        handle = dist.all_reduce(compressed_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group, async_op=True)
        self.val_queue[param] = {"buffer": compressed_grad, "handle": handle, "indices": selected_rows}

        # Log I/O data size
        self.data_transmit += compressed_grad.nbytes
        self.data_receive += compressed_grad.nbytes
    
    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        # No replication if compression rate is 0
        if self.compression_rate == 0:
            return sharded_grad.to(device=param.device, dtype=param.dtype)

        # Decay delta and add the gradient
        delta = param_state_dict["demo_delta"]
        if self.compression_decay != 1:
            delta.mul_(self.compression_decay)
        delta.add_(sharded_grad, alpha=param_group["lr"])

        # Replicating delta only if needed
        if self._replication_world_size == 1 or self.compression_rate == 1:
            new_grad = delta.clone()
            delta.zero_()
            return new_grad

        # Compress delta
        selected_rows = self.permutations[delta.size(0)]
        compressed_grad = delta[selected_rows]

        # Remove compressed gradient from delta
        mask = torch.zeros(delta.size(0), dtype=torch.bool, device=param.device)
        mask[selected_rows] = True
        delta.mul_(~mask.unsqueeze(1) if delta.dim() > 1 else ~mask)

        if param in self.val_queue:
            self.val_queue[param]["handle"].wait() # should be no-op guard

            # Decode new gradient from all nodes
            new_grad = torch.zeros_like(delta, device=param.device)
            idx = self.val_queue[param]["indices"]
            new_grad[idx] = self.val_queue[param]["buffer"] # now grab from self.val_queue
            new_grad = new_grad.view_as(sharded_grad)
            
            # remove from queue
            del self.val_queue[param]
            
            # post communication from this step
            self.post_communication(compressed_grad, param, selected_rows)
            return new_grad
        else: 
            # post communication new gradient
            self.post_communication(compressed_grad, param, selected_rows)
            return None # no grad available yet
