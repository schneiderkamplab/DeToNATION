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
        self.random_state = torch.Generator().manual_seed(self.seed)
        for group in optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    optim.state[p]["demo_delta"] = torch.zeros_like(p)
        self.replication_parallel_group = optim.replication_parallel_group if replication_parallel_group is None else replication_parallel_group
        self._replication_world_size = self.replication_parallel_group.size()
        self.data_transmitted = []
        self.data_received = []

    def pre_step(self):
        self.data_transmit = 0
        self.data_receive = 0

    def post_step(self):
        self.data_transmitted.append(self.data_transmit)
        self.data_received.append(self.data_receive)

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        # No replication if compression rate is 0
        if self.compression_rate == 0:
            return sharded_grad.to(param.device).to(param.dtype)

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
        selected_rows = torch.randperm(delta.size(0), generator=self.random_state)[:int(self.compression_rate * delta.size(0))]
        compressed_grad = delta[selected_rows]

        # Estimate transmitted delta
        transmit_grad = torch.zeros_like(delta)
        transmit_grad[selected_rows] = compressed_grad

        # Remove transmitted from delta
        delta.sub_(transmit_grad)

        # Average the compressed gradient
        dist.all_reduce(compressed_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group)

        # Log I/O data size
        self.data_transmit += compressed_grad.nbytes
        self.data_receive += compressed_grad.nbytes

        # Decode new gradient from all nodes
        new_grad = torch.zeros_like(delta)
        new_grad[selected_rows] = compressed_grad
        return new_grad
