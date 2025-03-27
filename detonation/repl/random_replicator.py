import torch
import torch.distributed as dist
import torch.fft

from typing import Dict, Any

from mltiming import timing

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
        self.max_size = max(p.size(0) for group in optim.param_groups for p in group["params"] if p.requires_grad)
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
        self.permutation = torch.randperm(self.max_size, generator=self.random_state, device=self.random_state.device)

    def post_step(self):
        self.data_transmitted.append(self.data_transmit)
        self.data_received.append(self.data_receive)

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
        step_metrics: dict = {},
    ) -> torch.Tensor:
        # No replication if compression rate is 0
        with timing(dict=step_metrics, key="train/optim/replicate/noreplication"):
            if self.compression_rate == 0:
                return sharded_grad.to(device=param.device, dtype=param.dtype)
            dist.barrier()

        # Decay delta and add the gradient
        with timing(dict=step_metrics, key="train/optim/replicate/delta"):
            delta = param_state_dict["demo_delta"]
            if self.compression_decay != 1:
                delta.mul_(self.compression_decay)
            delta.add_(sharded_grad, alpha=param_group["lr"])
            dist.barrier()

        # Replicating delta only if needed
        with timing(dict=step_metrics, key="train/optim/replicate/noreplication"):
            if self._replication_world_size == 1 or self.compression_rate == 1:
                new_grad = delta.clone()
                delta.zero_()
                return new_grad
            dist.barrier()
    
        # Compress delta
        with timing(dict=step_metrics, key="train/optim/replicate/encode"):
            num_selected = int(delta.size(0) * self.compression_rate)
            # rand_scores = torch.rand(delta.size(0), generator=self.random_state, device=param.device)
            # _, selected_rows = torch.topk(rand_scores, num_selected, largest=False)
            if dist.get_rank() == 0:
                print(f"num_selected: {num_selected}")
                print(f"max_size: {self.max_size}")
                print(f"permutation size: {self.permutation.size()}")
                print(f"delta size: {delta.size()}")
                print(f"param size: {param.size()}")
                print(f"sharded_grad size: {sharded_grad.size()}")
                1/0
            selected_rows = self.permutation[:num_selected]
            compressed_grad = delta[selected_rows]
            dist.barrier()

        # Remove compressed gradient from delta
        with timing(dict=step_metrics, key="train/optim/replicate/remove"):
            mask = torch.zeros(delta.size(0), dtype=torch.bool, device=param.device)
            mask[selected_rows] = True
            delta.mul_(~mask.unsqueeze(1) if delta.dim() > 1 else ~mask)
            dist.barrier()

        # Average the compressed gradient
        with timing(dict=step_metrics, key="train/optim/replicate/communicate"):
            dist.all_reduce(compressed_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group)
            dist.barrier()

        # Log I/O data size
        with timing(dict=step_metrics, key="train/optim/replicate/calculateio"):
            self.data_transmit += compressed_grad.nbytes
            self.data_receive += compressed_grad.nbytes
            dist.barrier()

        # Decode new gradient from all nodes
        with timing(dict=step_metrics, key="train/optim/replicate/decode"):
            new_grad = torch.zeros_like(delta, device=param.device)
            new_grad[selected_rows] = compressed_grad
            dist.barrier()
        return new_grad
