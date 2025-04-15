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
        compression_chunk: int = 64,
        seed: int = 42,
    ):
        self.compression_decay = compression_decay
        self.compression_rate = compression_rate
        self.compression_chunk = compression_chunk
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
                closest_chunk = self.__class__._get_smaller_split(len(p), self.compression_chunk)
                self.sizes.add(p.view(-1, closest_chunk).size(0))
                optim.state[p]["demo_delta"] = torch.zeros_like(p)
        self.replication_parallel_group = optim.replication_parallel_group if replication_parallel_group is None else replication_parallel_group
        self._replication_world_size = self.replication_parallel_group.size()
        self.data_transmitted = []
        self.data_received = []

    def pre_step(self):
        self.data_transmit = 0
        self.data_receive = 0
        rand_score = torch.rand(max(self.sizes), generator=self.random_state, device=self.random_state.device)
        self.permutations = {size: torch.topk(rand_score[:size], int(self.compression_rate * size), largest=False).indices for size in self.sizes}

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
        closest_chunk = self.__class__._get_smaller_split(len(delta), self.compression_chunk)
        delta = delta.view(-1, closest_chunk)
        selected_rows = self.permutations[delta.size(0)]
        compressed_grad = delta[selected_rows]

        # Remove compressed gradient from delta
        mask = torch.zeros(delta.size(0), dtype=torch.bool, device=param.device)
        mask[selected_rows] = True
        delta.mul_(~mask.unsqueeze(1) if delta.dim() > 1 else ~mask)

        # Average the compressed gradient
        dist.all_reduce(compressed_grad, dist.ReduceOp.AVG, group=self.replication_parallel_group)

        # Log I/O data size
        self.data_transmit += compressed_grad.nbytes
        self.data_receive += compressed_grad.nbytes

        # Decode new gradient from all nodes
        new_grad = torch.zeros_like(delta, device=param.device)
        new_grad[selected_rows] = compressed_grad
        new_grad = new_grad.view_as(sharded_grad)
        return new_grad

    def _get_prime_divisors(n):
        divisors = []
        while n % 2 == 0:
            divisors.append(2)
            n //= 2
        while n % 3 == 0:
            divisors.append(3)
            n //= 3
        i = 5
        while i * i <= n:
            for k in (i, i + 2):
                while n % k == 0:
                    divisors.append(k)
                    n //= k
            i += 6
        if n > 1:
            divisors.append(n)
        return divisors

    @classmethod
    def _get_divisors(cls, n):
        divisors = []
        if n == 1:
            divisors.append(1)
        elif n > 1:
            prime_factors = cls._get_prime_divisors(n)
            divisors = [1]
            last_prime = 0
            factor = 0
            slice_len = 0
            # Find all the products that are divisors of n
            for prime in prime_factors:
                if last_prime != prime:
                    slice_len = len(divisors)
                    factor = prime
                else:
                    factor *= prime
                for i in range(slice_len):
                    divisors.append(divisors[i] * factor)
                last_prime = prime
            divisors.sort()
        return divisors

    @classmethod
    def _get_smaller_split(cls, n, close_to):
        all_divisors = cls._get_divisors(n)
        for ix, val in enumerate(all_divisors):
            if val == close_to:
                return val
            if val > close_to:
                if ix == 0:
                    return val
                return all_divisors[ix - 1]
        return n