import torch
import torch.distributed as dist
from typing import Any, Dict

from .replicator import Replicator

__all__ = ["FullReplicator"]

class FullReplicator(Replicator):

    def init(
            self,
            optim: torch.optim.Optimizer,
            replication_parallel_group: dist.ProcessGroup | None = None,
        ):
        self.replication_parallel_group = optim.replication_parallel_group if replication_parallel_group is None else replication_parallel_group
        self.data_transmitted = []
        self.data_received = []
        self.val_queue = {}

    def pre_step(self):
        self.data_transmit = 0
        self.data_receive = 0

    def post_step(self):
        self.data_transmitted.append(self.data_transmit)
        self.data_received.append(self.data_receive)

    def post_communication(self, grad, param):
        # Average the full gradient
        handle = dist.all_reduce(grad, dist.ReduceOp.AVG, group=self.replication_parallel_group, async_op=True)
        self.val_queue[param] = {"buffer": grad, "handle": handle}

        # Log I/O data size
        self.data_transmit += grad.nbytes
        self.data_receive += grad.nbytes

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        param: torch.nn.Parameter,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        
        if param in self.val_queue:
            self.val_queue[param]["handle"].wait() # should be no-op guard
            new_grad = self.val_queue[param]["buffer"].to(device=param.device, dtype=param.dtype)
        
            # remove from queue
            del self.val_queue[param]
            
            # post communication from this step
            self.post_communication(sharded_grad, param)
            return new_grad
        else:
            self.post_communication(sharded_grad, param)
            return None # no grad available yet
