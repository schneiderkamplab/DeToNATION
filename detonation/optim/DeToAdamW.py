import torch
import torch.distributed as dist
import torch.fft
from torch.optim import AdamW
from torch.optim.optimizer import _device_dtype_check_for_fused, _get_scalar_dtype
from typing import Callable, List

from ..detonatiomixin import DeToNATIONMixin
from ..repl import DeMoReplicator, Replicator

__all__ = ["DeToAdamW"]

class DeToAdamW(AdamW, DeToNATIONMixin):
    
    def __init__(
        self,
        params,
        detonation_weight_decay: float = 0.0,
        detonation_sign: bool = True,
        sharding_parallel_group: dist.ProcessGroup | None = None,
        replication_parallel_group: dist.ProcessGroup | List[dist.ProcessGroup] | None = None,
        replicator: Replicator | List[Replicator] = DeMoReplicator(),
        replicate_every: int | List[int] = 1,
        skip_every: int | List[int] | None = None,
        *args,
        **kwargs,
    ):
        # Initialize AdamW base class
        super().__init__(params, *args, **kwargs)
        for group in self.param_groups:
            amsgrad = group['amsgrad']
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["demo_delta"] = torch.zeros_like(p)

                state = self.state[p]
                if group["fused"]:
                    _device_dtype_check_for_fused(p)
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

        DeToNATIONMixin.__init__(
            self,
            detonation_weight_decay=detonation_weight_decay,
            detonation_sign=detonation_sign,
            sharding_parallel_group=sharding_parallel_group,
            replication_parallel_group=replication_parallel_group,
            replicator=replicator,
            replicate_every=replicate_every,
            skip_every=skip_every,
        )

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        return DeToNATIONMixin.step(self, closure=closure, base_step=AdamW.step)
