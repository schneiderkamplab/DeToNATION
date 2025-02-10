"""DeMo: Decoupled Momentum Optimization

This implements DeMo replication for DeToNATION.
"""

from einops import rearrange
import math
import torch
import torch.fft
import torch.distributed as dist
from typing import Any, Dict, Optional

from .detonation import DeToNATION, Replicator
__all__ = ["DeMo", "DeMoReplicator"]


class DeMoReplicator(Replicator):
    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
    ):
        self.compression_decay = compression_decay
        self.compression_chunk = compression_chunk
        self.compression_topk = compression_topk
        self.compress = CompressDCT()

        if self.compression_topk <= 0:
            raise ValueError("topk_size has to be positive")
        if self.compression_chunk <= 0:
            raise ValueError("chunk_size has to be positive")
        if self.compression_decay < 0:
            raise ValueError("Negative compression_decay is currently not supported")
        if self.compression_decay >= 1:
            raise ValueError("Values of compression_decay bigger or equal to 1.0 is currently not supported")

    def init(self, optim: torch.optim.Optimizer):
        for group in optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    optim.state[p]["demo_delta"] = torch.zeros_like(p)
        self.transform = TransformDCT(optim.param_groups, self.compression_chunk)

    def step(self):
        self.data_transmit = 0
        self.data_receive = 0

    def replicate(
        self,
        sharded_grad: torch.Tensor,
        replication_parallel_group: dist.ProcessGroup,
        param_state_dict: dict,
        param_group: Dict[str, Any],
    ) -> torch.Tensor:
        # Decay delta and add the gradient
        delta = param_state_dict["demo_delta"]
        if self.compression_decay != 1:
            delta.mul_(self.compression_decay)
        delta.add_(sharded_grad, alpha=param_group["lr"])

        # Replicating delta only if needed
        if self._replication_world_size == 1:
            new_grad = delta.clone()
            delta.zero_()
            return new_grad
    
        # Compress delta
        sparse_idx, sparse_val, xshape = self.compress.compress(
            self.transform.encode(delta), self.compression_topk
        )

        # Estimate transmitted delta
        transmit_grad = self.transform.decode(
            self.compress.decompress(sparse_idx, sparse_val, xshape, sharded_grad.device, sharded_grad.dtype)
        )

        # Remove transmitted from delta
        delta.sub_(transmit_grad)

        # Prepare and gather the indices and values
        replication_world_size = replication_parallel_group.size()
        sparse_idx_gather = [torch.zeros_like(sparse_idx) for _ in range(replication_world_size)]
        sparse_val_gather = [torch.zeros_like(sparse_val) for _ in range(replication_world_size)]
        sparse_idx_handle = dist.all_gather(sparse_idx_gather, sparse_idx, group=replication_parallel_group, async_op=True)
        sparse_val_handle = dist.all_gather(sparse_val_gather, sparse_val, group=replication_parallel_group, async_op=True)
        sparse_idx_handle.wait()
        sparse_val_handle.wait()

        # Log I/O data size
        self.data_transmit += sparse_idx.nbytes + sparse_val.nbytes
        for si, v in zip(sparse_idx_gather, sparse_val_gather):
            self.data_receive += si.nbytes + v.nbytes

        # Decode new gradient from all nodes
        new_grad = self.transform.decode(
            self.compress.batch_decompress(sparse_idx_gather, sparse_val_gather, xshape, sharded_grad.device, sharded_grad.dtype)
        )
        return new_grad


class DeMo(DeToNATION):
    def __init__(
        self,
        params,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        sign: bool = True,
        sharding_parallel_group: Optional[dist.ProcessGroup] = None,
        replication_parallel_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        super().__init__(
            params,
            weight_decay=weight_decay,
            sign=sign,
            sharding_parallel_group=sharding_parallel_group,
            replication_parallel_group=replication_parallel_group,
            replicator=DeMoReplicator(
                compression_decay=compression_decay,
                compression_topk=compression_topk,
                compression_chunk=compression_chunk,
            ),
            **kwargs,
        )


class TransformDCT:
    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm="ortho"):
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    # Get the closest smallest divisor to the targeted DCT size
                    sc = _get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc

                    # Pregenerate DCT basis matrices
                    if sc not in self.f_dict:
                        I = torch.eye(sc)
                        self.f_dict[sc] = _dct(I, norm=norm).to(p.dtype).to(p.device)
                        self.b_dict[sc] = _idct(I, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x):
        if len(x.shape) > 2:  # 2D weights
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w

            x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")

        else:  # 1D weights
            n1 = x.shape[1]
            n1w = self.b_dict[n1].to(x.device)
            self.b_dict[n1] = n1w

            x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class CompressDCT:
    @torch.no_grad()
    def __init__(self):
        pass

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @torch.no_grad()
    def compress(self, x, topk):
        xshape = x.shape
        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        topk = self._clamp_topk(x, topk)

        idx = torch.topk(x.abs(), k=topk, dim=-1, largest=True, sorted=False).indices
        val = torch.gather(x, dim=-1, index=idx)

        return idx, val, xshape

    @torch.no_grad()
    def decompress(self, idx, val, xshape, device, dtype):
        x = torch.zeros(xshape, device=device, dtype=dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # TODO: Careful, this is nondeterministic across different CUDA devices! might cause errors to accumulate between nodes!
        x.scatter_reduce_(dim=-1, index=idx, src=val, reduce="mean", include_self=False).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x (h w) -> y x h w", h=xshape[2])

        return x

    @torch.no_grad()
    def batch_decompress(self, idx, val, xshape, device, dtype):
        idx = torch.concatenate(idx, dim=-1).to(device=device)
        val = torch.concatenate(val, dim=-1).to(device=device)
        return self.decompress(idx, val, xshape, device, dtype)


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


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


def _get_divisors(n):
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
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


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n