from einops import rearrange
import torch

__all__ = ["DCTCompress"]

class DCTCompress:

    def _clamp_topk(x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @classmethod
    @torch.no_grad()
    def compress(cls, x, topk):
        xshape = x.shape
        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        topk = cls._clamp_topk(x, topk)

        idx = torch.topk(x.abs(), k=topk, dim=-1, largest=True, sorted=False).indices
        val = torch.gather(x, dim=-1, index=idx)

        return idx, val, xshape

    @torch.no_grad()
    def decompress(idx, val, xshape, device, dtype):
        x = torch.zeros(xshape, device=device, dtype=dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # TODO: Careful, this is nondeterministic across different CUDA devices! might cause errors to accumulate between nodes!
        x.scatter_reduce_(dim=-1, index=idx, src=val, reduce="mean", include_self=False).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x (h w) -> y x h w", h=xshape[2])

        return x

    @classmethod
    @torch.no_grad()
    def batch_decompress(cls, idx, val, xshape, device, dtype):
        idx = torch.concatenate(idx, dim=-1).to(device=device)
        val = torch.concatenate(val, dim=-1).to(device=device)
        return cls.decompress(idx, val, xshape, device, dtype)
