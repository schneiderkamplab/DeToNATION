import torch
import torch.distributed as dist
from collections import defaultdict

class Bucket:
    def __init__(self, size_bytes, device):
        # allocate large float tensor as buffer (size in bytes -> elements)
        # we allocate as float32; adapt dtype if needed
        self.device = device
        self.dtype = torch.float32
        elem_size = torch.finfo(self.dtype).bits // 8
        n_elems = (size_bytes + elem_size - 1) // elem_size
        self.buffer = torch.empty(n_elems, dtype=self.dtype, device=device, requires_grad=False)
        self.offset = 0
        self.entries = []  # list of (param, start, numel, shape, dtype)
        self.locked = False
        self.work = None
        # comm stream for NCCL ops (separate for overlap)
        self.comm_stream = torch.cuda.Stream(device=device)

    def add_entry(self, param):
        assert not self.locked
        numel = param.numel()
        dtype = param.dtype
        # we store in float32 buffer; cast during copy if needed
        if self.offset + numel > self.buffer.numel():
            return False
        start = self.offset
        self.entries.append((param, start, numel, param.shape, dtype))
        self.offset += numel
        return True

    def reset_for_next(self):
        self.offset = 0
        self.locked = False
        self.work = None

    def is_full(self):
        return self.offset >= self.buffer.numel()

    def launch_async_allreduce(self):
        # make sure no other comm running
        assert not self.locked
        self.locked = True
        # sync default stream -> comm_stream so buffer data is visible
        default_stream = torch.cuda.current_stream(self.device)
        # ensure copies to buffer (issued on default stream) are visible
        self.comm_stream.wait_stream(default_stream)
        # run allreduce on comm_stream
        with torch.cuda.stream(self.comm_stream):
            # NOTE: pass the tensor directly; async_op=True returns a Work handle
            self.work = dist.all_reduce(self.buffer[:self.offset], op=dist.ReduceOp.SUM, async_op=True)
        return self.work

    def wait(self):
        if self.work is not None:
            # wait on the work (blocks until comm finishes)
            self.work.wait()
            # ensure comm_stream syncs back to default stream before CPU/GPU access
            default_stream = torch.cuda.current_stream(self.device)
            default_stream.wait_stream(self.comm_stream)
