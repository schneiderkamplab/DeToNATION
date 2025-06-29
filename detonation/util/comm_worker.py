import threading
import torch
import torch.distributed as dist
from collections import defaultdict
import queue
import time

from ..util import DCTCompress
class CommWorker:
    def __init__(self, world_size, group, transform):
        self._replication_world_size = world_size
        self.replication_parallel_group = group
        self.transform = transform

        self.pending_comm = queue.Queue()
        self.inflight = {}  # param -> (idx_buf, val_buf, idx_handle, val_handle, xshape)
        self.ready_grad = {}
        self.ready_lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def post_communication(self, sparse_idx, sparse_val, param, xshape):
        self.pending_comm.put((sparse_idx.cpu(), sparse_val.cpu(), param.cpu(), xshape))

    def _worker_loop(self):
        while self.running:
            # Post new communications
            try:
                while not self.pending_comm.empty():
                    sparse_idx, sparse_val, param, xshape = self.pending_comm.get_nowait()
                    idx_buf = [torch.zeros_like(sparse_idx) for _ in range(self._replication_world_size)]
                    val_buf = [torch.zeros_like(sparse_val) for _ in range(self._replication_world_size)]
                    idx_handle = dist.all_gather(idx_buf, sparse_idx, group=self.replication_parallel_group, async_op=True)
                    val_handle = dist.all_gather(val_buf, sparse_val, group=self.replication_parallel_group, async_op=True)
                    self.inflight[param] = (idx_buf, val_buf, idx_handle, val_handle, xshape)
            except queue.Empty:
                pass

            # Check for completed communications
            to_finalize = []
            for param, (idx_buf, val_buf, idx_handle, val_handle, xshape) in self.inflight.items():
                if idx_handle.is_completed() and val_handle.is_completed():
                    to_finalize.append(param)

            # Finalize and decode gradients
            for param in to_finalize:
                idx_buf, val_buf, _, _, xshape = self.inflight.pop(param)

                # Optionally wait to be extra safe
                idx_handle.wait(); val_handle.wait()  # Usually already completed
                idx_buf = [x.to(torch.int64) for x in idx_buf]
                new_grad = self.transform.decode(
                    DCTCompress.batch_decompress(idx_buf, val_buf, xshape, param.device, param.dtype)
                )

                with self.ready_lock:
                    self.ready_grad[param] = new_grad

            time.sleep(0.001)  # avoid tight loop

    def get_ready_grad(self, param, device='cuda'):
        with self.ready_lock:
            grad = self.ready_grad.pop(param, None)
        return grad.to(device) if grad is not None else grad

    def stop(self):
        self.running = False
        self.thread.join()
