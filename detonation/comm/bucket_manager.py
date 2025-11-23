import torch
from .bucket import Bucket

class BucketManager:
    def __init__(self, bucket_size_bytes=32 * 1024 * 1024, device=None):
        self.bucket_size_bytes = bucket_size_bytes
        self.device = device or torch.device('cuda')
        self.buckets = []  # list of Bucket objects
        # map param -> (bucket_idx, entry_idx)
        self.param_to_bucket = {}
        self._make_buckets()

    def _make_buckets(self):
        # create an initial bucket (we'll grow on demand)
        self.buckets = [Bucket(self.bucket_size_bytes, device=self.device)]

    def find_or_create_bucket_for(self, param):
        for i, b in enumerate(self.buckets):
            if b.add_entry(param):
                # record param mapping: bucket index and its entry position (entry == last index)
                return i, len(b.entries) - 1
        # if we reached, need a new bucket
        newb = Bucket(self.bucket_size_bytes, device=self.device)
        idx = len(self.buckets)
        self.buckets.append(newb)
        ok = newb.add_entry(param)
        assert ok, "single parameter larger than bucket size; increase bucket_size_bytes"
        return idx, len(newb.entries) - 1

    def register_param_hooks(self, model, optimizer=None):
        # walk params and place them into buckets and register hooks
        for p in model.parameters():
            if p.requires_grad:
                bidx, eidx = self.find_or_create_bucket_for(p)
                self.param_to_bucket[p] = (bidx, eidx)
                # register the hook
             #   print(f"bidx: {bidx}, eidx: {eidx}")
                p.register_hook(self._make_hook(p, bidx, eidx))

    def _make_hook(self, param, bucket_idx, entry_idx):
        # called during backward with grad (tensor)
        def hook(grad):
            bucket = self.buckets[bucket_idx]
            #print(f"Bucket {bucket_idx} offset before adding param grad: {bucket.offset}")
            # compute flat view into bucket.buffer
            # print("len buckets:", len(bucket.entries))
            # print("entry_idx:", entry_idx)
            # print(bucket.entries[entry_idx])
            _, start, numel, shape, dtype = bucket.entries[entry_idx]
            # flatten grad to contiguous
            if not grad.is_contiguous():
                grad = grad.contiguous()
            # copy tensor into bucket buffer (cast if needed)
            # use view to flatten param area
            dest = bucket.buffer[start:start+numel]
            # copy on GPU, casting to float32 if necessary
            if grad.dtype != bucket.dtype:
                dest.copy_(grad.view(-1).to(bucket.dtype))
            else:
                dest.copy_(grad.view(-1))
            # optionally clear the original grad to save memory (we'll restore later)
            # return None -> do not replace grad (we're doing out-of-band)
            # Trigger launch when bucket is full (or you can implement timer/finish logic)
            if bucket.offset == bucket.buffer.numel():
                # launch async op on comm stream
                bucket.launch_async_allreduce()
            return None
        return hook

    def finalize_all(self):
        # call at the end of backward (or before optimizer.step)
        for b in self.buckets:
            if not b.locked and b.offset > 0:
                b.launch_async_allreduce()
        # wait for all works to finish
        for b in self.buckets:
            b.wait()
        # write reduced values back into param.grad (and cast back)
        for b in self.buckets:
            for (param, start, numel, shape, dtype) in b.entries:
                out_flat = b.buffer[start:start+numel]
                # convert back to param dtype and view to original shape
                if dtype != b.dtype:
                    val = out_flat.to(dtype).view(shape)
                else:
                    val = out_flat.view(shape)
                # store reduced grad into param.grad (overwrite)
                if param.grad is None:
                    param.grad = val.clone()  # ensure grad is a separate tensor that autograd expects
                else:
                    param.grad.copy_(val)
            # after copying back, reset bucket for next iteration
            b.reset_for_next()
