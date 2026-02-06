import torch
from .bucket import Bucket

class BucketManager:
    def __init__(self, bucket_size_bytes=32 * 1024 * 1024, device=None, process_group=None):
        self.bucket_size_bytes = bucket_size_bytes
        self.device = device or torch.device('cuda')
        self.process_group = process_group  # for intra-node reduction
        self.buckets = []  # list of Bucket objects
        # map param -> (bucket_idx, entry_idx)
        self.param_to_bucket = {}
        # map param -> actual gradient size/shape (determined during first backward)
        self.param_grad_info = {}
        self._make_buckets()

    def _make_buckets(self):
        # create an initial bucket (we'll grow on demand)
        self.buckets = [Bucket(self.bucket_size_bytes, device=self.device, process_group=self.process_group)]

    def find_or_create_bucket_for(self, param):
        for i, b in enumerate(self.buckets):
            if b.add_entry(param):
                # record param mapping: bucket index and its entry position (entry == last index)
                return i, len(b.entries) - 1
        # if we reached, need a new bucket
        newb = Bucket(self.bucket_size_bytes, device=self.device, process_group=self.process_group)
        idx = len(self.buckets)
        self.buckets.append(newb)
        ok = newb.add_entry(param)
        assert ok, "single parameter larger than bucket size; increase bucket_size_bytes"
        return idx, len(newb.entries) - 1

    def register_param_hooks(self, model, optimizer=None):
        # walk params and register hooks - defer bucket allocation until first backward
        for p in model.parameters():
            if p.requires_grad:
                # Don't allocate buckets yet - do it dynamically on first gradient
                p.register_hook(self._make_hook(p))

    def _make_hook(self, param):
        # called during backward with grad (tensor)
        def hook(grad):
            # First time seeing this gradient - allocate bucket space
            if param not in self.param_to_bucket:
                grad_shape = grad.shape
                grad_numel = grad.numel()
                grad_dtype = grad.dtype
                
                # Store gradient info
                self.param_grad_info[param] = (grad_shape, grad_numel, grad_dtype)
                
                # Create a dummy param-like object with the correct size for bucket allocation
                class GradWrapper:
                    def __init__(self, shape, dtype, device):
                        self.shape = shape
                        self.dtype = dtype
                        self.device = device
                    def numel(self):
                        return grad_numel
                
                grad_wrapper = GradWrapper(grad_shape, grad_dtype, grad.device)
                bidx, eidx = self.find_or_create_bucket_for(grad_wrapper)
                self.param_to_bucket[param] = (bidx, eidx)
            
            bucket_idx, entry_idx = self.param_to_bucket[param]
            bucket = self.buckets[bucket_idx]
            
            _, start, numel, shape, dtype = bucket.entries[entry_idx]
            
            # flatten grad to contiguous
            if not grad.is_contiguous():
                grad = grad.contiguous()
            
            grad_flat = grad.view(-1)
            actual_numel = grad_flat.numel()
            
            # Sanity check - should match now
            if actual_numel != numel:
                raise RuntimeError(f"Gradient size mismatch: expected {numel}, got {actual_numel}")
            
            # copy tensor into bucket buffer (cast if needed)
            dest = bucket.buffer[start:start+numel]
            
            if grad.dtype != bucket.dtype:
                dest.copy_(grad_flat.to(bucket.dtype))
            else:
                dest.copy_(grad_flat)
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
            for (param_or_wrapper, start, numel, shape, dtype) in b.entries:
                # Find the actual parameter (param_or_wrapper might be GradWrapper)
                actual_param = None
                for p, (bidx, eidx) in self.param_to_bucket.items():
                    if bidx < len(self.buckets) and eidx < len(self.buckets[bidx].entries):
                        entry_param = self.buckets[bidx].entries[eidx][0]
                        if entry_param is param_or_wrapper:
                            actual_param = p
                            break
                
                if actual_param is None:
                    continue
                
                out_flat = b.buffer[start:start+numel]
                # convert back to param dtype and view to original shape
                if dtype != b.dtype:
                    val = out_flat.to(dtype).view(shape)
                else:
                    val = out_flat.view(shape)
                # store reduced grad into param.grad (overwrite)
                if actual_param.grad is None:
                    actual_param.grad = val.clone()  # ensure grad is a separate tensor that autograd expects
                else:
                    actual_param.grad.copy_(val)
            # after copying back, reset bucket for next iteration
            b.reset_for_next()
