import aimrun
import click
from datasets import load_dataset, Dataset
from detonation import DeMoReplicator, FullReplicator, NoReplicator, RandomReplicator, SlicingReplicator, StridingReplicator, prepare_detonation, Optimizers
import functools
import json
from mltiming import timing_iterator, timing
import numpy as np
import os
import random
import subprocess
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from olmo_core.nn.transformer.block import TransformerBlock
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

@click.command()
@click.option('--batch-size', default=2, help='input batch size for training and validation (default: 32)')
@click.option('--steps', default=10, help='steps to train for (default: 10)')
@click.option('--replicator', '--repl', default='deto-demo', type=click.Choice(['deto-demo', 'deto-full', 'deto-none', 'adamw', 'deto-random', 'deto-slice', 'deto-stride']))
@click.option("--optimizer", "--optim",type=click.Choice([opt.value for opt in Optimizers], case_sensitive=False), default="sgd")
@click.option('--compression-rate', default=0.0625)
@click.option('--compression-topk', default=4)
@click.option('--compression-chunk', default=64)
@click.option('--model', default='allenai/OLMo-1B', type=click.Choice(['allenai/OLMo-1B', 'allenai/OLMo-7B']))
@click.option('--replicate-every', default=1)
@click.option('--skip-every', default=None, type=int)
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']), default='cuda')
@click.option('--shards', default=None, type=int, help="Number of shards per replication group (default: number of GPUs per node)")
@click.option('--rand-seed', default=None, type=int, help="Seed for random generators in numpy and torch")
@click.option('--dataset', default='allenai/dolma', type=click.Choice(['allenai/dolma']), help='Dataset to train on.')
@click.option('--debug', default='False', type=bool, help="Enable debugging -> Limit dataset size.")
@click.option('--sign', default=True, type=bool, help="Use sign of gradients or full values.")
@click.option('--description', default='', type=click.STRING, help='String comment for aim.')
@click.option('--cluster', default='', type=click.STRING, help='Specify compute resource for aim logging')
@click.option('--lr', default=1e-3)
@click.option('--accum', default=1, type=int, help='Number of gradient accumulation steps (default: 1)')
@click.option('--save-dir', default='checkpoints', type=click.Path(exists=False, file_okay=False, dir_okay=True), help='Directory to save checkpoints')
@click.option('--save-every', default=-1, type=int, help='Save checkpoint every N steps (default: -1)')
def main(batch_size, steps, replicator, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, shards, rand_seed, dataset, debug, sign, description, cluster, lr, accum, save_dir, save_every):
    max_length = 1024
    use_fp16 = True
    if optimizer == 'deto-slice':
        raise Exception("The slicing replicator does not currently work.")
    rank, nnodes, gpu_per_node = int(os.environ['RANK']), int(os.environ['NNODES']), torch.cuda.device_count()
    git_hash = subprocess.getoutput('git rev-parse HEAD').strip()
    run_args = click.get_current_context().params
    run_args.update({
        'nnodes': nnodes,
        'gpu_per_node': gpu_per_node,
        'git_hash': git_hash,
    })
    run_args.pop('description')
    aimrun.init(repo='.', experiment='OLMo2', description=description, args=run_args)
    if rank == 0:
        print('Aim hash: ', aimrun.get_runs()[0].hash)
    single = device in ('cpu', 'mps') or (device == 'cuda' and nnodes == gpu_per_node == 1)
    model_and_co = setup(batch_size, replicator, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, single, shards, rand_seed, dataset, debug, sign, lr, max_length, use_fp16, steps)
    train(steps, replicator, single, accum, save_dir, save_every, *model_and_co)

def seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed()

def train(steps, repl, single, accum, save_dir, save_every, model, train_loader, optimizer, scheduler):
    rank = int(os.environ['RANK'])
    model.train()
    loss_samples = torch.zeros(2).to(model.device)
    metrics = {}
    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training steps", disable=rank>0, colour="blue", ncols=150)
    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if single:
            batch["input_ids"] = batch["input_ids"].to(model.device)
            batch["labels"] = batch["labels"].to(model.device)
        if repl == single:  # 'adamw'
            loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"]
            loss.backward()
        else:
            with model.no_sync():
                loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"]
                loss.backward()
        if (step + 1) % accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss_samples[0] += loss.item()
        loss_samples[1] += len(batch)
        metrics.update({'train/loss': loss.item()})
        aimrun.track(metrics)
        step += 1
        pbar.update(1)

        if save_every > 0 and (step % save_every == 0 or step == steps):
            if dist.get_rank() == 0:
                with FSDP.summon_full_params(model):
                    save_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                    model.save_pretrained(save_path)
                    print(f"Checkpoint saved at step {step}")
    pbar.close()
    dist.destroy_process_group()
    aimrun.close()

def setup(batch_size, repl, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, single, shards, rand_seed, dataset, debug, detonation_sign, lr, max_length, use_fp16, steps):
    if rand_seed is not None:
        seed(rand_seed)

    # Load tokenizer and model
    #tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    if debug:
        print("[DEBUG] Using debug tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("/leonardo_work/EUHPC_A04_086/OLMo-7B-local", local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # OLMo doesn't use pad_token by default
    
  
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", trust_remote_code=True)
    # Load Dolma dataset
    if debug:
        datadir = "/mnt/odinstorage/users/jnn/codes/DeToNATION/benchmarks/OLMo2/dolma-v1_6-sample"
    else:
        datadir = "/leonardo_work/EUHPC_A04_086/datasets/allenai/dolma" 
    stream_dataset = load_dataset('json', data_files=f"{datadir}/{'v1_5r2_sample-*.json.gz'}", streaming=True, trust_remote_code=True)['train']

    tokenized_train_dataset = TokenizedStreamingDataset(dataset=stream_dataset, tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, collate_fn=olmo_data_collator, num_workers=8)

    # prepare distributed training
    if device == 'cuda':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else None
    if single:
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
    elif repl.startswith('deto-'):
        if repl == 'deto-demo':
            replicator = DeMoReplicator(compression_topk=compression_topk, compression_chunk=compression_chunk)
        elif repl == 'deto-random':
            replicator = RandomReplicator(compression_rate=compression_rate, seed=rand_seed if rand_seed is not None else 42)
        elif repl == 'deto-full':
            replicator = FullReplicator()
        elif repl == 'deto-slice':
            replicator = SlicingReplicator(compression_rate=compression_rate, compression_chunk=compression_chunk)
        elif repl == 'deto-stride':
            replicator = StridingReplicator(compression_rate=compression_rate, compression_chunk=compression_chunk)
        else:
            replicator = NoReplicator()
        opt_enum = Optimizers(optimizer.lower())
        model, optimizer = prepare_detonation(model, opt_enum, replicator, fsdp_kwargs={"auto_wrap_policy": auto_wrap_policy, "mixed_precision": mixed_precision}, replicate_every=replicate_every, skip_every=skip_every, sharding_group_size=shards, detonation_sign=detonation_sign, lr=lr)
    else:
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, mixed_precision=mixed_precision, device_id=int(os.environ['LOCAL_RANK']), sharding_strategy=ShardingStrategy.HYBRID_SHARD)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
    optim = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    num_warmup_steps = int(0.03 * steps) 
    scheduler = get_linear_schedule_with_warmup(optimizer=optim, num_warmup_steps=num_warmup_steps, num_training_steps=steps)
    return model, train_loader, optimizer, scheduler


import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import hashlib

def olmo_data_collator(features):
    """
    Improved data collator for OLMo2 that handles edge cases
    """
    if not features:
        return {}
    
    batch = {}
    for key in features[0].keys():
        # Stack tensors and ensure they have the right shape
        stacked = torch.stack([f[key] for f in features])
        
        # Ensure attention_mask is properly formatted
        if key == "attention_mask":
            # Make sure attention mask has valid values (0s and 1s only)
            stacked = stacked.long()
            # Check for invalid attention masks (all zeros)
            mask_sums = stacked.sum(dim=1)
            if torch.any(mask_sums == 0):
                print(f"Warning: Found attention masks with all zeros. Mask sums: {mask_sums}")
                # Fix by setting at least the first token to be attended to
                stacked[mask_sums == 0, 0] = 1
        
        batch[key] = stacked
    
    return batch

class TokenizedStreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=2048):
        """
        Itearable dataset that tokenizes text data from a streaming dataset.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get rank and world size from torch.distributed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        for i, example in enumerate(self.dataset):
            # Shard across distributed workers. DistributedSampler needs to know the length of the dataset,
            if i % self.world_size != self.rank:
                continue

            text = example["text"]            
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            tokenized["labels"] = tokenized["input_ids"].clone()

            yield tokenized


if __name__ == '__main__':
    main()

    # CUDA_VISIBLE_DEVICES=3  NNODES=2 NPROC_PER_NODE=1 RANK=1 ENDPOINT=10.10.0.26:29500 ./run.sh --batch-size 2 --debug True
    # CUDA_VISIBLE_DEVICES=0  NNODES=2 NPROC_PER_NODE=1 RANK=0 ENDPOINT=10.10.0.26:29500 ./run.sh --batch-size 2 --debug True
