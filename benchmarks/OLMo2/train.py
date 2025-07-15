import aimrun
import click
from datasets import load_dataset
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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from olmo_core.nn.transformer.block import TransformerBlock
from transformers import AutoTokenizer, AutoModelForCausalLM

@click.command()
@click.option('--batch-size', default=2, help='input batch size for training and validation (default: 32)')
@click.option('--epochs', default=10, help='number of epochs to train (default: 10)')
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
def main(batch_size, epochs, replicator, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, shards, rand_seed, dataset, debug, sign, description, cluster, lr, accum):
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
    model_and_co = setup(batch_size, replicator, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, single, shards, rand_seed, dataset, debug, sign, lr, max_length, use_fp16)
    train(epochs, replicator, single, accum, *model_and_co)

def seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed()

def train(epochs, repl, single, accum, model, train_loader, val_loader, optimizer, scheduler, train_sampler):
    rank = int(os.environ['RANK'])
    for epoch in range(1, epochs+1):
        # train
        model.train()
        train_sampler.set_epoch(epoch)
        loss_samples = torch.zeros(2).to(model.device)
        metrics = {}
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}", disable=rank>0, colour="blue", ncols=150)):
            if single:
                batch["input_ids"] = batch["input_ids"].to(model.device)
                batch["labels"] = batch["labels"].to(model.device)
            if repl == single: # 'adamw'
                loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"]
                loss.backward()
            else:
                with model.no_sync(): # Disable gradient replication for the backward pass
                    loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"]
                    loss.backward()
            if (i+1) % accum == 0:             
                optimizer.step()                          
                optimizer.zero_grad()
            loss_samples[0] += loss.item()
            loss_samples[1] += len(batch)
            metrics.update({'train/loss': loss.item()})
            aimrun.track(metrics)
        if not repl == 'adamw':
            for i, replicator in enumerate(optimizer.replicators):
                if hasattr(replicator, "data_transmitted"):
                    metrics[f"data_transmitted_gb_{i}"] = sum(replicator.data_transmitted)/2**30
                    metrics[f"data_received_gb_{i}"] = sum(replicator.data_received)/2**30
        metrics.clear()
        # print training statistics
        if not single:
            dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        if rank == 0:
            train_loss = loss_samples[0] / loss_samples[1]
            print(f"Epoch {epoch} training loss  : {train_loss:.4f}")
            aimrun.track({'epoch/train/loss': train_loss}, step=epoch)
        # validate
        model.eval()
        loss_samples.zero_()
        metrics.clear()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating after epoch {epoch}", disable=rank>0, colour="green", ncols=150):
                if single:
                    batch["input_ids"] = batch["input_ids"].to(model.device)
                    batch["labels"] = batch["labels"].to(model.device)
                loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"]
                loss_samples[0] += loss.item()
                loss_samples[1] += len(batch)
                metrics.update({'val/loss': loss.item()})
                aimrun.track(metrics)
                metrics.clear()
        # print validation statistics
        if not single:
            dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        if rank == 0:
            val_loss = loss_samples[0] / loss_samples[1]
            print(f"Epoch {epoch} validation Loss: {val_loss:.4f}")
            aimrun.track({'epoch/val/loss': val_loss}, step=epoch)
        scheduler.step()
    dist.destroy_process_group()
    aimrun.close()

def setup(batch_size, repl, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, single, shards, rand_seed, dataset, debug, detonation_sign, lr, max_length, use_fp16):
    if rand_seed is not None:
        seed(rand_seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # OLMo doesn't use pad_token by default
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16 if use_fp16 else torch.float32, trust_remote_code=True)

    # Load Dolma dataset
    datadir = "/pfs/lustrep1/scratch/project_465001960/mhf/datasets/" 
    dataset = load_dataset('json', data_files=f"{datadir}/{'v1_5r2_sample-*.json.gz'}", trust_remote_code=True, streaming=True).train_test_split(test_size=0.1)
    tokenized_train_dataset = dataset['train'].map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True, remove_columns=dataset["train"].column_names)
    tokenized_val_dataset = dataset['test'].map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True, remove_columns=dataset["train"].column_names)

    # Add labels (causal LM: labels == input_ids)
    tokenized_train_dataset = tokenized_train_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True) 
    tokenized_val_dataset = tokenized_val_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True) 
    train_dataset=tokenized_train_dataset
    val_dataset=tokenized_val_dataset

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=DistributedSampler(val_dataset))

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
    scheduler = StepLR(optim, step_size=1, gamma=0.85)
    return model, train_loader, val_loader, optimizer, scheduler, train_sampler


def preprocess_function(example, tokenizer, max_length):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

if __name__ == '__main__':
    main()
