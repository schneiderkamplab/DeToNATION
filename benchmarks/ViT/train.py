import os
import aimrun
import click
import torch
import functools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mltiming import timing_iterator, timing
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from detonation import DeMoReplicator, FullReplicator, NoReplicator, RandomReplicator, prepare_detonation, Optimizers
from transformers.models.vit.modeling_vit import ViTLayer
import torch.distributed as dist


@click.command()
@click.option('--dataset', default='cifar100', type=click.Choice(['cifar10', 'cifar100']))
@click.option('--batch-size', default=32, help='input batch size for training and validation (default: 32)')
@click.option('--epochs', default=10, help='number of epochs to train (default: 10)')
@click.option('--repl', '--replicator', default='deto-demo', type=click.Choice(['deto-demo', 'deto-full', 'deto-none', 'adamw', 'deto-random']))
@click.option("--optimizer", "--optim",type=click.Choice([opt.value for opt in Optimizers], case_sensitive=False), default="sgd")
@click.option('--compression-rate', default=0.1)
@click.option('--compression-topk', default=2)
@click.option('--compression-chunk', default=64)
@click.option('--replicate-every', default=1)
@click.option('--skip-every', default=None, type=int)
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']), default='cuda')
@click.option('--shards', default=None, type=int, help="Number of shards per replication group (default: number of GPUs per node)")
@click.option('--rand-seed', default=None, type=int, help="Seed for random generators in numpy and torch")
def main(dataset, batch_size, epochs, repl, optimizer, compression_rate, compression_topk, compression_chunk, replicate_every, skip_every, device, shards, rand_seed):
    rank, nnodes, gpus = int(os.environ['RANK']), int(os.environ['NNODES']), 4
    run_args = click.get_current_context().params
    run_args.update({
        'nnodes': nnodes,
        'gpus': gpus,
    })
    aimrun.init(repo='.', experiment='ViT', args=run_args)
    if rank == 0:
        print(aimrun.get_runs()[0].hash)
    single = device in ('cpu', 'mps') or (device == 'cuda' and nnodes == gpus == 1)
    model_and_co = setup(dataset, batch_size, repl, optimizer, compression_rate, compression_topk, compression_chunk, replicate_every, skip_every, device, single, shards, rand_seed)
    train(epochs, repl, single, *model_and_co)

def seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed()

def train(epochs, repl, single, model, train_loader, val_loader, optimizer, scheduler, train_sampler):
    rank = int(os.environ['RANK'])
    for epoch in range(1, epochs+1):
        model.train()
        train_sampler.set_epoch(epoch)
        loss_samples = torch.zeros(2).to(model.device)
        metrics = {}
        for inputs, targets in tqdm(train_loader, desc=f"Training epoch {epoch}", disable=rank>0, colour="blue", ncols=150):
            if single:
                batch = batch.to(model.device)
            optimizer.zero_grad()
            if repl == 'adamw' or single:
                loss = model(inputs, labels=targets).loss
                loss.backward()
            else:
                with model.no_sync(): # Disable gradient replication for the backward pass
                    loss = model(inputs, labels=targets).loss
                    loss.backward()
            optimizer.step()
            loss_samples[0] += loss.item()
            loss_samples[1] += len(inputs)
            metrics.update({'train/loss': loss.item()})
            aimrun.track(metrics)
        # print training statistics
        if not single:
            dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        if rank == 0:
            train_loss = loss_samples[0] / loss_samples[1]
            print(f"Epoch {epoch} training loss  : {train_loss:.4f}")
            aimrun.track({'epoch/train/loss': train_loss}, step=epoch)
        # validation
        model.eval()
        loss_samples.zero_()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validating after epoch {epoch}", disable=rank>0, colour="green", ncols=150):
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                outputs = model(inputs, labels=targets)
                loss = outputs.loss
                loss_samples[0] += loss.item()
                loss_samples[1] += len(inputs)
                metrics.update({'val/loss': loss.item()})
                aimrun.track(metrics)
                _, predicted = outputs.logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        # End of epoch - calculate global accuracy
        correct_tensor = torch.tensor(correct, device=model.device)
        total_tensor = torch.tensor(total, device=model.device)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        global_train_acc = 100.0 * correct_tensor.item() / total_tensor.item()
        # print validation statistics
        if not single:
            dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        if rank == 0:
            val_loss = loss_samples[0] / loss_samples[1]
            print(f"Epoch {epoch} validation Loss: {val_loss:.4f}, Train Acc: {global_train_acc:.3f}%")
            aimrun.track({'epoch/val/loss': val_loss}, step=epoch)
            aimrun.track({'epoch/val/accuracy': global_train_acc}, step=epoch)
        scheduler.step()
    dist.destroy_process_group()
    aimrun.close()

def setup(dataset, batch_size, repl, optimizer, compression_rate, compression_topk, compression_chunk, replicate_every, skip_every, device, single, shards, rand_seed):
    if rand_seed is not None:
        seed(rand_seed)

    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=384,  # ViT-Small uses 384 dim
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=384 * 4, # MLP size is typically 4x hidden size
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels= 100 if dataset == 'cifar100' else 10
    )
    model = ViTForImageClassification(config)
    model = model.to(device)

    # Define Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),  # Resize to ViT's expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if dataset == 'cifar10' else
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),  # Resize to ViT's expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if dataset == 'cifar10' else
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load dataset
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:  
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=DistributedSampler(val_dataset))
    # prepare distributed training
    if device == 'cuda':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={ViTLayer})
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else None
    if single:
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    elif repl.startswith('deto-'):
        if repl == 'deto-demo':
            replicator = DeMoReplicator(compression_topk=compression_topk, compression_chunk=compression_chunk)
        elif repl == 'deto-random':
            replicator = RandomReplicator(compression_rate=compression_rate, seed=rand_seed if rand_seed is not None else 42)
        elif repl == 'deto-full':
            replicator = FullReplicator()
        else:
            replicator = NoReplicator()
        opt_enum = Optimizers(optimizer.lower())
        model, optimizer = prepare_detonation(model, opt_enum, replicator, fsdp_kwargs={"auto_wrap_policy": auto_wrap_policy, "mixed_precision": mixed_precision}, replicate_every=replicate_every, skip_every=skip_every, sharding_group_size=shards)
    optim = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    scheduler = StepLR(optim, step_size=1, gamma=0.85)
    return model, train_loader, val_loader, optimizer, scheduler, train_sampler

if __name__ == '__main__':
    main()