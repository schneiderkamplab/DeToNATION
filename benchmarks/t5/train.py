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
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block


@click.command()
@click.option('--batch-size', default=32, help='input batch size for training and validation (default: 32)')
@click.option('--epochs', default=10, help='number of epochs to train (default: 10)')
@click.option('--replicator', '--repl', default='deto-demo', type=click.Choice(['deto-demo', 'deto-full', 'deto-none', 'adamw', 'deto-random', 'deto-slice', 'deto-stride']))
@click.option("--optimizer", "--optim",type=click.Choice([opt.value for opt in Optimizers], case_sensitive=False), default="sgd")
@click.option('--compression-rate', default=0.0625)
@click.option('--compression-topk', default=4)
@click.option('--compression-chunk', default=64)
@click.option('--model', default='google-t5/t5-base', type=click.Choice(['google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large']))
@click.option('--replicate-every', default=1)
@click.option('--skip-every', default=None, type=int)
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']), default='cuda')
@click.option('--shards', default=None, type=int, help="Number of shards per replication group (default: number of GPUs per node)")
@click.option('--rand-seed', default=None, type=int, help="Seed for random generators in numpy and torch")
@click.option('--dataset', default='OpusBooks', type=click.Choice(['WikiHow', 'OpusBooks']), help='Dataset to train on.')
@click.option('--debug', default='False', type=bool, help="Enable debugging -> Limit dataset size.")
@click.option('--sign', default=True, type=bool, help="Use sign of gradients or full values.")
@click.option('--description', default='', type=click.STRING, help='String comment for aim.')
@click.option('--cluster', default='', type=click.STRING, help='Specify compute resource for aim logging')
def main(batch_size, epochs, replicator, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, shards, rand_seed, dataset, debug, sign, description, cluster):
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
    aimrun.init(repo='.', experiment='t5', description=description, args=run_args)
    if rank == 0:
        print('Aim hash: ', aimrun.get_runs()[0].hash)
    single = device in ('cpu', 'mps') or (device == 'cuda' and nnodes == gpu_per_node == 1)
    model_and_co = setup(batch_size, replicator, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, single, shards, rand_seed, dataset, debug, sign)
    train(epochs, replicator, single, *model_and_co)

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
        # train
        model.train()
        train_sampler.set_epoch(epoch)
        loss_samples = torch.zeros(2).to(model.device)
        metrics = {}
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}", disable=rank>0, colour="blue", ncols=150):
            if single:
                batch["source_ids"] = batch["source_ids"].to(model.device)
                batch["source_mask"] = batch["source_mask"].to(model.device)
                batch["target_ids"] = batch["target_ids"].to(model.device)
            optimizer.zero_grad()
            if repl == single: # 'adamw'
                loss = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )["loss"]
                loss.backward()
            else:
                with model.no_sync(): # Disable gradient replication for the backward pass
                    loss = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )["loss"]
                    loss.backward()
            optimizer.step()
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
                    batch["source_ids"] = batch["source_ids"].to(model.device)
                    batch["source_mask"] = batch["source_mask"].to(model.device)
                    batch["target_ids"] = batch["target_ids"].to(model.device)
                loss = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])["loss"]
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

def setup(batch_size, repl, optimizer, compression_rate, compression_topk, compression_chunk, model, replicate_every, skip_every, device, single, shards, rand_seed, dataset, debug, detonation_sign):
    if rand_seed is not None:
        seed(rand_seed)

    # prepare model
    tokenizer =  T5Tokenizer.from_pretrained(model, legacy=False)
    model = T5ForConditionalGeneration(T5Config.from_pretrained(model))
    # prepare dataset
    if dataset == 'WikiHow':
        train_test_split = load_dataset("gursi26/wikihow-cleaned", split="train").train_test_split(test_size=0.2)
        train_dataset = WikiHow(tokenizer, debug, train_test_split['train'], num_debug_samples=15000)
        val_dataset = WikiHow(tokenizer, debug, train_test_split['test'], num_debug_samples=3000)
    else:
        train_test_split = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train").train_test_split(test_size=0.2)
        train_dataset = OpusBooks(tokenizer, debug, train_test_split['train'], num_debug_samples=15000)
        val_dataset = OpusBooks(tokenizer, debug, train_test_split['test'], num_debug_samples=3000)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=DistributedSampler(val_dataset))
    # prepare distributed training
    if device == 'cuda':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})
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
        elif repl == 'deto-slice':
            replicator = SlicingReplicator(compression_rate=compression_rate, compression_chunk=compression_chunk)
        elif repl == 'deto-stride':
            replicator = StridingReplicator(compression_rate=compression_rate, compression_chunk=compression_chunk)
        else:
            replicator = NoReplicator()
        opt_enum = Optimizers(optimizer.lower())
        model, optimizer = prepare_detonation(model, opt_enum, replicator, fsdp_kwargs={"auto_wrap_policy": auto_wrap_policy, "mixed_precision": mixed_precision}, replicate_every=replicate_every, skip_every=skip_every, sharding_group_size=shards, detonation_sign=detonation_sign)
    else:
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, mixed_precision=mixed_precision, device_id=int(os.environ['LOCAL_RANK']), sharding_strategy=ShardingStrategy.HYBRID_SHARD)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    optim = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    scheduler = StepLR(optim, step_size=1, gamma=0.85)
    return model, train_loader, val_loader, optimizer, scheduler, train_sampler

class OpusBooks(Dataset):
    def __init__(self, tokenizer, debug, dataset, num_debug_samples):
        self.dataset, self.tokenizer = dataset.select(list(range(0, num_debug_samples))) if debug else dataset, tokenizer

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        source = self.dataset[index]['translation']['en']
        target = self.dataset[index]['translation']['fr']
        source = self.tokenizer.batch_encode_plus([source], max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target], max_length=150, padding='max_length', truncation=True, return_tensors="pt")
        return {"source_ids": source['input_ids'].squeeze(), "source_mask": source['attention_mask'].squeeze(), "target_ids": targets['input_ids'].squeeze(), "target_mask": targets['attention_mask'].squeeze()}

class WikiHow(Dataset):
    def __init__(self, tokenizer, debug, dataset, num_debug_samples):
        self.dataset, self.tokenizer = dataset.select(list(range(0, num_debug_samples))) if debug else dataset, tokenizer
    def __len__(self):
        return self.dataset.shape[0]
    def __getitem__(self, index):
        source = self.dataset[index]['text']
        target = self.dataset[index]['title']
        if not source or not target:
            source = "because of this , it is recommended to increase your intake of vitamin c while you are dealing with a virus . aside from taking a vitamin c supplement , you can also eat fruits that have high amounts of vitamin c . these include grapefruit , kiwi , strawberries , lemon , lime , blackberries , oranges , papaya , pineapple , pomelo , and raspberries . eat vegetables that are rich in vitamin c . these include brussel sprouts , broccoli , onions , garlic , red and green peppers , tomatoes , and radishes . you can also consider making vegetable soup , if you dont like eating raw veggies . if you have ever wondered why people always give their kids chicken noodle soup when they are sick , its because chicken soup is a wonder when it comes to recovering from a virus . not only does chicken soup act as an anti - inflammatory , it also temporarily helps to relieve congestion by unblocking your nasal passages . you can also add onions , garlic , and other veggies to your soup to boost its vitamin and mineral count . zinc governs enzymes in our body that activate different parts of our immune system that fight against infection . most people choose to take a 25 mg zinc supplement before one meal each day , but you can also add zinc - rich foods to your diet . these foods include spinach , mushrooms , beef , lamb , pork or chicken , and cooked oysters . zinc has been shown to be most effective when taken for two to three days at the beginning of a cold or flu . start taking zinc as soon as you think you may be getting sick . you can also purchase lozenges that contain zinc , which you can suck on . you can buy these and other zinc supplements at your local pharmacy . do not take zinc supplements if you take antibiotics such as tetracyclines , fluoroquinolones , penicillamine a drug used in wilsons disease , or cisplatin a medication used in cancer , due to the fact that zinc decreases the efficiency of these drugs . echinacea is a type of plant that is often made into a tea or taken as a supplement . when consumed , it helps to increase the number of leukocytes white blood cells that boost your immunity and other immune - related cells in your body . you can consume echinacea by drinking tea or juice made from the plant , or by taking supplements bought at a pharmacy or health foods store . other natural remedies to consider include eucalyptus , elderberry , honey , and reishi and shiitake mushrooms ."
            target = "how to treat a viral infection 2"
        for sub in ('Example of text:', 'Example of Summary:', '\n', '``', '"'):
            source, target = source.replace(sub, ''), target.replace(sub, '')
        source = self.tokenizer.batch_encode_plus([source], max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target], max_length=150, padding='max_length', truncation=True, return_tensors="pt")
        return {"source_ids": source['input_ids'].squeeze(), "source_mask": source['attention_mask'].squeeze(), "target_ids": targets['input_ids'].squeeze(), "target_mask": targets['attention_mask'].squeeze()}

if __name__ == '__main__':
    main()
