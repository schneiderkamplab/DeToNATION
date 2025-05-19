import click
from datasets import load_dataset
from detonation import DeMoReplicator, Optimizers, prepare_detonation
import functools
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

@click.command()
@click.option('--batch-size', default=32, help='input batch size for training and validation (default: 32)')
@click.option('--epochs', default=10, help='number of epochs to train (default: 10)')
def main(batch_size, epochs):
    model_and_co = setup(batch_size)
    train(epochs, *model_and_co)

def train(epochs, model, train_loader, val_loader, optimizer, scheduler, train_sampler):
    rank = int(os.environ['RANK'])
    for epoch in range(1, epochs+1):
        # train
        model.train()
        train_sampler.set_epoch(epoch)
        loss_samples = torch.zeros(2).to(model.device)
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}", disable=rank>0, colour="blue"):
            optimizer.zero_grad()
            with model.no_sync(): # Disable gradient replication for the backward pass
                loss = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )["loss"]
                loss.backward()
            optimizer.step()
            loss_samples[0] += loss.item()
            loss_samples[1] += len(batch)
        # print training statistics
        dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        if rank == 0:
            train_loss = loss_samples[0] / loss_samples[1]
            print(f"Epoch {epoch} training loss  : {train_loss:.4f}")
        # validate
        model.eval()
        loss_samples.zero_()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating after epoch {epoch}", disable=rank>0, colour="green"):
                loss = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])["loss"]
                loss_samples[0] += loss.item()
                loss_samples[1] += len(batch)
        # print validation statistics
        dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        if rank == 0:
            val_loss = loss_samples[0] / loss_samples[1]
            print(f"Epoch {epoch} validation Loss: {val_loss:.4f}")
        scheduler.step()
    dist.barrier()
    dist.destroy_process_group()

def setup(batch_size):
    torch.manual_seed(42)
    # prepare model
    tokenizer =  T5Tokenizer.from_pretrained("t5-small", legacy=False)
    model = T5ForConditionalGeneration(T5Config.from_pretrained("google-t5/t5-small"))
    # prepare dataset
    train_test_split = load_dataset("gursi26/wikihow-cleaned", split="train").train_test_split(test_size=0.2)
    train_dataset = WikiHow(tokenizer, train_test_split['train'], num_samples=15000)
    val_dataset = WikiHow(tokenizer, train_test_split['test'], num_samples=3000)
    dist.init_process_group()
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=DistributedSampler(val_dataset))
    # preapre distributed training
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else None
    model, optimizer = prepare_detonation(model, Optimizers.SGD, DeMoReplicator(compression_topk=4), fsdp_kwargs={"auto_wrap_policy": auto_wrap_policy, "mixed_precision": mixed_precision})
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
    return model, train_loader, val_loader, optimizer, scheduler, train_sampler

class WikiHow(Dataset):
    def __init__(self, tokenizer, dataset, num_samples):
        self.dataset, self.tokenizer = dataset.select(list(range(0, num_samples))), tokenizer
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
