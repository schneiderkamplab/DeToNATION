## T5 training
Run the T5 example with FlexDeMo for text summarization.

## Install the requirements
```
pip install -r requirements.txt
```

## Start the training on each node
Assuming a training on 2 nodes host0 and host1 with 2 GPUs each, we start on host0:
```
RANK=0 ENDPOINT=host0:29400 train.py
```
And similarly on host1:
```
RANK=1 ENDPOINT=host0:29400 train.py
```
