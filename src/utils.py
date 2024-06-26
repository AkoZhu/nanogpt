import torch
import tiktoken
import math
import os
import numpy as np


def load_tokens(filename) -> torch.Tensor:
    data = np.load(filename)
    data = data.astype(int)
    return torch.tensor(data, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train' ,dataset='tinyshakespeare'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val']

        # at init load tokens from disk and store them in memory
        # with open(f'../data/{dataset}/input.txt', 'r') as f:
        #     print(f'loading {dataset} dataset')
        #     text = f.read()
        #     print(f"text length {len(text)}")
        
        # get shard filenames
        data_root = "/root/autodl-tmp/data/edu_fineweb10B/train_data"
        shards = os.listdir(data_root)
        shards = [os.path.join(data_root, shard) for shard in shards]
        shards = sorted(shards)
        self.shards = shards

        assert len(shards) > 0, "no data shards found for split {split}"
        if process_rank == 0: # master process
            print(f"loading {len(shards)} shards for split {split}")
        
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens, dtype=torch.long)
        # print(f"loading {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.reset()

    def reset(self):
        # state
        # use shard and current position to keep track of where we are in the data
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the position
        self.current_position += B*T*self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + B*T*self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# cosine manner of learning rate
def get_lr(it, warmup_steps=10, max_steps=50, max_lr=3e-4, min_lr=3e-5):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decat_iters, return minimum learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coeff
