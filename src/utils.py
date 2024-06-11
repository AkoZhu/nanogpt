import torch
import tiktoken
import math

class DataLoaderLite:
    def __init__(self, B, T, dataset='tinyshakespeare'):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open(f'../data/{dataset}/input.txt', 'r') as f:
            print(f'loading {dataset} dataset')
            text = f.read()
            print(f"text length {len(text)}")
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loading {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the position
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
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
