import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper parameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

# dataset
dataset_name = 'tinyshakspeare'

# -------

torch.manual_seed(1337)

with open(f'../../data/{dataset_name}/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all unique characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: torch.LongTensor([stoi[ch] for ch in s])
decode = lambda x: ''.join([itos[i] for i in x])

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        X, Y = get_batch(split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)
        # i.e. 4 heads of 8-dimensional self-attention
        # the total dimension of the self-attention is n_embd = 32
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # self.ffwd = FeedForward(n_embd)

        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and target are both (B, T) tensor of integers
        B, T = idx.shape
        tok_embd = self.token_embedding(idx) # (B, T, C)
        pos_embd = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_embd + pos_embd

        # self-attention
        # x = self.sa_head(x) # (B, T, C)
        # x = self.sa_heads(x) # (B, T, C)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, V)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_token):
            # crop idx to the lst block_size tokens
            # since the embedding table only has block_size rows
            idx_cond = idx[:, -block_size:]
            # get prediction
            logits, loss = self(idx_cond)
            # focus only on the last timestamp
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append to the context
            idx = torch.cat([idx, next_token], dim=1)
        return idx

class FeedForward(nn.Module):
    """ a simple feed-forward module """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4* n_embd in transformer paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x) # (B, T, H)
        Q = self.query(x) # (B, T, H)
        # compute the attention scores("affinity")
        wei = (Q @ K.transpose(-2, -1)) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation
        V = self.value(x) # (B, T, H)
        out = wei @ V   # (B, T, T) @ (B, T, H) = (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel. """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # Add layer norm before going to self-attention and feed forward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # trick 1: residual connection
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

model = BigramModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter} Train loss {losses["train"]:.2f} Val loss {losses["val"]:.2f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    #eval the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(context, 1000)[0].tolist()))