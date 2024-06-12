from model.gpt2 import GPT, GPTConfig
from utils import DataLoaderLite
from utils import get_lr
import torch

# we start to train the gpt2 model

# attempt to use the gpt if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device:{device}")

# setting seeds
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# get a data batch from tinyshakespeare
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# with open("../data/tinyshakespeare/input.txt", "r") as f:
#     text = f.read()
#     print(f"text length {len(text)}")
# text = text[:1000]

# tokens = enc.encode(text)
# B, T = 4, 32 # batch = 4 and sequence length = 32
# buf = torch.tensor(tokens[:B*T + 1], device=device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)



# gradient accumulation
# the gradient is accumulated for N steps before the optimizer step is taken
# this is equivalent to having a larger batch size of B x N
total_batch_size = 524288   # 2**19, ~0.5M tokens, in GPT-3
B = 16                      # our micro batch size
T = 1024                    # sequence length
assert total_batch_size % (B*T) == 0, "make sure total_batch_size is divisible by B*T"
grad_accu_steps = total_batch_size // (B*T)
print(f"The total desired_batch_size={total_batch_size}, with grad_accu_steps={grad_accu_steps}")

# get the dataset
train_loader = DataLoaderLite(B=B, T=T)

# get logits
model = GPT(GPTConfig(vocab_size=50304)) # fix the ugly number of vocab_size
model.to(device)
model = torch.compile(model) # like gcc and make it faster, kernel fusion

# setting less precise and faster
torch.set_float32_matmul_precision("high") # by default it is highest
# logits, loss = model(x.to(device), y.to(device))
# print(loss)
# print(f"logits.shape: {logits.shape}")

#-----------
# training loop
import time

# hyper parameters
max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50

# use the GPT-3 hyperparameters
# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
# using the weight decay
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    # gradient accumulation
    for micro_step in range(grad_accu_steps):
        x, y = train_loader.next_batch()
        x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # make the logits bfloat16, optimizing the runing speed.
            # the parameters will be still float32
            logits, loss = model(x, y)
        loss /= grad_accu_steps # make sure the loss is averaged
        loss_accum += loss.detach() # accumulate the loss for printing.
        loss.backward()
    # clip the global norm of the gradient at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # determine and set the learning rate based on the cosine manner
    lr = get_lr(i, 
                warmup_steps=warmup_steps, 
                max_steps=max_steps, 
                max_lr=max_lr, 
                min_lr=min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accu_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {i} | lr:{lr} | loss:{loss_accum.item():.6f} | dt:{dt} | tokens/sec:{tokens_per_sec:.0f}")


import sys; sys.exit(0)

# prefix tokens
