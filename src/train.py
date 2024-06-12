from model.gpt2 import GPT, GPTConfig
from utils import DataLoaderLite
from utils import get_lr
import torch
import os 
# --------------
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# we start to train the gpt2 model

# attempt to use the gpt if available
# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = 'mps'
# print(f"using device:{device}")

# simple run
# python train.py
# DDP launch for e.g. 8 gpu
# torchrun --standalone --nproc_per_node=8 train.py

# Set up the DDP(distributed data parallel).
# torchrun command sets the env variables like RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a DDP process?
if ddp:
    # use DDP atm demands CUDA, we set the dev ice appropriately according to rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # if not DDP, we use the CPU or GPU
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
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
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accu_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"The total desired_batch_size={total_batch_size}, with grad_accu_steps={grad_accu_steps}")

# get the dataset
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

# create the model
model = GPT(GPTConfig(vocab_size=50304)) # fix the ugly number of vocab_size
model.to(device)
model = torch.compile(model) # like gcc and make it faster, kernel fusion
# wrap model using ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # ddp helps you average all backward gradient among gpus.
raw_model = model.module if ddp else model          # always contains the "raw" unwrapped model

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
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

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
        # only sync the gradient at the very last micro step 
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accu_steps - 1)
        loss.backward()
        # make the loss_accum to aggregate the loss from all processes rather than just th local.
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
    tokens_processed = train_loader.B * train_loader.T * grad_accu_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {i} | lr:{lr} | loss:{loss_accum.item():.6f} | dt:{dt} | tokens/sec:{tokens_per_sec:.0f}")


if ddp:
    destroy_process_group()

import sys; sys.exit(0)

# prefix tokens

