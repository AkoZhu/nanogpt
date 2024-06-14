from model.gpt2 import GPT, GPTConfig
from utils import DataLoaderLite
from utils import get_lr
import torch
import os 

# -------------
import tiktoken
# --------------
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#--------------
from hellaswag import iterate_examples, render_example, get_most_likely_row

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
B = 8                      # our micro batch size # Can use 32 
T = 2048                    # sequence length.     # can use 2048
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accu_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"The total desired_batch_size={total_batch_size}, with grad_accu_steps={grad_accu_steps}")

# get the dataset
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# create the model
model = GPT(GPTConfig(block_size=T, vocab_size=50304)) # fix the ugly number of vocab_size
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag and Generation
if use_compile:
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
max_lr = 6e-4 # * 3
min_lr = 0.1 * max_lr
warmup_steps = 715      # 375e6 / 2**19 = 715
max_steps = 19073       # 1e8 tokens / 2**19(0.5M) tokens per batch 

# use the GPT-3 hyperparameters
# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
# using the weight decay
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=max_lr, device=device)

# create the log directory and save the checkpoints and log to
log_dir = "../log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_rank{ddp_rank}.txt")
with open(log_file, "w") as f: # clear the log file
    pass 

enc = tiktoken.get_encoding('gpt2')
# we are doing 1 epoch
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluate the dataset to avoid overfitting
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.6f}")
            
            # writing the log_file and checkpoints 
            with open(log_file, "w") as f:
                f.write(f"step:{step} | val_loss:{val_loss_accum.item():.6f}\n")
            if step > 0 and step % 5000 == 0 or last_step:
                # save the model checkpoint
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # save the optimizer.state_dict()
                # careful about rng_seeds
                torch.save(checkpoint, checkpoint_path)

    # evaluate the hellaSwag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits 
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats to all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)

            num_total, num_correct_norm = num_total.item(), num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step} hella {acc_norm:.4f}\n")

    # Every 250 steps, we generate some samples 
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I am a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take th elogits at the last position
                logits = logits[:, -1, :]
                # get probabilities
                probs = torch.softmax(logits, dim=-1)
                # sample the top k
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # Getting (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # Getting (B, 1)
                # getting the exact token and append to the list
                xgen = torch.cat((xgen, xcol), dim=1)

        # print the generate text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # train the model
    optimizer.zero_grad()
    loss_accum = 0
    # gradient accumulation
    for micro_step in range(grad_accu_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
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
    lr = get_lr(step, 
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
        print(f"step {step} | lr:{lr:e} | loss:{loss_accum.item():.6f} | norm: {norm:.4f} |dt:{dt} | tokens/sec:{tokens_per_sec:.0f}")
        with open(log_file, "a") as f:
            f.write(f"step {step} | train {loss_accum.item():.6f}\n")

    # if step == 250:
    #     print("Trial run finished")
    #     break
if ddp:
    destroy_process_group()

# prefix tokens

