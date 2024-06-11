import inspect
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # but it calls the "bias"
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size() # batch_size, block_size, n_embd
        # calculate query, key, values for all heads in batch
        # e.g. in GPT-2(124M), n_head = 12, hs = 6, so nh*hs = 768 = C in transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # split q, k, v into multiple heads
        # nh is the number of heads, hs is the head size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materialize the large (T, T) matrix for all the queries and keys)
        # attention with the normalization factor, making the variance to around 1
        # using the triangle matrix to get the mask in decoder.
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att@v # (B, nh, T, T) @ (B, nh, T, ns) -> (B, nh, T, ns)
        
        # Use the flash attention, which is more efficient
        # it uses the online softmax computing and kernel fusion to optimize
        # the attention computation.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        #output the projection
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_gelu   = nn.GELU(approximate="tanh") 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    # block_size: int = 256
    # vocab_size: int = 65
    # n_layer:    int = 6
    # n_head:     int = 6
    # n_embd:     int = 384
    # -----------
    # 124M
    block_size:  int = 1024  # max sequence length
    vocab_size:  int = 50257 # number of tokens, 50,000 BPT merges + 256 bytes tokens + 1 <End of text token>
    n_layer:     int = 12    # number of layers
    n_head:      int = 12    # number of heads
    n_embd:      int = 768   # embedding dimension



class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte   = nn.Embedding(config.vocab_size, config.n_embd),
            wpe   = nn.Embedding(config.block_size, config.n_embd),
            h     = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f  = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme, the wte should be the same as the final lm_head
        # the reason is that if two tokens have the similarity, they should be mapped to
        # the same embedding space, and they should be projected back to the same 
        # vocabulary space. 
        # save a lot of parameters
        self.transformer.wte.weight = self.lm_head.weight

        # initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            # roughly 1/sqrt(in_dim)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Connot forward sequence of length {T}, block size is {self.config.block_size}"
        # embedding the token
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape T
        pos_emb = self.transformer['wpe'](pos)
        tok_emb = self.transformer['wte'](idx)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer['h']:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # logits.shape=(B, T, vocab_size), target.shape=(B, T)
            # (B*T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        # all of the candidate parameters that required grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p not in decay_params]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")
        
        # create AdamW optimizer 
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt: {model_type}")

        #n_layer, n_head and n_embd are the determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768), # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M 
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoint
        config_args['block_size'] = 1024 # always 1024 for GPT model
        
        # create a from-sratch initialized miniGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned, matching the name snd shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias') or not k.endswith('.attn.masked_bias')]
        # basically the openai checkpoints use a "Conv1D" module, but we onlu want to use Vanila linear layer
        # so we need to transpose the weight matrix
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #valilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

if __name__ == "__main__":
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to('cuda')

    # Tokenize the words
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to('cuda')

    # Generate the text
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sammpling of 50(huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the result
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

