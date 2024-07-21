from model.gpt2 import GPT, GPTConfig
import torch
import torch.nn as nn
from torch.nn import functional as F

if __name__ == '__main__':
    # argparse
    import argparse
    parser = argparse.ArgumentParser(description='nanoGPT inference')
    parser.add_argument('--model', type=str, default='gpt2', help='model type')
    parser.add_argument('--model_path', type=str, default='../model/model_19072.pt', help='model path')

    args = parser.parse_args()

    num_return_sequences = 5
    max_length = 30

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    if args.model != 'nano-gpt':
        model = GPT.from_pretrained(args.model)
    else:
        model = GPT(GPTConfig(block_size=1024, vocab_size=50304))
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    model.eval()
    model.to(device)

    # Tokenize the words
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm Ako,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)

    # Generate the text
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            if args.model != 'nano-gpt':
                logits = model(x) # (B, T, vocab_size)
            else:
                logits, _ = model(x) # (B, T, vocab_size), loss
            
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