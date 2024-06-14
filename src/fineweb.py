"""
FineWeb-EDU dataset for pretraining
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm

#------------
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
#------------
# local_dir = 'edu_fineweb10B'
# remote_name = 'sample-10BT'
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# the cache local directory if it doesn't exist
# DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), f"../data/{local_dir}")

LOCAL_RAW_DATA_DIR = '/root/autodl-tmp/data/edu_fineweb10B/raw_data/sample/10BT'
TARGET_DATA_DIR = '/root/autodl-tmp/data/edu_fineweb10B/train_data'
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
# creating the dir in /data/edu_fineweb10B
# print(DATA_CACHE_DIR)
# os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
# fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")


# load the dataset locally
# get all file name in the directory
data_files = os.listdir(LOCAL_RAW_DATA_DIR)
data_files = {"train": [file for file in data_files]}

fw = load_dataset("parquet", 
                data_dir=LOCAL_RAW_DATA_DIR, 
                data_files=data_files,
                cache_dir=HF_CACHE_DIR
            )

for split in fw.keys():
    print(f"{split}: {fw[split].num_rows} items")

# print(fw['train'].column_names)
fw = fw['train']
# import sys; sys.exit(0)
# init the tokenizer 
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>'] # end of text token

"""
    The utils function to tokenize the document
"""
def tokenize(doc):
    # tokenize s a single document and return a numpy array of unit16 tokens
    tokens = [eot] # special tokens 
    tokens.extend(enc.encode_ordinary(doc["text"])) # ignore the special character
    tokens_np = np.array(tokens)
    # we use the uint16 to encode 
    assert(0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# tokenize all documents and write output shards, each of  shard_size tokens(last shard has reminder)
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(TARGET_DATA_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            # finish the remainder
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            # finish the rest of the tokens
            shard_index += 1
            progress_bar = None 
            # populate the next shard with leftover of the current doc
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
    
    # write the remaining tokens as the last shard
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(TARGET_DATA_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])