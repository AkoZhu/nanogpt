#!/bin/bash

# lib
pip install -U huggingface_hub
pip install datasets tiktoken transformers


# 
source /etc/network_turbo

#
export HF_ENDPOINT=https://hf-mirror.com

# download the dataset
chmod +x fineweb-dataset.sh
./fineweb-dataset.sh

# preprocess the dataset 
python ./src/fineweb.py


