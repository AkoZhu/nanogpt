#!/bin/bash

# create the necessary folder
target_dir='/root/autodl-tmp/data/edu_fineweb10B/raw_data/sample/10BT'
mkdir -p $target_dir

urls=(
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/001_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/002_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/003_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/004_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/005_00000.parquet"    
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/006_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/007_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/008_00000.parquet"    
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/009_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/010_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/011_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/012_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/013_00000.parquet"
)

for url in "${urls[@]}"; do
    wget -P /root/autodl-tmp/data/edu_fineweb10B/raw_data/sample/10BT $url
done