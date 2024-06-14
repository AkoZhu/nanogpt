#!/bin/bash

urls=(
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/010_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/011_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/012_00000.parquet"
"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/013_00000.parquet"
)

for url in "${urls[@]}"; do
    wget -P /root/autodl-tmp/data/edu_fineweb10B/raw_data/sample/10BT $url
done