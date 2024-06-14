import os
# import requests


local_dir = 'edu_fineweb10B'
sub_dir = 'raw_data'
# # the cache local directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), f"../data/{local_dir}")

RAW_DATA_DIR = os.path.join(DATA_CACHE_DIR, sub_dir)


# # List of files in the subfolder (you should replace these with actual filenames)
# file_list = [
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/001_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/002_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/003_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/004_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/005_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/006_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/007_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/008_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/009_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/010_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/011_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/012_00000.parquet',
#     'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/013_00000.parquet',
# ]

# # Download each file
# def download_file(url, local_path):
#     response = requests.get(url)
#     response.raise_for_status()
#     with open(local_path, 'wb') as f:
#         f.write(response.content)


# for file_url in file_list:
#     local_path = os.path.join(RAW_DATA_DIR, os.path.basename(file_url))
#     print(f"Downloading {file_url} to {local_path}")
#     download_file(file_url, local_path)


from huggingface_hub import hf_hub_download

file_list = [
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/001_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/002_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/003_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/004_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/005_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/006_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/007_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/008_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/009_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/010_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/011_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/012_00000.parquet',
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/013_00000.parquet',
]

for i in range(len(file_list)):
    if i <= 3:
        continue
    file = file_list[i]
    # get the file name 
    file_name = file.split('/')[-1]
    prefix = 'sample/10BT'
    hf_hub_download(repo_id='HuggingFaceFW/fineweb-edu', repo_type='dataset', local_dir=RAW_DATA_DIR, filename=prefix + "/" + file_name)