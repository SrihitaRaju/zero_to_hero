import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e7) # 100M tokens per shard, total of 100 shards


# Set up environment variables and directories
workspace_dir = '/workspace'
dataset_cache_dir = os.path.join(workspace_dir, 'new_dataset_cache')
os.environ['HF_HOME'] = os.path.join(workspace_dir, 'huggingface_cache')
os.environ['HF_DATASETS_CACHE'] = dataset_cache_dir
os.environ['TRANSFORMERS_CACHE'] = os.path.join(workspace_dir, 'transformers_cache')
os.environ['HF_MODULES_CACHE'] = os.path.join(workspace_dir, 'modules_cache')
os.environ['TMPDIR'] = os.path.join(workspace_dir, 'tmp')

# Clear existing dataset cache
import shutil
shutil.rmtree('/workspace/dataset_cache', ignore_errors=True)
shutil.rmtree('/workspace/huggingface_cache', ignore_errors=True)
shutil.rmtree('/workspace/transformers_cache', ignore_errors=True)
shutil.rmtree('/workspace/modules_cache', ignore_errors=True)


# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join('/workspace', local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

import pdb;pdb.set_trace()