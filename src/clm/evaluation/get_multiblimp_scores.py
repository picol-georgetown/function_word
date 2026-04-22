from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import os
import argparse
import torch

# Set the MKL threading layer to GNU to avoid conflicts
os.environ['MKL_THREADING_LAYER'] = 'GNU'

CHECKPOINTS= list(range(1, 11))
MODEL_HUB = 'xiulinyang'
parser = argparse.ArgumentParser('get multiblimp scores')
parser.add_argument('model_name', type=str, help='model name')
parser.add_argument('experiment', help='experiment name', choices=['parallel10', 'parallel3', 'parallel3-100'])

args = parser.parse_args()

experiment = args.experiment
model_name = args.model_name
lang = args.model_name.split('_')[2]
# create results dataframe
results = pd.DataFrame(columns=['model', 'checkpoint','acc'])
results.to_csv(f'multiblimp_results_{experiment}/{model_name}.csv', mode='w', index=False)

# mapping for language codes used for model to language codes used for multiblimp
language_map = {
    'EN': 'eng',
    'RU': 'rus',
    'TR': 'tur',
    'DE': 'deu',
    'AR': 'arb'
}

# loop through all B-GPT models

    # loop through checkpoints
for c in CHECKPOINTS:
    m_str = f'{MODEL_HUB}/{model_name}'
    c_str = str(c)
    try:
        # Run the first evaluation for L1
        subprocess.run([
            "python", "eval_model.py",
            "--model_name", m_str,
            "--data_dir", f"multiblimp/hf_cache/{lang_data_id}/",
            "--revision",f'checkpoint-{c_str}',
            "--src_dir", "multiblimp",
            "--results_dir", f"multiblimp/multiblimp_results_epoch/{lang}_{vocab_size}-{c_str}",
            "--cache_dir", "multiblimp/hf_cache/"
        ], check=True, env={**os.environ})

        # Collect results for L1
        l1_results_path = f"multiblimp/multiblimp_results_epoch/{lang}_{vocab_size}-{c_str}/hf_cache_{lang_data_id}_data.tsv"
        df = pd.read_csv(l1_results_path, sep='\t')
        total_samples = len(df)
        correct_predictions = len(df[df['delta'] > 0])
        l1_accuracy = correct_predictions / total_samples

        # Append the new line with results to the CSV file
        new_line = pd.DataFrame({
            'model': [m_str],
            'checkpoint': [c],
            'acc': [l1_accuracy],

        })
        new_line.to_csv('multiblimp/multiblimp_results_epoch.csv', mode='a', header=False, index=False)
        print(new_line)
    except ValueError as e:
        print(f"Error processing model {m_str} at checkpoint {c_str}:")
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error details: {e}")

        # Continue with next checkpoint instead of stopping the entire script
        continue
