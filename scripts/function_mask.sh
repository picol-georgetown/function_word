#!/bin/bash

python ablation_experiments/ablation_mask.py five_function 53
python ablation_experiments/ablation_mask.py five_function 42
python ablation_experiments/ablation_mask.py five_function 67



python ablation_experiments/ablation_mask.py more_function 53
python ablation_experiments/ablation_mask.py more_function 42
python ablation_experiments/ablation_mask.py more_function 67


python ablation_experiments/ablation_mask.py random_function 53
python ablation_experiments/ablation_mask.py random_function 42
python ablation_experiments/ablation_mask.py random_function 67


python ablation_experiments/ablation_mask.py natural_function 53
python ablation_experiments/ablation_mask.py natural_function 42
python ablation_experiments/ablation_mask.py natural_function 67


python ablation_experiments/ablation_mask.py bigram_function 53
python ablation_experiments/ablation_mask.py bigram_function 42
python ablation_experiments/ablation_mask.py bigram_function 67

python ablation_experiments/ablation_mask.py within_boundary 53
python ablation_experiments/ablation_mask.py within_boundary 42
python ablation_experiments/ablation_mask.py within_boundary 67