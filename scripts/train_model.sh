#!/bin/bash
set -euo pipefail

data_name=$1
seed=$2

model_name="GPT2"_"$data_name"_"$seed"

python generate_config.py \
  --data_name $data_name \
  --model_name $model_name \
  --seed $seed \
  --pretokenized_file \
  --overwrite_output_dir True


python train_tokenizer.py -c configs/"$model_name".json
python save_config.py -c configs/"$model_name".json
python train_clm.py configs/"$model_name".json
