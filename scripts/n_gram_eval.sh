#!/bin/bash

N_GRAM_MODELS=('natural_function' 'five_function' 'within_boundary' 'no_function' 'more_function' 'bigram_function' 'five_function_within_boundary' 'bigram_function_within_boundary' 'random_function' 'five_function_random_function')

for model in "${N_GRAM_MODELS[@]}"
do
    mkdir -p "$model"_blimp_results/results_n_gram
    echo "Evaluating $model model..."
    python n_gram_eval.py n_gram/"$model".binary "$model"_blimp
done
