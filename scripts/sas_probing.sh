#!/bin/bash

python src/clm/evaluation/sas_probe.py -d blimp/five_function_blimp -m xiulinyang/GPT2_five_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_five_function_42 -f five_function

python src/clm/evaluation/sas_probe.py -d blimp/natural_function_blimp -m xiulinyang/GPT2_natural_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_natural_function_42 -f natural_function

python src/clm/evaluation/sas_probe.py -d blimp/bigram_function_blimp -m xiulinyang/GPT2_bigram_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_bigram_function_42 -f bigram_function

python src/clm/evaluation/sas_probe.py -d blimp/more_function_blimp -m xiulinyang/GPT2_more_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_more_function_42 -f more_function


python src/clm/evaluation/sas_probe.py -d blimp/no_function_blimp -m xiulinyang/GPT2_no_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_no_function_42 -f no_function


python src/clm/evaluation/sas_probe.py -d blimp/random_function_blimp -m xiulinyang/GPT2_random_function_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_random_function_42 -f random_function


python src/clm/evaluation/sas_probe.py -d blimp/within_boundary_blimp -m xiulinyang/GPT2_within_boundary_42 -o sas_prob/
python src/clm/evaluation/sas_head.py -m xiulinyang/GPT2_within_boundary_42 -f within_boundary