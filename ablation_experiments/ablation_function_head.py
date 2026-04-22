from glob import glob
from collections import Counter
import pandas as pd

p_d = {
    "natural_function": "natural_function_blimp_results/",
    "random_function": "random_function_blimp_results/",
    "five_function": "five_function_blimp_results/",
    "within_boundary": "within_boundary_blimp_results/",
    "more_function": "more_function_blimp_results/",
    "bigram_function": "bigram_function_blimp_results/",
}

EPOCH = 10

with open("function_head_42.csv", "w") as f_h:
    f_h.write("condition,layer,head,freq\n")

    for cond in p_d.keys():
        pattern = f"uas_results/results_best_head_*_GPT2_{cond}_42-ckpt-{EPOCH}-sas_preds.tsv"
        best_lh = []

        for p in glob(pattern):

            if "determiner" in p:
                continue
            if "quantifier" in p:
                continue
            sas_pred = pd.read_csv(p, sep="\t")
            best_layer = int(sas_pred["best_layer"][0])
            best_head = int(sas_pred["best_head"][0])
            best_lh.append(f"{best_layer}-{best_head}")

        counter = Counter(best_lh)
        for layer in range(12):
            for head in range(12):
                key = f"{layer}-{head}"
                freq = counter.get(key, 0)
                f_h.write(f"{cond},{layer},{head},{freq}\n")