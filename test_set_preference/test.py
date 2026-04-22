from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

functions = [
    'natural_function',
    'no_function',
    'more_function',
    'five_function',
    'random_function',
    'bigram_function',
    'within_boundary'
]


cond_colors = {
    'natural_function':   '#1b9e77',
    'no_function':        '#d95f02',
    'more_function':      '#7570b3',
    'five_function':      '#e7298a',
    'random_function':    '#66a61e',
    'bigram_function':    '#e6ab02',
    'within_boundary':    '#a6761d'
}

BASE_DIR = "/Users/xiulinyang/Desktop/conll/data/"

plt.figure(figsize=(8, 6))

for f in tqdm(functions):
    pa = f"{BASE_DIR}/{f}/train.txt"

    lines = Path(pa).read_text().strip().split("\n")
    words = [w for line in lines for w in line.split()]

    freq = Counter(words)
    freqs_sorted = np.array(sorted(freq.values(), reverse=True))
    ranks = np.arange(1, len(freqs_sorted) + 1)

    plt.plot(
        np.log10(ranks),
        np.log10(freqs_sorted),
        label=f,
        color=cond_colors[f],
        alpha=0.3,
        linewidth=2
    )

plt.xlabel("log10(Rank)")
plt.ylabel("log10(Frequency)")
plt.title("Zipf Distributions Across Function-Word Conditions")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("zipf_all_conditions.pdf")
plt.close()