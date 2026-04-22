from glob import glob
from pathlib import Path
from conllu import parse
import pandas as pd
from collections import Counter
closed_class = {'ADP', 'AUX', 'CCONJ', 'DET', 'SCONJ'}

function_words = {
    'ADP': [],
    'AUX': [],
    'CCONJ': [],
    'DET': [],
    'SCONJ': []
}
for ud_data in glob('/Users/xiulinyang/Desktop/TODO/en_ud/*.conllu'):
    sents = Path(ud_data).read_text().strip().split('\n\n')
    for sent in sents:
        sentence = parse(sent)[0]
        for word in sentence:
            if word['upos'] in closed_class:
                function_words[word['upos']].append(word['form'].lower())


with open('function_words.txt', 'w') as f:
    for k, v in function_words.items():
        print(Counter(v).most_common())

