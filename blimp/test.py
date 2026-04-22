from pathlib import Path
from tqdm import tqdm
text = Path('/Users/xiulinyang/Desktop/conll/train.txt').read_text().strip().split('\n')
tokens = 0

for t in tqdm(text):
    tokens+=len(t.split())


print(tokens)