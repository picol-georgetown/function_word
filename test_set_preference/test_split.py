from pathlib import Path
import json
functions = ['no_function','more_function','five_function', 'natural_function', 'random_function','bigram_function','within_boundary']
test_dict = {}

for f in functions:
    path = Path(f'/Users/xiulinyang/Desktop/conll/data/{f}/test.txt')
    lines = path.read_text().strip().split('\n')
    test_dict[f] = lines
    print(f, len(lines))

lengths = {k: len(v) for k, v in test_dict.items()}
if len(set(lengths.values())) != 1:
    raise ValueError(f"Test set size mismatch: {lengths}")

N = list(lengths.values())[0]
print("All test sets aligned, N =", N)

with open('test.jsonl', 'w') as js:
    for i in range(N):
        tt = {cond: test_dict[cond][i] for cond in functions}
        js.write(json.dumps(tt, ensure_ascii=False) + '\n')
