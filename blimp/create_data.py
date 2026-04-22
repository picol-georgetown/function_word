from conllu import parse
from tqdm import tqdm
from glob import glob
from pathlib import Path
e = 0

open_class = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB'}
closed_class = {'ADP', 'AUX', 'CCONJ', 'DET', 'SCONJ'}
other = {'PUNCT', 'SYM', 'X'}

input_path = glob("/Users/xiulinyang/Desktop/data/baby/100M/train/source/*.conll")
output_path = "long_distance_baby.conll"
with open(output_path, "w", encoding="utf-8") as f_out:
    for i_p in tqdm(input_path):
        sents = Path(i_p).read_text(encoding="utf-8").strip().split("\n\n")
        for sentence in tqdm(sents):
            try:
                sentence = parse(sentence)[0]
                for word in sentence:
                    if word["upos"] in closed_class:
                        head_id = int(word["xpos"])
                        if head_id > word["id"] + 1:
                            f_out.write(sentence.serialize())
                            f_out.write("\n")
                            break
            except Exception:
                e += 1
                continue

print("error sentences:", e)