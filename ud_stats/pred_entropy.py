import os
import math
from glob import glob
from tqdm import tqdm
from conllu import parse_incr
from collections import defaultdict, Counter
import json

DATA_PATH = '/Users/xiulinyang/Downloads/ud-treebanks-v2.17/'
OUTPUT_FILE = 'dependency_entropy_stats.tsv'

CLOSED_CLASS = {'ADP', 'AUX', 'CCONJ', 'DET', 'SCONJ', 'PRON', 'PART'}
OPEN_CLASS = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PART', 'PROPN', 'VERB', 'NUM'}
IGNORE_TAGS = {'PUNCT', 'SYM', 'X', '_'}

with open('morph_complexity_updated.json', 'r') as fam:
    fam_results = json.load(fam)

def calculate_entropy(counts_dict):
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts_dict.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def process_dependency_entropy(data_path, output_file):
    ud_folders = sorted(glob(os.path.join(data_path, '*')))

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("Data\tLanguage\tDep_Closed_Entropy\tDep_Open_Entropy\n")

        for folder in tqdm(ud_folders, desc="Processing Languages"):
            if not os.path.isdir(folder):
                continue


            folder_name = os.path.basename(folder)
            try:
                lang_name = folder_name.split('-')[0].replace('UD_', '')
            except:
                lang_name = folder_name

            conllu_files = glob(os.path.join(folder, '*.conllu'))

            pair_counts = defaultdict(Counter)

            for conllu_path in conllu_files:
                with open(conllu_path, 'r', encoding='utf-8') as f:
                    for tokenlist in parse_incr(f):
                        id_to_pos = {t['id']: t['upos'] for t in tokenlist if isinstance(t['id'], int)}

                        for token in tokenlist:
                            if not isinstance(token['id'], int): continue

                            dep_pos = token['upos']
                            head_id = token['head']


                            if dep_pos in IGNORE_TAGS: continue


                            if head_id != 0 and head_id in id_to_pos:
                                head_pos = id_to_pos[head_id]

                                if head_pos not in IGNORE_TAGS:
                                    pair_counts[dep_pos][head_pos] += 1
                                    pair_counts[head_pos][dep_pos]+=1

            if not pair_counts:
                continue

            def get_weighted_entropy(target_class_set):
                total_entropy = 0.0
                total_weight = 0

                for dep_pos in target_class_set:
                    if dep_pos in pair_counts:

                        head_dist = pair_counts[dep_pos]
                        ent = calculate_entropy(head_dist)
                        weight = sum(head_dist.values())
                        print(ent)
                        print(weight)
                        total_entropy += ent * weight
                        total_weight += weight

                return total_entropy / total_weight if total_weight > 0 else 0

            closed_ent = get_weighted_entropy(CLOSED_CLASS)
            open_ent = get_weighted_entropy(OPEN_CLASS)
            if lang_name in fam_results:
                family = fam_results[lang_name]['language_family']
            else:
                family='unknown'
                print(lang_name)
            f_out.write(f"{folder_name}\t{family}\t{lang_name}\t{closed_ent:.4f}\t{open_ent:.4f}\n")


if __name__ == "__main__":
    process_dependency_entropy(DATA_PATH, OUTPUT_FILE)