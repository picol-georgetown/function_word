import os
from glob import glob
from tqdm import tqdm
from conllu import parse_incr
from collections import defaultdict

DATA_PATH = '/Users/xiulinyang/Downloads/ud-treebanks-v2.17/'
OUTPUT_FILE = 'boundary_stats_function.tsv'

CLOSED_CLASS = {'ADP', 'DET', 'SCONJ', 'CCONJ'}
# CLOSED_CLASS = {'ADJ', 'NUM'} # remove 'NOUN', 'PROPN', 'VERB'
def get_subtree_indices(head_id, children_map, exclude_rels=None):
    indices = {head_id}

    if head_id in children_map:
        for child_id, rel in children_map[head_id]:
            if exclude_rels and rel in exclude_rels:
                continue
            child_indices = get_subtree_indices(child_id, children_map, set())
            indices.update(child_indices)

    return indices


def process_boundary_stats(data_path, output_file):
    ud_folders = sorted(glob(f'{data_path}/*'))
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("Data\tLanguage\tBoundary_Rate\tTotal_Func_Words\n")
        for folder in tqdm(ud_folders, desc="Processing Languages"):
            if not os.path.isdir(folder): continue
            folder_name = os.path.basename(folder)
            try:
                lang_name = folder_name.split('-')[0].replace('UD_', '')
                tree_bank_name = folder_name.split('/')[-1]
            except:
                lang_name = folder_name

            conllu_files = glob(f'{folder}/*.conllu')

            total_func = 0
            boundary_func = 0

            for file_path in conllu_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for tokenlist in parse_incr(f):
                            children_map = defaultdict(list)
                            id_to_pos = {}
                            for token in tokenlist:
                                if not isinstance(token['id'], int): continue
                                if token['upos'] =='PUNCT': continue
                                id_to_pos[token['id']] = token['upos']
                                head_id = token['head']
                                if head_id != 0:
                                    children_map[head_id].append((token['id'], token['deprel'])) # how many dependent this head has

                            for token in tokenlist:
                                word = token['form']
                                if not isinstance(token['id'], int): continue
                                pos = token['upos']
                                if pos in CLOSED_CLASS:
                                    head_id = token['head']
                                    if head_id == 0:
                                        continue
                                    # print(word, pos)
                                    span_indices = get_subtree_indices(head_id, children_map)
                                    # print(span_indices)
                                    # print([x['form'] for x in tokenlist if x['id'] in span_indices])
                                    if not span_indices: continue

                                    min_idx = min(span_indices)
                                    max_idx = max(span_indices)
                                    curr_idx = token['id']

                                    pos_min_pos = [x['upos'] for x in tokenlist if x['id']==min_idx][0]
                                    pos_max_pos = [x['upos'] for x in tokenlist if x['id']==max_idx][0]

                                    if curr_idx == min_idx or curr_idx == max_idx:
                                        boundary_func += 1
                                    elif pos_min_pos in CLOSED_CLASS and min_idx+1 in span_indices:
                                        boundary_func+=1
                                    elif pos_max_pos in CLOSED_CLASS and max_idx-1 in span_indices:
                                        boundary_func+=1

                                    # else:
                                    #     print(tokenlist)
                                    total_func += 1

                except Exception as e:
                    # print(f"Error: {e}")
                    continue


            if total_func > 0:
                rate = boundary_func / total_func
                f_out.write(f"{tree_bank_name}\t{lang_name}\t{rate:.4f}\t{total_func}\n")


if __name__ == "__main__":
    process_boundary_stats(DATA_PATH, OUTPUT_FILE)