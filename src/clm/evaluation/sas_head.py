import os
import torch, pathlib
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
import torch
import random
import numpy as np
import json
# ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = pathlib.Path(os.getcwd()).resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'blimp')
SAS_PREDS_DIR = os.path.join(ROOT_DIR, 'sas_prob')

OUTPUT_DIR='uas_results'
DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because", "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has","did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as","over","than","during","about","between","among","except","through","around","after","like","off","without","under","before","throughout","unlike","across","toward","along","above","aboard","until","upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards","underneath","alongside"]
FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP)



def get_data(data_fp, function_list):
    sents = Path(data_fp).read_text().strip().split('\n')
    sents_all = []
    func_words_all = []
    for sent in sents:
        sentence = json.loads(sent)['sentence_good'][:-1].split()
        function_words = [(i, x) for i, x in enumerate(sentence) if x in function_list]
        punct = json.loads(sent)['sentence_good'][-1]
        sentence+=[punct]
        function_words_all = function_words*len(sentence)
        sents_all.append(sentence)
        func_words_all.append(function_words_all)
    return sents_all, func_words_all

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def read_sas_preds(pred_path):
    data = []
    with open(pred_path, encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            sent_id = parts[0]

            wid = int(parts[1])
            layer_strs = parts[2:]

            layer_preds = []
            for ls in layer_strs:
                heads = [int(x) for x in ls.split("-")]
                layer_preds.append(heads)
            data.append(((sent_id, wid), layer_preds))
    return data

def get_by_head_acc(data_fp, pred_data, function_list):
    sents_all, func_all = get_data(data_fp,function_list)
    _, first_layers = pred_data[0]
    L = len(first_layers)
    H = len(first_layers[0])
    assert L==12 and H==12, "Expected 12 layers and 12 heads"
    correct = torch.zeros(L, H, dtype=torch.long)
    total = torch.zeros(L, H, dtype=torch.long)

    for (sent_id, wid), layer_preds in pred_data:
        sent_id = int(sent_id)
        functions = func_all[sent_id]
        gold_functions = [x[0] for x in functions]
        for l in range(L):
            for h in range(H):
                pred_head = layer_preds[l][h]
                total[l, h] += 1
                if int(pred_head) in gold_functions:
                    correct[l, h] += 1

    uas = correct.float() / total.clamp_min(1)

    best_uas, best_idx = torch.max(uas.view(-1), dim=0)
    best_layer = best_idx // H
    best_head = best_idx % H

    return uas, best_layer.item(), best_head.item(), best_uas.item()

#
# def get_per_relation_acc(gold_rels, gold_heads, pred_data):
#     by_rel_results = {}
#     _, first_layers = pred_data[0]
#     L = len(first_layers)
#     H = len(first_layers[0])
#     assert L == 12 and H == 12, "Expected 12 layers and 12 heads"
#
#     for rel in tqdm(FUNCTION_WORDS):
#         correct = torch.zeros(L, H, dtype=torch.long)
#         total = torch.zeros(L, H, dtype=torch.long)
#
#         for (sid, wid), layer_preds in pred_data:
#             key = f"{sid}-{wid}"
#             if gold_rels[key] != rel:
#                 continue
#
#             gold = gold_heads[key]
#             for l in range(L):
#                 for h in range(H):
#                     pred_head = layer_preds[l][h]
#                     total[l, h] += 1
#                     if pred_head == gold:
#                         correct[l, h] += 1
#
#         uas = correct.float() / total.clamp_min(1)
#         best_uas, best_idx = torch.max(uas.view(-1), dim=0)
#         best_layer = best_idx // H
#         best_head = best_idx % H
#
#         by_rel_results[rel] = {
#             'best_uas': best_uas.item(),
#             'best_layer': best_layer.item(),
#             'best_head': best_head.item(),
#             'uas': uas,  # L×H tensor
#         }
#
#     return by_rel_results


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, default=None,
                        help='model name whose SAS head scores will be computed. default=None')
    parser.add_argument('-f', '--function_setting', help='language')
    args = parser.parse_args()
    function_setting = args.function_setting
    all_pesudo_words = []
    if function_setting =='more_function':
        pesudo_words = Path('function_word_pseudowords.txt').read_text().strip().split('\n')
        for line in pesudo_words:
            word, pseudo = line.strip().split('\t')
            all_pesudo_words.append(pseudo)
    fun_json = f'{function_setting}_blimp/adjunct_island.jsonl'
    PUD_FP = os.path.join(DATA_DIR, fun_json)
    function_w = list(FUNCTION_WORDS)+all_pesudo_words
    print(function_w)
    if args.model_name:
        model_name = args.model_name.split('/')[-1]
        fns = [fn for fn in os.listdir(SAS_PREDS_DIR) if model_name in fn]
    else:
        fns = [fn for fn in os.listdir(SAS_PREDS_DIR)]
    for fn in tqdm(fns):
        model_name = fn.split('@')[0]
        # get prediction data
        preds = read_sas_preds(f'{SAS_PREDS_DIR}/{fn}')
        by_head_results, best_layer, best_head, best_uas = get_by_head_acc(PUD_FP, preds,function_w)
        # by_rel_results = get_per_relation_acc(gold_rels, gold_heads,preds)
        # write UAS
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f'results_by_head_{model_name}.tsv'), 'w') as f:
            head_header = '\t'.join([f'h{h}' for h in range(by_head_results.shape[1])])
            f.write('layer\t' + head_header + '\n')
            for l_idx in range(by_head_results.shape[0]):
                row = by_head_results[l_idx]
                row_vals = '\t'.join(f'{float(x):.6f}' for x in row)
                f.write(f'{l_idx}\t{row_vals}\n')

        with open(os.path.join(OUTPUT_DIR, f'results_best_head_{model_name}.tsv'), 'w') as f:
            best_info = {'best_layer': best_layer, 'best_head': best_head, 'best_uas': best_uas}
            df = pd.DataFrame([best_info], columns=['best_layer', 'best_head', 'best_uas'])
            df.to_csv(f, sep='\t', index=False)


        # with open(os.path.join(OUTPUT_DIR, f'results_by_rel_{model_name}.tsv'), 'w') as f:
        #     f.write('relation\tbest_layer\tbest_head\tbest_uas\n')
        #     for rel in REL_NAMES:
        #         rel_info = by_rel_results[rel]
        #         f.write(f"{rel}\t{rel_info['best_layer']}\t{rel_info['best_head']}\t{rel_info['best_uas']}\n")


main()