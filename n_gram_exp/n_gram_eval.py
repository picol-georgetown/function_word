import math
import kenlm
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd
import os

def read_data(data_path):
    test_set = {}
    phenomenon_paths = glob(f'{data_path}/*.jsonl')
    for p in tqdm(phenomenon_paths):
        phenomenon_n = p.split('/')[-1].split('.')[0]
        phenomenon = pd.read_json(p, lines=True).to_dict(orient='records')
        sent_pair = [(x['sentence_bad'], x['sentence_good']) for x in phenomenon]
        test_set[phenomenon_n] = sent_pair
    return test_set

def eval_sent_pair(model, test_set):
    results = {}
    for phe, sents in tqdm(test_set.items()):
        correct = 0
        for sent in sents:
            sent_bad, sent_good = list(sent)
            score_bad = model.score(sent_bad,bos=True, eos=True)
            score_good = model.score(sent_good,bos=True, eos=True)
            if score_bad <score_good:
                correct+=1
        acc = correct/len(sents)
        results[phe] = acc
        print(phe, acc)
    return results



if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('eval_dataset', type=str, help='dataset name')

    args = args.parse_args()
    dataset = args.eval_dataset
    os.makedirs(f'{dataset}_results', exist_ok=True)
    model_name = args.model_name
    model = kenlm.LanguageModel(model_name)
    test = read_data(f'blimp/blimp/{dataset}')
    f_results = {}
    acc = eval_sent_pair(model, test)
    f_results['best'] = acc
    pd.DataFrame(f_results).to_csv(f'{dataset}_results/results_{model_name}_best.csv')