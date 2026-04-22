from minicons import scorer
import argparse
from huggingface_hub import list_repo_refs
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os

empty_categories = ['superlative_quantifiers_1', 'determiner_noun_agreement_irregular_2',
                    'determiner_noun_agreement_with_adj_2','superlative_quantifiers_2',
                    'determiner_noun_agreement_with_adj_irregular_2','determiner_noun_agreement_2',
                    'matrix_npi']

functions = ['no_function','more_function','five_function', 'natural_function', 'random_function','bigram_function','within_boundary']

def read_data(data_path):
    test_set = pd.read_json(data_path, lines=True).to_dict(orient='records')
    return test_set


def eval_sent_pair(ilm_model, tokenizer, lang,test_set):
    results = []
    correct=0
    e=0
    for sent in tqdm(test_set):
        ppls={}
        for k, v in sent.items():
            try:
                tokenized = tokenizer.encode(v,add_special_tokens=False,truncation=True,max_length=128)
                num_token= len(tokenized)

                v_trun = tokenizer.decode(tokenized)

                nll = ilm_model.sequence_score(v_trun,reduction=lambda x: -x.sum(0).item())[0]
                ppl = nll/num_token
                ppls[k]=ppl
            except IndexError:
                continue
        if len(list(ppls.keys()))<len(functions):
            e+=1
            print(e)
            continue
        best_lang = min(ppls, key=ppls.get)
        if best_lang == lang:
            correct += 1
        results.append(ppls)
    acc = correct/len(test_set)
    print(len(test_set))
    return results,acc


if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('lang',type=str, help='lang name')
    args = args.parse_args()
    os.makedirs(f'test_results', exist_ok=True)
    model_name = args.model_name
    test = read_data(f'test.jsonl')
    model_name_name = model_name.split('/')[-1]
    results = {}
    print(model_name)
    checkpoint='epoch-10'
    ilm_model = scorer.IncrementalLMScorer(model_name, 'cpu',revision=checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=checkpoint)
    ppl, acc = eval_sent_pair(ilm_model, tokenizer, args.lang, test)
    pd.DataFrame(ppl).to_csv(f'test_results/results_{model_name_name}_{checkpoint}.csv')
    with open('test_results.csv', 'a') as r:
        r.write(f'{args.lang}\t{acc}\n')