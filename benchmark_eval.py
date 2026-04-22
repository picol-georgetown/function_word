from minicons import scorer
import argparse
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from glob import glob
from pathlib import Path
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

def eval_sent_pair(ilm_model, tokenizer, test_set):
    results = {}
    distributions = {}
    for phe, sents in tqdm(test_set.items()):
        correct = 0
        distribution = []
        for sent in sents:
            sent = list(sent)
            num_token0 = len(tokenizer.encode(sent[0],add_special_tokens=False))
            num_token1 = len(tokenizer.encode(sent[1],add_special_tokens=False))
            nll0, nll1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
            ppl0 = nll0/num_token0
            ppl1 = nll1/num_token1
            distribution.append(f'{sent[0]}\t{ppl0}\t{sent[1]}\t{ppl1}')
            if ppl0 > ppl1:
                correct+=1
        acc = correct/len(sents)
        results[phe] = acc
        distributions[phe] = '|||'.join(distribution)
        print(phe, acc)
    return results, distributions



if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('eval_dataset', type=str, help='dataset name')
    args.add_argument('--best_checkpoint', action='store_true')

    args = args.parse_args()
    dataset = args.eval_dataset
    os.makedirs(f'{dataset}_results', exist_ok=True)
    model_name = args.model_name
    best_checkpoint = args.best_checkpoint
    refs = list_repo_refs(model_name, repo_type="model")
    num_checkpoints = refs.branches
    checkpoints = range(2,11)
    test = read_data(f'blimp/blimp/{dataset}')

    model_name_name = model_name.split('/')[-1]
    f_results = {}
    if best_checkpoint:
        print(model_name)
        ilm_model = scorer.IncrementalLMScorer(model_name, 'cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        acc, dist = eval_sent_pair(ilm_model, tokenizer, test)
        f_results['best'] = acc
        pd.DataFrame(f_results).to_csv(f'{dataset}_results/results_{model_name_name}_best.csv')
        df_dist = pd.DataFrame.from_dict(dist, orient='index', columns=['distribution'])
        df_dist.index.name = 'phenomenon'
        df_dist.to_csv(f'{dataset}_results/distributions_{model_name_name}_best.csv')
    else:

        for checkpoint in checkpoints:
            checkpoint ='epoch-'+str(checkpoint)
            results = {}
            print(model_name, checkpoint)
            ilm_model = scorer.IncrementalLMScorer(model_name, 'cuda',revision=checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            acc, dist = eval_sent_pair(ilm_model, tokenizer, test)
            results[checkpoint] = acc
            pd.DataFrame(results).to_csv(f'{dataset}_results/results_{model_name_name}_{checkpoint}.csv')
            df_dist = pd.DataFrame.from_dict(dist, orient='index', columns=['distribution'])
            df_dist.index.name = 'phenomenon'
            df_dist.to_csv(f'{dataset}_results/distributions_{checkpoint}.csv')