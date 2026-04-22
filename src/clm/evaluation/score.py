from minicons import scorer
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import os

HF_REPO = 'parallelm'
CHECKPOINTS= range(1,11)

def read_data(lang):
    lang_lower = lang.lower()
    sentences = pd.read_csv(f'para-multi-blimp/{lang_lower}_multiblimp.tsv', sep='\t').to_dict(orient='records')
    sent_pair = [[sent['sent_correct'], sent['sent_wrong']] for sent in sentences if not pd.isna(sent['sent_wrong'])]
    return sent_pair

def eval_sent_pair(ilm_model, tokenizer, sent_pair):
    correct = 0
    for sent in tqdm(sent_pair):
        # distribution = []
        num_token0 = len(tokenizer.encode(sent[0],add_special_tokens=False))
        num_token1 = len(tokenizer.encode(sent[1],add_special_tokens=False))
        nll0, nll1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
        ppl0 = nll0/num_token0
        ppl1 = nll1/num_token1
        # distribution.append([(0, ppl0), (1, ppl1)])
        if ppl0 < ppl1:
            correct+=1
    acc = correct/len(sent_pair)
    return acc



if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('lang', type=str, help='language')
    args = args.parse_args()
    result_dir = 'multiblimp_results'
    os.makedirs(f'{result_dir}', exist_ok=True)
    model_name = args.model_name
    lang = args.lang
    test = read_data(lang)

    f_results = {}
    tokenizer = AutoTokenizer.from_pretrained(f'{HF_REPO}/{model_name}')
    for checkpoint in tqdm(CHECKPOINTS):
        revision = f"checkpoint-{checkpoint}"
        print(model_name, checkpoint)
        ilm_model = scorer.IncrementalLMScorer(f'{HF_REPO}/{model_name}', 'cuda',revision=revision)

        acc = eval_sent_pair(ilm_model, tokenizer, test)
        f_results[checkpoint] = acc
    df = pd.DataFrame(
        [{"checkpoint": f'ckpt-{ckpt}', "accuracy": acc} for ckpt, acc in f_results.items()]
    )
    df.to_csv(f'{result_dir}/results_{model_name}.csv')