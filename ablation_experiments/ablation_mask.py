import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from glob import glob
import argparse
from pathlib import Path
from minicons import scorer

empty_categories = [
    "superlative_quantifiers_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "superlative_quantifiers_2",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_2",
    "matrix_npi",
]

DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because",
         "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has",
       "did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as",
       "over","than","during","about","between","among","except","through","around","after","like","off",
       "without","under","before","throughout","unlike","across","toward","along","above","aboard","until",
       "upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards",
       "underneath","alongside"]

FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP )
def build_function_token_spans(tokenizer, func_words):
    spans = set()
    for w in func_words:
        w = w.strip().lower()
        if not w:
            continue
        ids1 = tokenizer.encode(w, add_special_tokens=False)
        if ids1:
            spans.add(tuple(ids1))

        ids2 = tokenizer.encode(" " + w, add_special_tokens=False)
        if ids2:
            spans.add(tuple(ids2))

    spans = sorted(spans, key=len, reverse=True)
    return spans


def mark_spans(input_ids, spans):
    B, T = input_ids.shape
    mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)

    for b in range(B):
        seq = input_ids[b].tolist()
        for pat in spans:
            m = len(pat)
            if m == 0 or m > T:
                continue
            for i in range(T - m + 1):
                if tuple(seq[i:i+m]) == pat:
                    mask[b, i:i+m] = True
    return mask


def register_function_word_span_mask_hooks(model, tokenizer, func_words, mask_value=-1e4):
    hooks = []
    ctx = {"func_mask": None}
    spans = build_function_token_spans(tokenizer, func_words)
    print(f"[INFO] Built {len(spans)} unique function-word token spans.")

    def embed_pre_hook(module, args, kwargs):
        input_ids = args[0]  # (B, T)
        with torch.no_grad():
            func_mask = mark_spans(input_ids, spans)  # (B, T) bool
            # print('this is all func_mask ')
            # print(func_mask)
        ctx["func_mask"] = func_mask

        # if debug:
        #   ids0 = input_ids[0]
        #   m0 = func_mask[0]
        #   toks_all = tokenizer.convert_ids_to_tokens(ids0.tolist())
        #   toks_masked = [t for t, mm in zip(toks_all, m0.tolist()) if mm]
        #   print("[DEBUG] full:", toks_all[:80])
        #   print("[DEBUG] masked:", toks_masked[:80])

        return args, kwargs

    emb_hook = model.transformer.wte.register_forward_pre_hook(embed_pre_hook, with_kwargs=True)
    hooks.append(emb_hook)

    def make_attn_pre_hook():
        def attn_pre_hook(module, args, kwargs):
            attention_mask = kwargs.get("attention_mask", None)
            func_mask = ctx.get("func_mask", None)
            # #debug
            # print(func_mask)
            if attention_mask is None and len(args) >= 3:
                attention_mask = args[2]

            if func_mask is None:
                return args, kwargs

            fw_mask = func_mask.to(module.c_attn.weight.device).unsqueeze(1).unsqueeze(1)
            fw_mask = fw_mask.to(dtype=torch.float32) * mask_value  # additive mask

            if attention_mask is None:
                kwargs["attention_mask"] = fw_mask
            else:
                kwargs["attention_mask"] = attention_mask + fw_mask

            return args, kwargs
        return attn_pre_hook

    for block in model.transformer.h:
        h = block.attn.register_forward_pre_hook(make_attn_pre_hook(), with_kwargs=True)
        hooks.append(h)

    return hooks


def read_data(data_path):
    test_set = {}
    phenomenon_paths = glob(f"{data_path}/*.jsonl")
    for p in tqdm(phenomenon_paths):
        phenomenon_n = p.split("/")[-1].split(".")[0]
        if phenomenon_n in empty_categories or 'quantifier' in phenomenon_n:
            continue
        phenomenon = pd.read_json(p, lines=True).to_dict(orient="records")
        sent_pair = [(x["sentence_bad"], x["sentence_good"]) for x in phenomenon]
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
            num_token0 = len(tokenizer.encode(sent[0], add_special_tokens=False))
            num_token1 = len(tokenizer.encode(sent[1], add_special_tokens=False))
            nll0, nll1 = ilm_model.sequence_score(sent,reduction=lambda x: -x.sum(0).item())
            ppl0 = nll0 / num_token0
            ppl1 = nll1 / num_token1
            distribution.append(f"{sent[0]}\t{ppl0}\t{sent[1]}\t{ppl1}")
            if ppl0 > ppl1:
                correct += 1
        acc = correct / len(sents)
        results[phe] = acc
        distributions[phe] = "|||".join(distribution)
        print(phe, acc)
    return results, distributions


if __name__ == "__main__":
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('random_seed', type=int, help='random seed')
    args = args.parse_args()
    lang_name = args.model_name
    seed = args.random_seed
    for i in [10]:
        tokenizer = AutoTokenizer.from_pretrained(f"xiulinyang/GPT2_{lang_name}_{seed}", revision=f"epoch-{i}")
        model = AutoModelForCausalLM.from_pretrained(f"xiulinyang/GPT2_{lang_name}_{seed}", use_cache=False,attn_implementation="eager",revision=f"epoch-{i}")
        BLIMP_DIR = f"blimp/{lang_name}_blimp/"
        OUT_PREFIX = f"blimp_ablation_epoch{i}_fw_mask"
        os.makedirs(OUT_PREFIX, exist_ok=True)
        test_set = read_data(BLIMP_DIR)
        model.eval()
        all_pesudo_words = []
        if 'more_function' in lang_name:
            print('this is more function!')
            pesudo_words = Path('function_word_pseudowords.txt').read_text().strip().split('\n')
            for line in pesudo_words:
                word, pseudo = line.strip().split('\t')
                all_pesudo_words.append(pseudo)
        func_l = set(DET + CCONJ + SCONJ + AUX + ADP + all_pesudo_words)
        print(len(func_l))
        hooks = register_function_word_span_mask_hooks(model, tokenizer, func_l)
        ilm_model = scorer.IncrementalLMScorer(model, device="cuda", tokenizer=tokenizer)
        results = {}
        acc, dist = eval_sent_pair(ilm_model, tokenizer, test_set)
        results[f"epoch-{i}"] = acc
        pd.DataFrame(results).to_csv(f"{OUT_PREFIX}/results_GPT2_{lang_name}_{seed}_epoch-{i}.csv")
        for h in hooks:
            h.remove()