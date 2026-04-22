
from clm.utils.tokenizer_and_config import load_config, autoreg_config
import argparse
import pathlib
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoTokenizer
import argparse, json, os, pathlib
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from clm.utils.tokenizer_and_config import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN


def load_json_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fast_tokenizer(tok_file: str):
    return PreTrainedTokenizerFast(
        tokenizer_file=tok_file,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        bos_token=BOS_TOKEN,
        add_prefix_space=True
    )


def main(args):
    model_cfg = load_json_cfg(args.config)
    tok_cfg = load_json_cfg(args.config)['tokenizer']
    model_cfg = model_cfg['model']
    hidden_size = model_cfg['n_embd']
    n_head = model_cfg['n_heads']
    n_layer = model_cfg['num_layer']
    max_len = model_cfg['n_positions']

    model_name = tok_cfg.get('model_name')
    model_type = model_cfg.get('model_type')
    tokenizer = tok_cfg.get('save_dir')+'/tokenizer.json'

    tokenizer = build_fast_tokenizer(tokenizer)
    vocab = len(tokenizer)
    cfg = autoreg_config(
        model_type,
        model_name,
        tokenizer,
        vocab,
        hidden_size,
        n_head,
        n_layer,
        max_len,
    )

    pathlib.Path(f"models/{model_name}").mkdir(parents=True, exist_ok=True)
    cfg._name_or_path = model_name
    cfg.save_pretrained(f"models/{model_name}")
    print(f'Saved model config to models/{model_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    args = parser.parse_args()
    main(args)