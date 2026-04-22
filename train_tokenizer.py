import argparse, json, os, pathlib
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast
from clm.tokenization.byte_level_bpe import ByteLevelBPETokenizer
from clm.tokenization.sentencepiece_unigram import SentencePieceUnigramTokenizer
from tokenizers import Tokenizer
from clm.utils import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from tokenizers import AddedToken
import pyarrow.compute as pc
import random
from pathlib import Path
random.seed(42)
SPECIALS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because", "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has","did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as","over","than","during","about","between","among","except","through","around","after","like","off","without","under","before","throughout","unlike","across","toward","along","above","aboard","until","upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards","underneath","alongside"]
FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP)

pesudo_class = {x:[x] for x in FUNCTION_WORDS}
pesudo_words = Path('function_word_pseudowords.txt').read_text().strip().split('\n')
all_pesudo_words = []
for line in pesudo_words:
    word, pseudo = line.strip().split('\t')
    all_pesudo_words.append(pseudo)

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def load_json_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_fast_tokenizer(tok_file: str):
    print(tok_file)
    return PreTrainedTokenizerFast(
        tokenizer_file=tok_file,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        bos_token=BOS_TOKEN,
        add_prefix_space=True,
    )

def main(args):
    cfg = load_json_cfg(args.config)
    tok_cfg = cfg['tokenizer']
    model_name = tok_cfg['model_name']
    raw_data = cfg['tokenizer']['raw_data']
    data_name = cfg['tokenizer']['data_name']
    save_dir = tok_cfg['save_dir']
    print(save_dir)
    tok_train_file = tok_cfg['train_file']
    ensure_dir(save_dir)


    if not os.path.exists(tok_train_file):
        raise FileNotFoundError(f"train_file not found: {tok_train_file}")

    special_tokens = list(SPECIALS)
    # if data_name == 'more_function':
    #
    #     special_tokens += list(FUNCTION_WORDS)
    #     special_tokens += all_pesudo_words
    #     print(special_tokens)

    base_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    base_tokenizer.train(
        files=tok_train_file,
        vocab_size=32768,
        special_tokens=special_tokens,
    )
    tok_json_path = os.path.join(save_dir, "tokenizer.json")
    base_tokenizer.save(tok_json_path)
    base_tokenizer.save_model(save_dir)
    tokenizer = build_fast_tokenizer(tok_json_path)
    eos_id = tokenizer.eos_token_id

    def encode(ex):
        ids = tokenizer(ex["text"]).input_ids
        ids.append(eos_id)
        return {"input_ids": ids}

    if len(tokenizer) != 32768:
        raise ValueError(f"Vocab mismatch: tokenizer len={len(tokenizer)} vs config vocab_size=32768")

    files = {}
    if tok_cfg.get("train_file"): files["train"] = tok_cfg["train_file"]
    if tok_cfg.get("validation_file"): files["validation"] = tok_cfg["validation_file"]

    raw_all = load_dataset("text", data_files=files, keep_linebreaks=True)
    encoded = raw_all.map(encode, batched=False, remove_columns=raw_all["train"].column_names, num_proc=20, desc="Encoding to IDs")
    tbl = encoded["train"].data
    arr = tbl.column("input_ids")
    total_tokens = pc.sum(pc.list_value_length(arr)).as_py()
    print(total_tokens)
    print('total tokens for the training split: ', total_tokens)
    tok_ds_dir = os.path.join(f'{raw_data}', model_name)
    ensure_dir(tok_ds_dir)
    encoded.save_to_disk(tok_ds_dir)
    tokenizer.save_pretrained(save_dir)
    print(f" Tokenizer + tokenized dataset saved under: {save_dir} and {raw_data}")
    print(f" - tokenizer.json: {tok_json_path}")
    print(f" - tokenized ds:   {tok_ds_dir}")
    ds = load_from_disk(tok_ds_dir)
    print("Example:", ds["train"][0]["input_ids"][:50])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to JSON config generated earlier")
    args = ap.parse_args()
    main(args)
