import yaml
from transformers import AutoConfig, AutoTokenizer
from clm.models import RNNConfig, RNNForLanguageModeling


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|endoftext|>"

def load_config(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must be .yaml/.yml or .json")


def autoreg_config(
    model_type,
    model_name,
    tokenizer,
    vocab,
    hidden_size,
    attention,
    layers,
    max_len=512,
    dropout_p = 0.1
):

    if model_type.lower() in ["rnn", 'lstm', 'gru']:
        return RNNConfig(
                rnn_type = model_type.upper(),
                vocab_size=vocab,
                embedding_dim=hidden_size,
                hidden_dim=hidden_size,
                num_layers=layers,
                dropout_p=dropout_p,
                tie_weights=False,
                bidirectional=False,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            n_ctx=max_len,
                name_or_path=model_name,
            )
    elif model_type.lower() =='gpt2':
        return AutoConfig.from_pretrained(
            model_type,
            name_or_path=model_name,
            vocab_size=vocab,
            hidden_size=hidden_size,
            num_attention_heads=attention,
            num_hidden_layers=layers,
            max_position_embeddings=max_len,
            word_embed_proj_dim=hidden_size,
            intermediate_size=hidden_size*4,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    else:
        raise ValueError(f'{model_type} is not supported!')



