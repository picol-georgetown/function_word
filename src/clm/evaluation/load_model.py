import tempfile
from typing import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from minicons import scorer
import torch


# Valid script types and model sizes
SCRIPTS = [
    "arab", "beng", "cyrl", "ethi", "tibt", "thaa", "grek", "hebr", "deva", "armn", "cans", "latn"
]
SIZES = ["5mb", "10mb", "100mb", "1000mb"]



def load_hf_model(model_name: str, no_cache=False, revision_step="main", tokenizer_revision_step="main",**kwargs):
    """
    Load a HuggingFace model and tokenizer, optionally using a cache and a specific revision.
    No need for token if the model is public.
    """
    model = None
    tokenizer = None

    try:
        if no_cache:
            with tempfile.TemporaryDirectory() as tmpdirname:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=tmpdirname,
                    revision=revision_step,
                    **kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=tmpdirname,
                    revision=tokenizer_revision_step,
                    **kwargs
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision_step,
                **kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=tokenizer_revision_step,
                **kwargs
            )
    except OSError as e:
        print(f"Failed to load model '{model_name}' @ revision '{revision_step}'")
        print(f"Error: {e}")
        return None

    if model is None or tokenizer is None:
        print(f"Model or tokenizer not loaded properly for: {model_name}")
        return None

    print(f"Successfully loaded model '{model_name}' @ revision '{revision_step}'")

    # Updated to support MPS on Mac
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        
    ilm_model = scorer.IncrementalLMScorer(
        model,
        device=device,
        tokenizer=tokenizer,
    )

    return ilm_model


def load_goldfish_model(langcode: str, size: str, script: Optional[str] = None, no_cache=False, **kwargs):
    if script is not None:
        return load_goldfish(langcode, script, size, no_cache=no_cache)
    else:
        model = None

        for script in SCRIPTS:
            model_name = f"goldfish-models/{langcode}_{script}_{size}"
            model = load_hf_model(model_name, no_cache=no_cache, **kwargs)
            if model is not None:
                break

        return model
