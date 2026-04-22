# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation

This script has been slightly modified by Kanishka Misra and Najoung Kim, and later Xiulin Yang for customized training.
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
from datasets import load_from_disk
import evaluate
import torch
from datasets import load_dataset
import json
import numpy as np
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
)
from typing import List
# from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.34.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class SaveAtStepsCallback(TrainerCallback):
    """Custom callback to save model checkpoints at specific training steps."""

    def __init__(self, save_steps: List[int], output_dir: str):
        """
        Args:
            save_steps: List of global steps at which to save the model.
            output_dir: Base directory for saving checkpoints.
        """
        self.save_steps = sorted(save_steps)
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        if 0 in self.save_steps:
            ckpt_dir = os.path.join(self.output_dir, "checkpoint-0")
            os.makedirs(ckpt_dir, exist_ok=True)
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            if model is not None:
                model.save_pretrained(ckpt_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved initial model at step 0 → {ckpt_dir}")
            self.save_steps.remove(0)

    def on_step_end(self, args, state, control, **kwargs):
        """
        This hook runs after every optimizer step.
        """
        model = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)

        if model is None:
            return control  # no model to save

        if state.global_step in self.save_steps:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            model.save_pretrained(checkpoint_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)

            print(f"Saved model at step {state.global_step} → {checkpoint_dir}")

        return control


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which models/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The models checkpoint for weights initialization. Don't set if you want to train a models from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a models type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a models is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific models version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    add_prefix_space: Optional[bool] = field(
        default=False, metadata={"help": "whether to add a prefix space."}
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the models under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the models's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the models as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our models for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    pretokenized_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "the path of the pretokenized file"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(
        default=False, metadata={"help": "Enable streaming mode"}
    )
    block_size: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the models max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={
            "help": "Whether to keep line breaks when using TXT files or not."
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0",
                "The streaming feature requires `datasets>=2.0.0`",
            )

        if (
            self.pretokenized_file_path is None
            and self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either pretokenized dir, a dataset name, or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                    "train",
                    "sents",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                    "dev",
                    "sents",
                ], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1])) as f:
            cfg = json.load(f)

        flat = {}
        flat.update(cfg["model"])
        flat.update(cfg["data"])
        flat.update(cfg["training"])

        print(flat)

        model_args, data_args, training_args = parser.parse_dict(flat, True)
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing models.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.pretokenized_file_path:
        raw_datasets = load_from_disk(data_args.pretokenized_file_path)
    else:
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                # token=model_args.token,
                streaming=data_args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    # token=model_args.token,
                    streaming=data_args.streaming,
                )
        else:
            data_files = {}
            dataset_args = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file

            print(f"DATA FILES: {data_files}")

            extension = (
                data_args.train_file.split(".")[-1]
                if data_args.train_file is not None
                else data_args.validation_file.split(".")[-1]
            )
            if extension == "txt" or extension == "train" or extension == "sents":
                extension = "text"
                dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                # token=model_args.token,
                **dataset_args,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained models and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download models & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch."
        )
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "add_prefix_space": model_args.add_prefix_space,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    for name in ["eos_token_id", "unk_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, name, None)
        if tid is not None and not (0 <= tid < len(tokenizer)):
            raise ValueError(f"{name} out of range.")

    vocab_size = len(tokenizer)
    str_vocab_size = str(vocab_size)
    print(f'=================tokenizer vocab size: {str_vocab_size}==========================')
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        config.vocab_size = vocab_size
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=model_args.trust_remote_code
        )
        n_params = sum(
            {p.data_ptr(): p.numel() for p in model.parameters()}.values()
        )
        logger.info(
            f"Training new models from scratch - Total size={n_params/2**20:.2f}M params"
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a models from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     models.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        # with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        # if "Token indices sequence length is longer than the" in cl.out:
        #     tok_logger.warning(
        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
        #         " before being passed to the models."
        #     )
        return output

    vocab_size = len(tokenizer)
    if model.get_input_embeddings().weight.shape[0] != vocab_size:
        print('model vocab size: ', model.get_input_embeddings().weight.shape[0])
        print('tokenizer vocab size: ', vocab_size)
        raise ValueError('The embedding of the model and the vocab size do not match!')


    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the models"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        if data_args.pretokenized_file_path:
            concatenated = list(chain(*examples["input_ids"]))
            total_length = (len(concatenated) // block_size) * block_size
            concatenated = concatenated[:total_length]
            result = {
                "input_ids": [concatenated[i:i + block_size] for i in range(0, total_length, block_size)]
            }
        else:
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        result["labels"] = result["input_ids"].copy()
        return result


    if data_args.pretokenized_file_path:
        with training_args.main_process_first(desc="grouping tokenized ids"):
            lm_datasets = raw_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc=f"Grouping input_ids in chunks of {block_size}",
            )
    else:
        with training_args.main_process_first(desc="dataset map tokenization"):
            # tokenization first
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
        # concat the tokenized data
        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    if training_args.do_train:
        if data_args.pretokenized_file_path:
            if 'train' not in lm_datasets:
                raise ValueError("--do_train requires a train dataset")
        else:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if data_args.pretokenized_file_path:
            if 'validation' not in lm_datasets:
                raise ValueError("--do_eval requires a validation dataset")
        else:
            if "validation" not in tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the models and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

        # def compute_metrics(eval_pred):
        #     logits, labels = eval_pred
        #     logits = torch.tensor(logits)
        #     labels = torch.tensor(labels)
        #
        #     # Shift for next-token prediction
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #
        #     mask = shift_labels.ne(-100)
        #     loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        #
        #     # Per-token NLL in nats
        #     nll = loss_fct(
        #         shift_logits.view(-1, shift_logits.size(-1)),
        #         shift_labels.view(-1)
        #     ).view(shift_labels.size())
        #
        #     nll = nll * mask
        #
        #     # Sequence-level NLL: sum over tokens, not divided by length
        #     seq_nll = nll.sum(dim=1)  # one scalar per sequence
        #
        #     # Report both mean and median across sequences
        #     return {
        #         "seq_nll": seq_nll.mean().item()
        #     }
    callback = SaveAtStepsCallback(
        save_steps=list(range(0, 4000, 1000)) + list(range(4000, 100000, 4000)) + list(range(100000, 300000, 10000)),
        output_dir=training_args.output_dir,
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
        if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
        # callbacks=[callback],
        # callbacks=[EarlyStoppingCallback(
        #     early_stopping_patience=3,  # stop after 3 evals with no improvement
        #     early_stopping_threshold=0.0  # require strictly better (set ~1e-4 if metric jitters)
        # )]

    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()