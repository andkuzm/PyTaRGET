"""
Helpers for the resumability tests of the evaluation pipeline (train / test / LLM test).

Everything is tiny and offline: the data are 28 real cases (tests/data/tiny_cases.csv, 7 repositories, 7 commits),
the tokenizer is a word-level tokenizer trained on that file, the models are randomly initialised T5 / GPT-2 models
with a few thousand parameters. Nothing is downloaded and no GPU is used, so the whole pipeline
(encode -> train -> predict) runs in seconds and the runs are deterministic.
"""
import csv
import inspect
import sys
from pathlib import Path

import pytest

TINY_CSV = Path(__file__).parent / "data" / "tiny_cases.csv"

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers, trainers          # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, T5Config,   # noqa: E402
                          T5ForConditionalGeneration)

from data_processing.dataset_types import EncDecDataset, LLMSeqDataset      # noqa: E402
from data_processing.encode import Encoder                                  # noqa: E402
from data_processing.encode_tune_test import Tokens                        # noqa: E402

csv.field_size_limit(sys.maxsize)


def case_texts(path=TINY_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            yield row["annotated_code"]
            yield row["outdated_test_log"]


def build_tokenizer(model_max_length=1024):
    """A word-level tokenizer over the tiny data, with the special tokens Eftt.create_tokenizer adds."""
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(case_texts(), trainers.WordLevelTrainer(
        vocab_size=3000, special_tokens=["<pad>", "</s>", "<unk>", "<s>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>",
                                   unk_token="<unk>", bos_token="<s>")
    fast.model_max_length = model_max_length
    fast.add_special_tokens({"additional_special_tokens": [v for k, v in inspect.getmembers(Tokens) if not k.startswith("_")]})
    fast.add_tokens(["<TAB>", "<NL>"])
    return fast


def tiny_t5(tokenizer, directory):
    """A randomly initialised T5 (about 70k parameters) saved to `directory`; returns the path."""
    torch.manual_seed(0)
    config = T5Config(vocab_size=len(tokenizer), d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1,
                      num_heads=4, relative_attention_num_buckets=8, decoder_start_token_id=tokenizer.pad_token_id,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    T5ForConditionalGeneration(config).save_pretrained(directory)
    return directory


def tiny_gpt2(tokenizer, directory):
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=4096, n_embd=32, n_layer=1, n_head=2,
                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    GPT2LMHeadModel(config).save_pretrained(directory)
    return directory


def encode(out_path, tokenizer, model="codet5p", train_size=0.5, llm=False):
    """Run the real Encoder over the tiny data: writes train/valid/test .pkl and splits/*.json."""
    dataset_class = LLMSeqDataset if llm else EncDecDataset
    Encoder(TINY_CSV, Path(out_path), train_size, tokenizer, dataset_class, model, llm, Tokens).encode()
    return Path(out_path) / model / str(train_size)
