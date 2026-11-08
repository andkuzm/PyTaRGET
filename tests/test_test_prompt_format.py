"""
A model must be asked at test time in the format it was trained on: Tester builds the prompt of a test case with the
dataset class of the model (<NL>/<TAB> tokens, PLBART's "</s>__python__" suffix), not from the raw input text.
Found in the first smoke test: raw prompts made a trained CodeGen answer with an immediate end-of-text.
"""
import json
import shutil

import pandas as pd
import pytest

import tiny_eval_fixtures as fixtures
from data_processing.dataset_types import CodeGenDataset, EncDecDataset, PLBARTDataset
from data_processing.encode import Encoder
from data_processing.encode_tune_test import Tokens
from data_processing.test import Tester
from transformers import AutoTokenizer, GPT2LMHeadModel, T5ForConditionalGeneration

torch = fixtures.torch
CASES = 3


@pytest.fixture(autouse=True)
def cheap_code_bleu(monkeypatch):
    monkeypatch.setattr("data_processing.test.calc_code_bleu", lambda *args, **kwargs: 0.5)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompt-tokenizer")
    fixtures.build_tokenizer().save_pretrained(path)
    return AutoTokenizer.from_pretrained(path)


@pytest.mark.parametrize("name,dataset_class,model_class,decoder_only", [
    ("codet5p", EncDecDataset, T5ForConditionalGeneration, False),
    ("plbart", PLBARTDataset, T5ForConditionalGeneration, False),
    ("codegen", CodeGenDataset, GPT2LMHeadModel, True),
])
def test_the_prompt_given_to_generate_is_the_training_prompt(
        name, dataset_class, model_class, decoder_only, tokenizer, tmp_path, monkeypatch):
    encoded = tmp_path / "enc"
    Encoder(fixtures.TINY_CSV, encoded, 0.5, tokenizer, dataset_class, name, False, Tokens).encode()
    experiment = tmp_path / "out" / name / "0.5"
    shutil.copytree(encoded / name / "0.5" / "splits", experiment / "splits")
    checkpoint = (fixtures.tiny_gpt2 if decoder_only else fixtures.tiny_t5)(tokenizer, experiment / "checkpoint-best")
    rows = json.loads((experiment / "splits" / "test.json").read_text(encoding="utf-8"))[:CASES]

    # what the model saw in training for the same cases
    training = dataset_class(pd.DataFrame(rows), tokenizer, "reference", tmp_path / "reference")
    assert any(("\n" in r["input"]) for r in rows)      # otherwise raw text and <NL> text could not differ

    seen = []
    original = Tester.new_tokens

    def spy(model, outputs, input_ids):
        seen.append(input_ids.squeeze(0).cpu())
        return original(model, outputs, input_ids)
    monkeypatch.setattr(Tester, "new_tokens", staticmethod(spy))
    monkeypatch.setattr(tokenizer, "model_max_length", 1100 if decoder_only else 64)
    Tester(name, model_class, tokenizer, tmp_path / "out", 1, 0.5, dataset_class).validate(limit=CASES)

    assert len(seen) == CASES
    nl, tab = tokenizer.convert_tokens_to_ids(["<NL>", "<TAB>"])
    for prompt, item in zip(seen, training.data):
        trained = item["input_ids"]
        if decoder_only:        # training text = prompt + repair + end-of-text
            trained = trained[:len(prompt)]
        assert prompt.tolist() == trained.tolist()
        assert nl in prompt.tolist()


# ---- decoding: PLBART's tokenizer puts spaces around <NL>/<TAB> ----------------------------------------------------

def test_spaces_that_decoding_adds_around_the_markers_are_removed():
    from data_processing.test import restore_markers
    decoded = "<NL> <TAB> <TAB> result = f(x) <NL> <TAB> <TAB> return result"
    text = restore_markers(decoded).replace("<TAB>", "    ").replace("<NL>", "\n")
    assert text == "\n        result = f(x)\n        return result"
    # an odd indentation stays what the model wrote: a real space is a second space after the marker
    assert restore_markers("<NL> <TAB>  x").replace("<TAB>", "    ").replace("<NL>", "\n") == "\n     x"


def test_only_tokenizers_that_add_the_spaces_are_corrected():
    from data_processing.test import marker_spaces_added_by_decode

    class Spacing:
        def encode(self, text, add_special_tokens=False): return [1]
        def decode(self, ids, skip_special_tokens=True): return "a <NL> <TAB> b"

    class Plain(Spacing):
        def decode(self, ids, skip_special_tokens=True): return "a<NL><TAB>b"
    assert marker_spaces_added_by_decode(Spacing()) and not marker_spaces_added_by_decode(Plain())
