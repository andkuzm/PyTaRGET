"""
Resumability of the prediction step (Tester for the fine-tuned models, Tester_llm for the instruction-tuned ones) on
tiny random models and 28 real cases (see tiny_eval_fixtures.py). Includes a real SIGKILL of a subprocess, which is how
Slurm ends a job that ignored the time limit.
"""
import json
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import tiny_eval_fixtures as fixtures
from data_processing import encode_tune_test, prediction_files
from data_processing.encode_tune_test import Eftt
from data_processing.dataset_types import PLBARTDataset
from data_processing.llm_test import Tester_llm
from data_processing.test import Tester
from transformers import AutoTokenizer, T5ForConditionalGeneration

torch = fixtures.torch
ROOT = Path(__file__).resolve().parent.parent
GENERATED_TOKENS = 12           # keeps the random models' babble short


def raise_keyboard_interrupt():
    raise KeyboardInterrupt


@pytest.fixture(autouse=True)
def cheap_code_bleu(monkeypatch):
    """CodeBLEU needs tree-sitter >= 0.23 (the README pins 0.24.0); these tests are about saving and resuming
    predictions, not about the metric, so the call is replaced and every other metric stays real."""
    monkeypatch.setattr("data_processing.test.calc_code_bleu", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("data_processing.llm_test.calc_code_bleu", lambda *args, **kwargs: 0.5)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-testing")
    tokenizer = fixtures.build_tokenizer()
    tokenizer_dir = root / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return {
        "tokenizer": tokenizer, "tokenizer_dir": tokenizer_dir,
        "seq2seq": fixtures.encode(root / "seq2seq-encoded", tokenizer, model="codet5p", train_size=0.5),
        "llm": fixtures.encode(root / "llm-encoded", tokenizer, model="qwen", train_size=0.0, llm=True),
        "llm_cases": 28,                  # train_size 0.0: every case of the tiny data is a test case
        "t5": fixtures.tiny_t5(tokenizer, root / "t5"),
        "gpt2": fixtures.tiny_gpt2(tokenizer, root / "gpt2"),
    }


# ---- fine-tuned models: Tester ---------------------------------------------------------------------------------------

@pytest.fixture
def seq2seq(tiny, tmp_path, monkeypatch):
    """A private experiment directory with the encoded test split and a (random) best checkpoint."""
    monkeypatch.setattr(tiny["tokenizer"], "model_max_length", 64)
    experiment = tmp_path / "out" / "codet5p" / "0.5"
    shutil.copytree(tiny["seq2seq"] / "splits", experiment / "splits")
    shutil.copytree(tiny["t5"], experiment / "checkpoint-best")
    tester = Tester("codet5p", T5ForConditionalGeneration, tiny["tokenizer"], tmp_path / "out", 2, 0.5)
    return {"tester": tester, "experiment": experiment, "file": experiment / "test_predictions.json"}


def interrupt_after(monkeypatch, cases):
    """Raise KeyboardInterrupt in the (cases + 1)-th generated case, i.e. after `cases` finished ones."""
    original = Tester.new_tokens
    calls = {"n": 0}

    def counting(model, outputs, input_ids):
        calls["n"] += 1
        if calls["n"] == cases + 1:
            raise KeyboardInterrupt
        return original(model, outputs, input_ids)
    monkeypatch.setattr(Tester, "new_tokens", staticmethod(counting))


@pytest.fixture
def reference(tiny, tmp_path_factory, monkeypatch):
    """The predictions of an uninterrupted run."""
    monkeypatch.setattr(tiny["tokenizer"], "model_max_length", 64)
    root = tmp_path_factory.mktemp("seq2seq-reference")
    experiment = root / "out" / "codet5p" / "0.5"
    shutil.copytree(tiny["seq2seq"] / "splits", experiment / "splits")
    shutil.copytree(tiny["t5"], experiment / "checkpoint-best")
    Tester("codet5p", T5ForConditionalGeneration, tiny["tokenizer"], root / "out", 2, 0.5).validate(save_every=5)
    return prediction_files.load(experiment / "test_predictions.json")


class TestSeq2SeqTester:
    def test_a_full_run_predicts_every_case_and_reports_the_scores(self, seq2seq, capsys):
        scores = seq2seq["tester"].validate(save_every=5)
        saved = prediction_files.load(seq2seq["file"])
        assert [r["ID"] for r in saved] == list(range(12))
        assert all(len(r["preds"]) == 2 for r in saved)                    # beam size 2
        assert len(scores) == 4
        assert "over 12/12 cases" in capsys.readouterr().out

    def test_interrupted_run_keeps_its_predictions_and_resumes_to_the_same_result(self, seq2seq, monkeypatch, reference):
        with monkeypatch.context() as m:
            interrupt_after(m, 5)
            with pytest.raises(KeyboardInterrupt):
                seq2seq["tester"].validate(save_every=100)              # no periodic save: the interrupt must save
        assert [r["ID"] for r in prediction_files.load(seq2seq["file"])] == [0, 1, 2, 3, 4]

        generated = []
        original = Tester.new_tokens
        monkeypatch.setattr(Tester, "new_tokens", staticmethod(lambda m, o, i: generated.append(1) or original(m, o, i)))
        seq2seq["tester"].validate(save_every=100)
        assert len(generated) == 7                                         # only the missing cases were generated
        assert prediction_files.load(seq2seq["file"]) == reference

    def test_periodic_saves_survive_repeated_interruptions(self, seq2seq, monkeypatch, reference):
        for stop_after in (3, 3, 3):                                       # three runs of 3 new cases each
            with monkeypatch.context() as m:
                interrupt_after(m, stop_after)
                with pytest.raises(KeyboardInterrupt):
                    seq2seq["tester"].validate(save_every=2)
        assert len(prediction_files.load(seq2seq["file"])) == 9
        seq2seq["tester"].validate(save_every=2)
        assert prediction_files.load(seq2seq["file"]) == reference

    def test_metrics_of_a_partial_file_say_so(self, seq2seq, monkeypatch, capsys):
        with monkeypatch.context() as m:
            interrupt_after(m, 4)
            with pytest.raises(KeyboardInterrupt):
                seq2seq["tester"].validate()
        eftt = Eftt.__new__(Eftt)
        eftt.out_path, eftt.model, eftt.train_size = seq2seq["experiment"].parents[1], "codet5p", 0.5
        eftt._check_complete(prediction_files.load(seq2seq["file"]))
        out = capsys.readouterr().out
        assert "over 4/12 cases" in out and "WARNING" in out and "incomplete" in out

    def test_predictions_of_another_test_split_are_refused_and_untouched(self, seq2seq):
        stale = [{"ID": 0, "target": "something else entirely", "preds": ["x"]}]
        prediction_files.save(stale, seq2seq["file"])
        with pytest.raises(ValueError, match="do not belong"):
            seq2seq["tester"].validate()
        assert prediction_files.load(seq2seq["file"]) == stale

    def test_a_damaged_predictions_file_is_not_silently_replaced(self, seq2seq):
        seq2seq["file"].write_text('{"ID": {"0": 0, "1": 1}, "target": {"0": "a')
        with pytest.raises(RuntimeError, match="damaged"):
            seq2seq["tester"].validate()
        assert seq2seq["file"].read_text().startswith('{"ID"')

    def test_limit_runs_only_the_first_cases_and_a_later_full_run_continues(self, seq2seq, capsys, reference):
        seq2seq["tester"].validate(limit=3)
        assert [r["ID"] for r in prediction_files.load(seq2seq["file"])] == [0, 1, 2]
        assert "over 3/3 cases" in capsys.readouterr().out
        seq2seq["tester"].validate()
        assert prediction_files.load(seq2seq["file"]) == reference

    def test_resume_false_starts_over(self, seq2seq):
        prediction_files.save([{"ID": 0, "target": "x", "preds": ["y"]}], seq2seq["file"])
        seq2seq["tester"].validate(resume=False)
        assert [r["ID"] for r in prediction_files.load(seq2seq["file"])] == list(range(12))

    def test_a_real_sigkill_leaves_a_readable_file_and_the_run_continues(self, seq2seq, tiny, reference):
        """Run the Tester in a subprocess, kill -9 it after its first saves, then continue in this process."""
        script = Path(seq2seq["experiment"]).parent / "slow_run.py"
        script.write_text(textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            from data_processing.test import Tester
            tokenizer = AutoTokenizer.from_pretrained({str(tiny['tokenizer_dir'])!r})
            tokenizer.model_max_length = 64
            original = Tester.new_tokens
            Tester.new_tokens = staticmethod(lambda m, o, i: (time.sleep(0.5), original(m, o, i))[1])
            Tester("codet5p", T5ForConditionalGeneration, tokenizer, {str(seq2seq['experiment'].parents[1])!r}, 2, 0.5).validate(save_every=2)
        """))
        errors = script.with_suffix(".err")
        process = subprocess.Popen([sys.executable, "-I", str(script)], stdout=subprocess.DEVNULL, stderr=open(errors, "w"))
        try:
            deadline = time.time() + 180
            while time.time() < deadline and process.poll() is None:
                if seq2seq["file"].exists() and len(prediction_files.load(seq2seq["file"])) >= 4:
                    break
                time.sleep(0.2)
            assert process.poll() is None, "the subprocess ended before it could be killed: " + errors.read_text()[-1500:]
            process.send_signal(signal.SIGKILL)
        finally:
            process.kill()
            process.wait()
        saved = prediction_files.load(seq2seq["file"])                     # never a truncated file
        assert 4 <= len(saved) < 12
        seq2seq["tester"].validate(save_every=2)
        assert prediction_files.load(seq2seq["file"]) == reference


# ---- instruction-tuned models: Tester_llm ----------------------------------------------------------------------------

@pytest.fixture
def llm(tiny, tmp_path):
    experiment = tmp_path / "out" / "qwen" / "0.0"
    shutil.copytree(tiny["llm"] / "splits", experiment / "splits")

    def make(batch_size=4, generation=None):
        return Tester_llm("qwen", str(tiny["gpt2"]), experiment, None, tiny["tokenizer"], 0.0, batch_size=batch_size,
                          generation=generation, dtype=torch.float32)
    return {"make": make, "experiment": experiment, "file": experiment / "qwen_llm_test_predictions.json"}


@pytest.fixture
def llm_reference(tiny, tmp_path_factory):
    root = tmp_path_factory.mktemp("llm-reference")
    experiment = root / "out" / "qwen" / "0.0"
    shutil.copytree(tiny["llm"] / "splits", experiment / "splits")
    tester = Tester_llm("qwen", str(tiny["gpt2"]), experiment, None, tiny["tokenizer"], 0.0, batch_size=4, dtype=torch.float32)
    tester.run(out_path=experiment, max_gen_tokens=GENERATED_TOKENS)
    return prediction_files.load(experiment / "qwen_llm_test_predictions.json")


def interrupt_llm_at_batch(monkeypatch, batch_number, action=raise_keyboard_interrupt):
    original = Tester_llm.safe_generate
    calls = {"n": 0}

    def counting(self, prompts, max_gen_tokens):
        calls["n"] += 1
        if calls["n"] == batch_number:
            action()
        return original(self, prompts, max_gen_tokens)
    monkeypatch.setattr(Tester_llm, "safe_generate", counting)


class TestLlmTester:
    def test_a_full_run_predicts_every_case(self, llm, llm_reference):
        assert [r["ID"] for r in llm_reference] == list(range(28))
        assert all(len(r["preds"]) == 4 for r in llm_reference)           # 4 returned sequences by default

    def test_interrupted_run_resumes_to_exactly_the_same_predictions(self, llm, monkeypatch, llm_reference):
        """The defaults sample at temperature 1.5; seeding per batch makes the resumed run identical."""
        with monkeypatch.context() as m:
            interrupt_llm_at_batch(m, 3)
            with pytest.raises(KeyboardInterrupt):
                llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert [r["ID"] for r in prediction_files.load(llm["file"])] == list(range(8))   # two batches of four
        llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert prediction_files.load(llm["file"]) == llm_reference

    def test_sigterm_during_a_batch_keeps_the_finished_batches(self, llm, monkeypatch, llm_reference):
        import os
        with monkeypatch.context() as m:
            interrupt_llm_at_batch(m, 2, lambda: os.kill(os.getpid(), signal.SIGTERM))
            with pytest.raises(KeyboardInterrupt):
                llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert [r["ID"] for r in prediction_files.load(llm["file"])] == [0, 1, 2, 3]
        llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert prediction_files.load(llm["file"]) == llm_reference

    def test_cases_that_failed_are_retried_by_the_next_run(self, llm, monkeypatch, llm_reference):
        original = Tester_llm.safe_generate
        calls = {"n": 0}

        def second_batch_runs_out_of_memory(self, prompts, max_gen_tokens):
            calls["n"] += 1
            return [None] * len(prompts) if calls["n"] == 2 else original(self, prompts, max_gen_tokens)
        with monkeypatch.context() as m:
            m.setattr(Tester_llm, "safe_generate", second_batch_runs_out_of_memory)
            llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert [r["ID"] for r in prediction_files.load(llm["file"])] == [0, 1, 2, 3] + list(range(8, 28))
        llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert prediction_files.load(llm["file"]) == llm_reference

    def test_a_different_seed_gives_different_samples_and_none_means_unseeded(self, llm):
        assert Tester_llm.resolve_generation("qwen")["seed"] == 1234
        assert Tester_llm.resolve_generation("qwen", {"seed": None})["seed"] is None
        assert Tester_llm.resolve_generation("qwen", {"seed": 7})["seed"] == 7

    def test_greedy_decoding_needs_no_seed(self, llm):
        greedy = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1, "seed": None}
        llm["make"](generation=greedy).run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert all(len(r["preds"]) == 1 for r in prediction_files.load(llm["file"]))

    def test_a_damaged_predictions_file_stops_the_run_and_is_not_overwritten(self, llm):
        llm["experiment"].mkdir(parents=True, exist_ok=True)
        llm["file"].write_text('{"ID": {"0": 0')
        with pytest.raises(RuntimeError, match="damaged"):
            llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert llm["file"].read_text() == '{"ID": {"0": 0'

    def test_predictions_of_another_test_split_are_refused(self, llm):
        prediction_files.save([{"ID": 3, "target": "not the target", "preds": ["x"]}], llm["file"])
        with pytest.raises(ValueError, match="do not belong"):
            llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)

    def test_the_final_report_says_how_many_cases_it_covers(self, llm, capsys):
        llm["make"]().run(out_path=llm["experiment"], max_gen_tokens=GENERATED_TOKENS)
        assert "over 28/28 cases" in capsys.readouterr().out


# ---- Eftt --------------------------------------------------------------------------------------------------------------

class TestMaxLength:
    """The input window per model: 1024 for CodeT5+ and PLBART (only 41% of the cases fit in 512), CodeGen keeps its own."""

    def eftt(self, tiny, tmp_path, model, native, max_length=None):
        eftt = Eftt.__new__(Eftt)
        eftt.model, eftt.max_length, eftt.out_path, eftt.train_size = model, max_length, tmp_path, 0.8
        eftt.tokenizer = fixtures.build_tokenizer(model_max_length=native)
        eftt.create_tokenizer()
        return eftt

    @pytest.mark.parametrize("model,native,expected", [("codet5p", 512, 1024), ("plbart", 512, 1024), ("plbart", 1024, 1024),
                                                       ("codegen", 2048, 2048)])
    def test_default_window_per_model(self, tiny, tmp_path, model, native, expected):
        assert self.eftt(tiny, tmp_path, model, native).tokenizer.model_max_length == expected

    def test_explicit_max_length_wins(self, tiny, tmp_path):
        assert self.eftt(tiny, tmp_path, "codet5p", 512, max_length=512).tokenizer.model_max_length == 512
        assert self.eftt(tiny, tmp_path, "codegen", 2048, max_length=1024).tokenizer.model_max_length == 1024

    def test_a_placeholder_length_is_still_replaced_for_other_models(self, tiny, tmp_path):
        assert self.eftt(tiny, tmp_path, "somemodel", int(1e30)).tokenizer.model_max_length == 512


class TestEftt:
    def test_train_and_validate_pass_their_options_on(self, monkeypatch):
        seen = {}

        class FakeTrainer:
            def __init__(self, *args, **options):
                seen["trainer"] = options

            def train(self, resume=True):
                seen["resume"] = resume

        class FakeTester:
            def __init__(self, *args, **options):
                seen["tester_args"] = args

            def validate(self, **options):
                seen["tester"] = options
                return "scores"
        monkeypatch.setattr(encode_tune_test, "Trainer", FakeTrainer)
        monkeypatch.setattr(encode_tune_test, "Tester", FakeTester)
        eftt = Eftt.__new__(Eftt)
        eftt.model, eftt.model_path, eftt.tokenizer, eftt.out_path = "codet5p", "p", None, "o"
        eftt.model_class, eftt.train_size, eftt.beam_size = None, 0.8, 20
        eftt.dataset_class = PLBARTDataset
        eftt.train(resume=False, epochs=1, batch_size=2)
        assert seen["trainer"] == {"epochs": 1, "batch_size": 2} and seen["resume"] is False
        assert eftt.validate(limit=5, save_every=1) == "scores" and seen["tester"] == {"limit": 5, "save_every": 1}
        assert seen["tester_args"][-1] is PLBARTDataset      # the prompts are built like the training inputs


    def test_no_login_without_a_token(self, monkeypatch):
        calls = []
        monkeypatch.setattr(encode_tune_test, "login", lambda token=None: calls.append(token))
        eftt = Eftt.__new__(Eftt)
        eftt.hftok = None
        eftt._hf_login()
        assert calls == []                                                  # login(None) would prompt on the terminal

    def test_login_with_a_token(self, monkeypatch):
        calls = []
        monkeypatch.setattr(encode_tune_test, "login", lambda token=None: calls.append(token))
        eftt = Eftt.__new__(Eftt)
        eftt.hftok = "hf_dummy"
        eftt._hf_login()
        assert calls == ["hf_dummy"]
