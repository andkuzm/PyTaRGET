"""
Resumability of Trainer (data_processing/train.py) on a tiny model and 28 real cases (see tiny_eval_fixtures.py).

Every test kills the training in a different way (Ctrl-C in validation, SIGTERM, a failing disk write, a kill right
after a checkpoint) and checks that the next call to train() continues and ends exactly where an uninterrupted run ends.
"""
import json
import os
import pickle
import shutil
import signal

import pytest

import tiny_eval_fixtures as fixtures
from data_processing import atomic_io
from data_processing.train import Trainer
from transformers import T5ForConditionalGeneration

torch = fixtures.torch
EPOCHS = 4


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    tokenizer = fixtures.build_tokenizer()
    experiment = fixtures.encode(root / "encoded", tokenizer)
    base = fixtures.tiny_t5(tokenizer, root / "base-model")
    return {"tokenizer": tokenizer, "encoded": experiment, "base": base}


def trainer(tiny, root, epochs=EPOCHS, early_stop=10):
    """A Trainer over a private copy of the encoded data, so each test has its own output directory."""
    experiment = root / "out" / "codet5p" / "0.5"
    if not experiment.exists():
        shutil.copytree(tiny["encoded"], experiment)
    return Trainer("codet5p", str(tiny["base"]), tiny["tokenizer"], root / "out", T5ForConditionalGeneration, 0.5,
                   epochs=epochs, batch_size=4, eval_batch_size=4, early_stop=early_stop, learning_rate=1e-3)


def experiment(root):
    return root / "out" / "codet5p" / "0.5"


def weights(path):
    return {k: v.clone() for k, v in T5ForConditionalGeneration.from_pretrained(path).state_dict().items()}


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def losses(root):
    stats = json.loads((experiment(root) / "training_stats.json").read_text())
    return [(e["epoch"], e["train_loss"], e["valid_loss"]) for e in stats["epochs"]]


def saved_epoch(root):
    return json.loads((experiment(root) / "checkpoint-last" / "trainer_state.json").read_text())["epoch"]


def leftovers(root):
    return [p.name for p in experiment(root).iterdir() if ".tmp-" in p.name or p.name.endswith(".old")]


@pytest.fixture(scope="module")
def uninterrupted(tiny, tmp_path_factory):
    """The reference: the same training without any interruption."""
    root = tmp_path_factory.mktemp("reference")
    trainer(tiny, root).train()
    return {"losses": losses(root), "last": weights(experiment(root) / "checkpoint-last"),
            "best": weights(experiment(root) / "checkpoint-best")}


class Kill:
    """Make the n-th call of a Trainer method raise (a stand-in for the job being killed at that point)."""

    def __init__(self, monkeypatch, name, nth, action):
        self.calls = 0
        original = getattr(Trainer, name)

        def wrapper(trainer_self, *args, **kwargs):
            self.calls += 1
            if self.calls == nth:
                action()
            return original(trainer_self, *args, **kwargs)
        monkeypatch.setattr(Trainer, name, wrapper)


def raise_keyboard_interrupt():
    raise KeyboardInterrupt


class TestTraining:
    def test_a_full_run_leaves_a_complete_consistent_state(self, tiny, tmp_path):
        trainer(tiny, tmp_path, epochs=3).train()
        stats = json.loads((experiment(tmp_path) / "training_stats.json").read_text())
        assert stats["completed"] is True and len(stats["epochs"]) == 3
        assert stats["train_set_size"] == 12 and stats["valid_set_size"] == 4
        for name in ("model", "checkpoint-best", "checkpoint-last"):
            assert (experiment(tmp_path) / name / "config.json").exists()
        assert saved_epoch(tmp_path) == 3
        assert leftovers(tmp_path) == []

    def test_training_actually_changes_the_weights(self, tiny, tmp_path):
        trainer(tiny, tmp_path, epochs=2).train()
        assert not same_weights(weights(experiment(tmp_path) / "model"), weights(experiment(tmp_path) / "checkpoint-last"))


class TestResume:
    def test_interrupted_in_validation_then_resumed_equals_uninterrupted(self, tiny, tmp_path, monkeypatch, uninterrupted):
        with monkeypatch.context() as m:
            Kill(m, "validate", 3, raise_keyboard_interrupt)            # epoch 3 trained, killed before it was saved
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path).train()
        assert saved_epoch(tmp_path) == 2
        assert [e[0] for e in losses(tmp_path)] == [1, 2]
        assert json.loads((experiment(tmp_path) / "training_stats.json").read_text())["completed"] is False

        trainer(tiny, tmp_path).train()                                  # a new process would do exactly this

        assert losses(tmp_path) == uninterrupted["losses"]
        assert same_weights(weights(experiment(tmp_path) / "checkpoint-last"), uninterrupted["last"])
        assert same_weights(weights(experiment(tmp_path) / "checkpoint-best"), uninterrupted["best"])
        assert leftovers(tmp_path) == []

    def test_killed_several_times_still_ends_the_same(self, tiny, tmp_path, monkeypatch, uninterrupted):
        for nth in (2, 2, 2):                                            # each restart loses only the epoch in progress
            with monkeypatch.context() as m:
                Kill(m, "validate", nth, raise_keyboard_interrupt)
                with pytest.raises(KeyboardInterrupt):
                    trainer(tiny, tmp_path).train()
        assert saved_epoch(tmp_path) == 3
        trainer(tiny, tmp_path).train()
        assert losses(tmp_path) == uninterrupted["losses"]
        assert same_weights(weights(experiment(tmp_path) / "checkpoint-last"), uninterrupted["last"])

    def test_sigterm_from_slurm_stops_training_cleanly_and_it_resumes(self, tiny, tmp_path, monkeypatch, uninterrupted):
        with monkeypatch.context() as m:
            Kill(m, "validate", 2, lambda: os.kill(os.getpid(), signal.SIGTERM))
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path).train()
        assert saved_epoch(tmp_path) == 1
        trainer(tiny, tmp_path).train()
        assert losses(tmp_path) == uninterrupted["losses"]

    def test_killed_right_after_the_checkpoint_was_saved(self, tiny, tmp_path, monkeypatch, uninterrupted):
        with monkeypatch.context() as m:
            Kill(m, "write_stats", 4, raise_keyboard_interrupt)         # 1 = fresh start, 2..4 = epochs 1..3
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path).train()
        assert saved_epoch(tmp_path) == 3
        trainer(tiny, tmp_path).train()
        assert losses(tmp_path) == uninterrupted["losses"]
        assert same_weights(weights(experiment(tmp_path) / "checkpoint-last"), uninterrupted["last"])

    def test_a_failing_checkpoint_write_keeps_the_previous_state_usable(self, tiny, tmp_path, monkeypatch, uninterrupted):
        real_replace = os.replace
        swaps = {"checkpoint-last": 0}

        def failing_replace(src, dst):
            if ".tmp-" in str(src) and str(dst).endswith("checkpoint-last"):
                swaps["checkpoint-last"] += 1
                if swaps["checkpoint-last"] == 2:                        # the disk fails while saving epoch 2
                    raise OSError("No space left on device")
            return real_replace(src, dst)

        with monkeypatch.context() as m:
            m.setattr(atomic_io.os, "replace", failing_replace)
            with pytest.raises(OSError):
                trainer(tiny, tmp_path).train()
        assert saved_epoch(tmp_path) == 1                                # the epoch-1 state is intact
        assert leftovers(tmp_path) == []
        trainer(tiny, tmp_path).train()
        assert losses(tmp_path) == uninterrupted["losses"]
        assert same_weights(weights(experiment(tmp_path) / "checkpoint-last"), uninterrupted["last"])

    def test_a_kill_between_the_two_renames_of_the_swap_is_recovered(self, tiny, tmp_path, monkeypatch, uninterrupted):
        with monkeypatch.context() as m:
            Kill(m, "validate", 3, raise_keyboard_interrupt)
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path).train()
        last = experiment(tmp_path) / "checkpoint-last"
        os.replace(last, experiment(tmp_path) / "checkpoint-last.old")  # what a kill inside the swap leaves behind
        (experiment(tmp_path) / "checkpoint-last.tmp-4242").mkdir()
        trainer(tiny, tmp_path).train()
        assert losses(tmp_path) == uninterrupted["losses"]
        assert leftovers(tmp_path) == []

    def test_early_stopping_decided_before_the_kill_is_not_forgotten(self, tiny, tmp_path, monkeypatch):
        """Epoch 2 is worse than epoch 1 and patience is 1: training must end at epoch 2, also after a restart."""
        count = {"n": 0}

        def worse_every_epoch(self, *args, **kwargs):
            count["n"] += 1
            return float(count["n"])
        monkeypatch.setattr(Trainer, "validate", worse_every_epoch)
        with monkeypatch.context() as m:
            Kill(m, "write_stats", 3, raise_keyboard_interrupt)         # after the epoch-2 checkpoint, before the stop
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path, epochs=6, early_stop=1).train()
        assert saved_epoch(tmp_path) == 2
        calls_before = count["n"]
        trainer(tiny, tmp_path, epochs=6, early_stop=1).train()
        stats = json.loads((experiment(tmp_path) / "training_stats.json").read_text())
        assert count["n"] == calls_before                                # no further epoch was trained
        assert stats["completed"] is True and stats["stopped_early"] is True and len(stats["epochs"]) == 2


class TestFinishedAndFreshRuns:
    def test_calling_train_again_after_completion_does_nothing(self, tiny, tmp_path):
        trainer(tiny, tmp_path, epochs=2).train()
        state = experiment(tmp_path) / "checkpoint-last" / "trainer_state.json"
        before = (state.stat().st_mtime_ns, state.read_text(), losses(tmp_path))
        trainer(tiny, tmp_path, epochs=2).train()
        assert (state.stat().st_mtime_ns, state.read_text(), losses(tmp_path)) == before

    def test_a_fresh_run_never_destroys_the_best_checkpoint_of_the_earlier_one(self, tiny, tmp_path):
        trainer(tiny, tmp_path, epochs=2).train()
        earlier_best = weights(experiment(tmp_path) / "checkpoint-best")
        trainer(tiny, tmp_path, epochs=1).train(resume=False)
        assert same_weights(weights(experiment(tmp_path) / "checkpoint-best.previous"), earlier_best)
        assert len(losses(tmp_path)) == 1                                # the new run's own statistics

    def test_a_restart_after_a_kill_does_not_overwrite_the_best_checkpoint(self, tiny, tmp_path, monkeypatch):
        with monkeypatch.context() as m:
            Kill(m, "validate", 3, raise_keyboard_interrupt)
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path).train()
        best_after_kill = weights(experiment(tmp_path) / "checkpoint-best")
        best_loss = json.loads((experiment(tmp_path) / "checkpoint-last" / "trainer_state.json").read_text())["best_loss"]
        # while resuming, the best checkpoint is only touched when an epoch beats the best loss of the whole run
        replaced = []
        real = atomic_io.atomic_directory

        def spy(path):
            replaced.append(str(path).split("/")[-1])
            return real(path)
        monkeypatch.setattr("data_processing.train.atomic_directory", spy)
        monkeypatch.setattr("data_processing.train.atomic_save_pretrained", lambda model, path: replaced.append("BEST") or
                            atomic_io.atomic_save_pretrained(model, path))
        trainer(tiny, tmp_path).train()
        stats = json.loads((experiment(tmp_path) / "training_stats.json").read_text())
        improved_after = [e for e in stats["epochs"] if e["epoch"] > 2 and e["valid_loss"] < round(best_loss, 4)]
        assert replaced.count("BEST") == len(improved_after)
        if not improved_after:
            assert same_weights(weights(experiment(tmp_path) / "checkpoint-best"), best_after_kill)

    def test_resuming_on_data_that_was_encoded_again_is_refused(self, tiny, tmp_path, monkeypatch):
        with monkeypatch.context() as m:
            Kill(m, "validate", 2, raise_keyboard_interrupt)
            with pytest.raises(KeyboardInterrupt):
                trainer(tiny, tmp_path).train()
        train_file = experiment(tmp_path) / "train.pkl"
        smaller = pickle.load(open(train_file, "rb"))
        smaller.data = smaller.data[:-2]
        pickle.dump(smaller, open(train_file, "wb"))
        with pytest.raises(ValueError, match="encoded again"):
            trainer(tiny, tmp_path).train()
        assert saved_epoch(tmp_path) == 1                                # nothing was damaged

    def test_an_incomplete_checkpoint_directory_is_not_used(self, tiny, tmp_path):
        trainer(tiny, tmp_path, epochs=1)                                # creates the experiment directory
        broken = experiment(tmp_path) / "checkpoint-last"
        broken.mkdir()
        (broken / "trainer_state.json").write_text('{"epoch": 7}')      # no optimizer / scheduler / weights
        trainer(tiny, tmp_path, epochs=1).train()
        assert saved_epoch(tmp_path) == 1
        assert (broken / "optimizer.pt").exists()


class TestInputs:
    def test_an_empty_validation_set_is_a_clear_error_not_a_crash_in_the_loop(self, tiny, tmp_path):
        trainer(tiny, tmp_path)                                          # copies the encoded data
        valid_file = experiment(tmp_path) / "valid.pkl"
        empty = pickle.load(open(valid_file, "rb"))
        empty.data = []
        pickle.dump(empty, open(valid_file, "wb"))
        with pytest.raises(ValueError, match="validation"):
            trainer(tiny, tmp_path).train()
        assert not (experiment(tmp_path) / "checkpoint-last").exists()   # nothing was started
