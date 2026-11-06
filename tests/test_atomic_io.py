"""Tests for data_processing/atomic_io.py: a write that is interrupted must never damage what is already saved."""
import json
import os
import signal

import pandas as pd
import pytest

from data_processing import atomic_io


class TestAtomicFile:
    def test_write_replaces_the_file_and_leaves_no_temporary(self, tmp_path):
        target = tmp_path / "out.json"
        target.write_text("old")
        atomic_io.atomic_write_text(target, "new")
        assert target.read_text() == "new"
        assert [p.name for p in tmp_path.iterdir()] == ["out.json"]

    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "out.txt"
        atomic_io.atomic_write_text(target, "x")
        assert target.read_text() == "x"

    def test_an_error_while_writing_keeps_the_old_content(self, tmp_path):
        target = tmp_path / "out.json"
        target.write_text("complete old content")
        with pytest.raises(RuntimeError):
            with atomic_io.atomic_path(target) as temporary:
                temporary.write_text("half of the new con")
                raise RuntimeError("killed while writing")
        assert target.read_text() == "complete old content"
        assert [p.name for p in tmp_path.iterdir()] == ["out.json"]

    def test_a_keyboard_interrupt_while_writing_keeps_the_old_content(self, tmp_path):
        target = tmp_path / "out.json"
        target.write_text("old")
        with pytest.raises(KeyboardInterrupt):
            with atomic_io.atomic_path(target) as temporary:
                temporary.write_text("partial")
                raise KeyboardInterrupt
        assert target.read_text() == "old"

    def test_a_failing_replace_keeps_the_old_content_and_cleans_up(self, tmp_path, monkeypatch):
        target = tmp_path / "out.json"
        target.write_text("old")

        def broken_replace(src, dst):
            raise OSError("disk went away")
        monkeypatch.setattr(atomic_io.os, "replace", broken_replace)
        with pytest.raises(OSError):
            atomic_io.atomic_write_text(target, "new")
        assert target.read_text() == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["out.json"]

    def test_json_roundtrip(self, tmp_path):
        target = tmp_path / "stats.json"
        atomic_io.atomic_write_json(target, {"epochs": [1, 2], "text": "ä\nline"}, indent=2)
        assert json.loads(target.read_text()) == {"epochs": [1, 2], "text": "ä\nline"}

    def test_dataframe_json_keeps_the_pandas_format(self, tmp_path):
        frame = pd.DataFrame([{"ID": 0, "target": "a", "preds": ["a", "b"]}, {"ID": 1, "target": "c", "preds": ["c"]}])
        target = tmp_path / "predictions.json"
        atomic_io.atomic_dataframe_to_json(frame, target, indent=2)
        loaded = pd.read_json(target, dtype=False, convert_dates=False)
        assert loaded.to_dict("records") == frame.to_dict("records")

    def test_pickle_roundtrip(self, tmp_path):
        import pickle
        target = tmp_path / "train.pkl"
        atomic_io.atomic_pickle({"a": [1, 2, 3]}, target)
        assert pickle.loads(target.read_bytes()) == {"a": [1, 2, 3]}

    def test_stale_temporaries_of_a_killed_writer_are_removed(self, tmp_path):
        target = tmp_path / "out.json"
        (tmp_path / "out.json.tmp-4242").write_text("leftover")
        (tmp_path / "out.json.tmp-9999").mkdir()
        atomic_io.remove_leftovers(target)
        assert list(tmp_path.iterdir()) == []


class TestAtomicDirectory:
    def fill(self, directory, text):
        (directory / "weights.bin").write_text(text)
        (directory / "config.json").write_text("{}")

    def test_swap_replaces_the_directory(self, tmp_path):
        target = tmp_path / "checkpoint-best"
        target.mkdir()
        (target / "weights.bin").write_text("old")
        with atomic_io.atomic_directory(target) as temporary:
            self.fill(temporary, "new")
        assert (target / "weights.bin").read_text() == "new"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-best"]

    def test_creates_the_directory_when_there_is_none(self, tmp_path):
        target = tmp_path / "exp" / "checkpoint-best"
        with atomic_io.atomic_directory(target) as temporary:
            self.fill(temporary, "first")
        assert (target / "weights.bin").read_text() == "first"

    def test_an_error_while_filling_keeps_the_previous_version(self, tmp_path):
        target = tmp_path / "checkpoint-best"
        target.mkdir()
        (target / "weights.bin").write_text("complete old checkpoint")
        with pytest.raises(KeyboardInterrupt):
            with atomic_io.atomic_directory(target) as temporary:
                (temporary / "weights.bin").write_text("half a checkpoint")
                raise KeyboardInterrupt
        assert (target / "weights.bin").read_text() == "complete old checkpoint"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-best"]

    def test_a_failing_second_rename_puts_the_previous_version_back(self, tmp_path, monkeypatch):
        target = tmp_path / "checkpoint-last"
        target.mkdir()
        (target / "weights.bin").write_text("previous version")
        real_replace = os.replace

        def fail_when_the_new_version_goes_in(src, dst):
            if ".tmp-" in str(src):
                raise OSError("No space left on device")
            return real_replace(src, dst)
        monkeypatch.setattr(atomic_io.os, "replace", fail_when_the_new_version_goes_in)
        with pytest.raises(OSError):
            with atomic_io.atomic_directory(target) as temporary:
                self.fill(temporary, "new version")
        assert (target / "weights.bin").read_text() == "previous version"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-last"]

    def test_a_kill_between_the_two_renames_is_recovered(self, tmp_path):
        """The swap moved the old version aside and died before the new one was in place."""
        target = tmp_path / "checkpoint-best"
        old = tmp_path / "checkpoint-best.old"
        old.mkdir()
        (old / "weights.bin").write_text("last complete version")
        (tmp_path / "checkpoint-best.tmp-123").mkdir()
        atomic_io.recover_directory(target)
        assert (target / "weights.bin").read_text() == "last complete version"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-best"]

    def test_a_kill_after_the_new_version_is_in_place_drops_the_old_one(self, tmp_path):
        target = tmp_path / "checkpoint-best"
        target.mkdir()
        (target / "weights.bin").write_text("new")
        (tmp_path / "checkpoint-best.old").mkdir()
        atomic_io.recover_directory(target)
        assert (target / "weights.bin").read_text() == "new"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-best"]

    def test_the_next_swap_recovers_by_itself(self, tmp_path):
        target = tmp_path / "checkpoint-best"
        old = tmp_path / "checkpoint-best.old"
        old.mkdir()
        (old / "weights.bin").write_text("v1")
        with atomic_io.atomic_directory(target) as temporary:
            self.fill(temporary, "v2")
        assert (target / "weights.bin").read_text() == "v2"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-best"]

    def test_save_pretrained_goes_through_a_swap(self, tmp_path):
        class Model:
            def save_pretrained(self, path):
                (path / "model.safetensors").write_text("weights")
        atomic_io.atomic_save_pretrained(Model(), tmp_path / "checkpoint-best")
        assert (tmp_path / "checkpoint-best" / "model.safetensors").read_text() == "weights"


class TestSigterm:
    def test_sigterm_becomes_a_keyboard_interrupt_and_the_handler_is_restored(self):
        before = signal.getsignal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            with atomic_io.sigterm_as_keyboard_interrupt():
                os.kill(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) == before

    def test_work_that_finishes_is_not_disturbed(self):
        with atomic_io.sigterm_as_keyboard_interrupt():
            value = 1 + 1
        assert value == 2
