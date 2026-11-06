"""Tests for data_processing/prediction_files.py (no torch needed)."""
import json

import pytest

from data_processing import prediction_files


def records(*ids):
    return [{"ID": i, "target": f"t{i}", "preds": [f"p{i}a", f"p{i}b"]} for i in ids]


class TestSaveAndLoad:
    def test_roundtrip_and_sorted_by_id(self, tmp_path):
        path = tmp_path / "preds.json"
        prediction_files.save(records(3, 1, 2), path)
        assert [r["ID"] for r in prediction_files.load(path)] == [1, 2, 3]
        assert prediction_files.load(path) == sorted(records(1, 2, 3), key=lambda r: r["ID"])

    def test_numeric_looking_strings_stay_strings(self, tmp_path):
        """DataFrame/read_json type guessing would turn these into numbers and dates."""
        tricky = [{"ID": 0, "target": "007", "preds": ["1e5", "2024-01-01"]}, {"ID": 1, "target": "42", "preds": ["True"]}]
        path = tmp_path / "preds.json"
        prediction_files.save(tricky, path)
        assert prediction_files.load(path) == tricky

    def test_the_file_keeps_the_pandas_column_format_other_code_reads(self, tmp_path):
        path = tmp_path / "preds.json"
        prediction_files.save(records(0, 1), path)
        data = json.loads(path.read_text())
        assert set(data) == {"ID", "target", "preds"} and data["target"] == {"0": "t0", "1": "t1"}

    def test_reads_a_list_of_records_too(self, tmp_path):
        path = tmp_path / "preds.json"
        path.write_text(json.dumps(records(0, 1)))
        assert prediction_files.load(path) == records(0, 1)

    def test_a_damaged_file_is_an_error_and_is_left_alone(self, tmp_path):
        path = tmp_path / "preds.json"
        path.write_text('{"ID": {"0": 0, "1"')
        with pytest.raises(RuntimeError, match="damaged"):
            prediction_files.load(path)
        assert path.read_text() == '{"ID": {"0": 0, "1"'

    def test_saving_over_a_complete_file_never_exposes_a_partial_one(self, tmp_path, monkeypatch):
        path = tmp_path / "preds.json"
        prediction_files.save(records(0, 1), path)

        def fail(src, dst):
            raise OSError("killed")
        monkeypatch.setattr("data_processing.atomic_io.os.replace", fail)
        with pytest.raises(OSError):
            prediction_files.save(records(0, 1, 2), path)
        monkeypatch.undo()
        assert prediction_files.load(path) == records(0, 1)


class TestChecks:
    ROWS = [{"output": "t0"}, {"output": "t1"}, {"output": "t2"}]

    def test_matching_split_is_accepted(self):
        prediction_files.check_matches_test_split(records(0, 2), self.ROWS)

    def test_different_target_is_refused(self):
        with pytest.raises(ValueError, match="do not belong"):
            prediction_files.check_matches_test_split([{"ID": 1, "target": "other", "preds": []}], self.ROWS)

    def test_id_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="do not belong"):
            prediction_files.check_matches_test_split(records(7), self.ROWS)

    def test_describe_complete_and_partial(self):
        assert prediction_files.describe(12, 12) == ("over 12/12 cases", None)
        text, warning = prediction_files.describe(5, 12)
        assert text == "over 5/12 cases" and "only 5 of 12" in warning and "incomplete" in warning
