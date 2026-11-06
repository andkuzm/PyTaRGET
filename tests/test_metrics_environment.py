"""CodeBLEU needs tree-sitter >= 0.23 (the README pins 0.24.0); with an older one every score fails with
"TypeError: an integer is required". This test tells that apart from a bug in the metric."""
import pytest

pytest.importorskip("tree_sitter")
tspython = pytest.importorskip("tree_sitter_python")

from tree_sitter import Language                                           # noqa: E402

try:
    Language(tspython.language())
except TypeError:
    pytest.skip("tree-sitter is too old for CodeBLEU: pip install tree-sitter==0.24.0", allow_module_level=True)

from data_processing.CodeBLEU.code_bleu import calc_code_bleu              # noqa: E402

REFERENCE = ["self.assertEqual(f(1), 2)", "x = old_name(1)\nreturn x"]


def score(candidates):
    return calc_code_bleu([REFERENCE], candidates, lang="python")


def test_codebleu_orders_identical_close_and_unrelated_predictions():
    close = ["self.assertEqual(f(1), 2)", "x = new_name(1)\nreturn x"]
    unrelated = ["pass", "???"]
    assert score(REFERENCE) > score(close) > score(unrelated) >= 0
    assert score(REFERENCE) <= 1
