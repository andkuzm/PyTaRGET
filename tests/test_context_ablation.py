"""Context ablation (WP5): the added sections and their leak-safety."""
import json

import pytest

from data_processing import context_ablation as ca

BASE = "### Instruction:\nX\n\n### Broken Lines:\nenv.simulate(2)\n\n### Source Code Changes:\nH\n\n### Repaired Code:\n"
ROW = {"repository_name": "o/r", "broken_hash": "b" * 40, "test_name": "test_x", "relative_path": "tests/test_x.py",
       "outdated_test_log": "[<TESTLOG>]\nF\n" + "=" * 40 + " FAILURES " + "=" * 40 + "\n"
                           "/gpfs/helios/home/u/repos_v2/slot1/pkg/mod.py:12: AttributeError\nE       AttributeError: 'Env' has no attribute 'simulate'\n"}


class TestFailureContext:
    def test_marker_rulers_and_machine_paths_are_removed(self):
        text = ca.failure_context(ROW)
        assert "[<TESTLOG>]" not in text and "=====" not in text and "/gpfs/" not in text
        assert "pkg/mod.py:12: AttributeError" in text and "no attribute 'simulate'" in text

    def test_long_logs_keep_head_and_tail(self):
        row = dict(ROW, outdated_test_log="[<TESTLOG>]\n" + "A" * 3000 + "\nTAILMARK")
        text = ca.failure_context(row, max_chars=500)
        assert len(text) <= 520 and text.startswith("A") and text.endswith("TAILMARK") and "..." in text

    def test_missing_log_gives_nothing(self):
        assert ca.failure_context({"outdated_test_log": None}) == ""


class TestSymbols:
    def test_identifiers_in_order_without_boilerplate(self):
        assert ca.symbols_from_broken("assert self.client.fetch_items(page_size=3) == expected") == ["client", "fetch_items", "page_size"]

    def test_limit_and_duplicates(self):
        assert ca.symbols_from_broken("aaa bbb aaa ccc", limit=2) == ["aaa", "bbb"]


class TestIndex:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod.py").write_text('class Env:\n    """Event loop."""\n    def run(self, until=None):\n        """Run it.\n\n        more"""\n        pass\n\ndef helper(a, b=1) -> int:\n    return a\n')
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def secret_repair(): pass\n")
        (tmp_path / "pkg" / "test_inline.py").write_text("def also_hidden(): pass\n")
        (tmp_path / "pkg" / "broken.py").write_text("def (\n")
        return tmp_path

    def test_sources_are_indexed_tests_and_unparseable_files_are_not(self, project):
        index = ca.index_source(project)
        assert {"Env", "run", "helper"} <= set(index)
        assert "secret_repair" not in index and "also_hidden" not in index
        assert "def helper(a, b=1) -> int" in index["helper"][0][1]
        assert "# Event loop." in index["Env"][0][1] and "methods: run" in index["Env"][0][1]

    def test_the_cases_own_test_file_is_excluded(self, project):
        (project / "pkg" / "extra.py").write_text("def own(): pass\n")
        assert "own" not in ca.index_source(project, exclude={"pkg/extra.py"})

    def test_rendering_and_budget(self, project):
        index = ca.index_source(project)
        text = ca.definitions_context(["helper", "Env", "missing"], index)
        assert text.startswith("# pkg/mod.py\ndef helper") and "class Env" in text
        assert ca.definitions_context(["helper", "Env"], index, max_chars=60).count("# pkg/mod.py") == 1


class TestPrompt:
    def test_sections_go_before_the_tail_in_a_fixed_order(self):
        out = ca.add_sections(BASE, failure="FAIL", definitions="DEFS")
        assert out.endswith("### Repaired Code:\n")
        assert out.index("### Source Code Changes:") < out.index("### Test Failure Output:\nFAIL") < out.index("### Definitions of Changed Symbols:\nDEFS") < out.index("### Repaired Code:")

    def test_nothing_added_leaves_the_prompt(self):
        assert ca.add_sections(BASE) == BASE

    def test_unexpected_tail_is_refused(self):
        with pytest.raises(ValueError):
            ca.add_sections("no tail", failure="x")

    @pytest.mark.parametrize("variant,has_b,has_c", [("B", True, False), ("C", False, True), ("BC", True, True)])
    def test_attach_uses_only_the_sources_of_the_variant(self, variant, has_b, has_c):
        class T:
            def build_prompt(self, row, max_hunks=None):
                return BASE
        t = ca.attach(T(), variant, {ca.case_key(ROW): {"failure": "FAIL", "definitions": "DEFS"}})
        out = t.build_prompt(ROW)
        assert ("### Test Failure Output:" in out) == has_b and ("### Definitions of Changed Symbols:" in out) == has_c

    def test_unknown_variant_and_unknown_case(self):
        class T:
            def build_prompt(self, row, max_hunks=None):
                return BASE
        with pytest.raises(ValueError):
            ca.attach(T(), "A", {})
        assert ca.attach(T(), "BC", {}).build_prompt(ROW) == BASE


def test_report_counts_reference_leaks(capsys):
    row = dict(ROW, output="[<REPAIREDTEST>]env.run(until=2) and more text here[</REPAIREDTEST>]")
    ca.report([row], {ca.case_key(row): {"failure": "", "definitions": "x env.run(until=2) and more text here y", "index_ok": True}})
    assert "verbatim in the added context: 1" in capsys.readouterr().out
