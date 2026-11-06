"""Tests for data_processing/filter_cases.py."""
import csv
import json

import pytest

from data_processing import filter_cases

HEADER = ["repository_name", "annotated_code", "relative_path", "broken_hash", "repaired_hash", "outdated_test_log",
          "test_name", "broken_date", "repaired_date"]


def code(broken, fixed):
    return ("[<TESTCONTEXT>]\ndef test_x():\n[<BREAKAGE>]\n%s\n[</BREAKAGE>]\n[</TESTCONTEXT>]\n"
            "[<REPAIREDTEST>]\n%s\n[</REPAIREDTEST>]\n" % (broken, fixed))


def log(error):
    return "[<TESTLOG>]\nF\n>       x = main()\nE       %s\n\nFAILED t.py::test - %s\n[</TESTLOG>]\n" % (error, error)


def case(name="o/r", pair=("b1", "r1"), error="NameError: name 'main' is not defined", broken="x = main()",
         fixed="x = cli.main()", test="t"):
    return [name, code(broken, fixed), "t.py", pair[0], pair[1], log(error), test, "2026-01-01", "2026-01-02"]


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_ALL)
        writer.writerow(HEADER)
        writer.writerows(rows)


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="|"))


class TestHelpers:
    def test_final_error_is_the_last_e_line(self):
        text = "E   first\nsome text\nE       AttributeError: boom\n\nFAILED x - y"
        assert filter_cases.final_error(text) == "AttributeError: boom"

    def test_final_error_empty_without_e_lines(self):
        assert filter_cases.final_error("no errors here\nFAILED x") == ""
        assert filter_cases.final_error("") == ""

    def test_final_error_ignores_lines_that_only_start_with_e(self):
        assert filter_cases.final_error("Exception was raised\nE   real") == "real"

    def test_change_pattern_is_the_substitution(self):
        pattern = filter_cases.change_pattern(code("x = main()", "x = cli.main()"))
        assert pattern == {((), ("cli", "."))}          # "cli ." is inserted before "main"

    def test_pattern_of_a_rename(self):
        pattern = filter_cases.change_pattern(code("x = old_name(1)", "x = new_name(1)"))
        assert pattern == {(("old_name",), ("new_name",))}

    def test_same_substitution_in_different_tests_gives_the_same_pattern(self):
        a = filter_cases.change_pattern(code("a = main(1)", "a = cli.main(1)"))
        b = filter_cases.change_pattern(code("a = main(1)", "a = cli.main(1)"))
        assert a == b

    def test_different_substitution_gives_a_different_pattern(self):
        a = filter_cases.change_pattern(code("a = main()", "a = cli.main()"))
        b = filter_cases.change_pattern(code("a = helper()", "a = utils.helper()"))
        assert a != b

    def test_pattern_empty_without_blocks(self):
        assert filter_cases.change_pattern("nothing") == frozenset()


class TestCap:
    def run_filter(self, tmp_path, rows, cap=5):
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, rows)
        report = filter_cases.run(str(src), str(dst), [filter_cases.CapSamePairErrorChange(cap)])
        return report, read(dst)

    def test_repeated_repair_in_one_pair_is_capped(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(8)]
        report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 5
        assert report["initial"]["cases"] == 8 and report["final"]["cases"] == 5

    def test_first_cases_in_file_order_are_kept(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(8)]
        _report, out = self.run_filter(tmp_path, rows, cap=3)
        assert [r[6] for r in out[1:]] == ["t0", "t1", "t2"]

    def test_different_error_is_not_merged(self, tmp_path):
        rows = [case(test="a%d" % i) for i in range(6)] + [case(test="b%d" % i, error="TypeError: other") for i in range(6)]
        _report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 10

    def test_different_change_pattern_is_not_merged(self, tmp_path):
        rows = [case(test="a%d" % i) for i in range(6)]
        rows += [case(test="b%d" % i, broken="y = helper()", fixed="y = utils.helper()") for i in range(6)]
        _report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 10

    def test_different_commit_pair_is_not_merged(self, tmp_path):
        rows = [case(pair=("b1", "r1"), test="a%d" % i) for i in range(6)]
        rows += [case(pair=("b2", "r2"), test="b%d" % i) for i in range(6)]
        _report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 10

    def test_different_repository_with_same_hashes_is_not_merged(self, tmp_path):
        rows = [case(name="o/a", test="a%d" % i) for i in range(6)] + [case(name="o/b", test="b%d" % i) for i in range(6)]
        _report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 10

    def test_cap_one_keeps_one_per_group(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(4)] + [case(test="u", error="TypeError: x")]
        _report, out = self.run_filter(tmp_path, rows, cap=1)
        assert len(out) - 1 == 2

    def test_small_groups_are_untouched(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(3)]
        report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 3 and report["steps"][0]["removed"] == 0

    def test_cases_without_error_lines_are_grouped_by_empty_error(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(7)]
        for r in rows:
            r[5] = "no error lines in this log"
        _report, out = self.run_filter(tmp_path, rows)
        assert len(out) - 1 == 5


class TestFormatAndReport:
    def test_rows_are_written_unchanged_with_multiline_fields(self, tmp_path):
        row = case(test="multi")
        row[1] = row[1] + "line with | pipe and \"quotes\"\nand a second line\n"
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, [row])
        filter_cases.run(str(src), str(dst), [filter_cases.CapSamePairErrorChange(5)])
        assert read(dst) == read(src)
        assert dst.read_bytes() == src.read_bytes()

    def test_report_numbers(self, tmp_path):
        rows = [case(name="o/a", test="a%d" % i) for i in range(8)] + [case(name="o/b", pair=("b9", "r9"), test="b")]
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, rows)
        report = filter_cases.run(str(src), str(dst), [filter_cases.CapSamePairErrorChange(5)])
        step = report["steps"][0]
        assert (report["initial"]["cases"], step["cases_before"], step["cases_after"], step["removed"]) == (9, 9, 6, 3)
        assert step["removed_percent"] == pytest.approx(33.33, abs=0.01)
        assert step["after"]["repositories"] == 2 and step["after"]["commit_pairs"] == 2
        assert step["after"]["cases_per_pair_max"] == 5 and step["before"]["cases_per_pair_max"] == 8
        assert step["details"] == {"groups": 2, "groups_with_more_than_cap": 1, "largest_group": 8}
        assert report["final"] == step["after"]

    def test_markdown_has_the_chain(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(8)]
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, rows)
        text = filter_cases.markdown(filter_cases.run(str(src), str(dst), [filter_cases.CapSamePairErrorChange(5)]))
        assert "Chain: 8 -> same_pair_error_change_cap: 5" in text
        assert "| initial | 8 |" in text

    def test_dry_run_writes_no_csv(self, tmp_path):
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, [case(test="t%d" % i) for i in range(8)])
        report = filter_cases.run(str(src), str(dst), [filter_cases.CapSamePairErrorChange(5)], dry_run=True)
        assert not dst.exists() and report["output"] is None and report["final"]["cases"] == 5

    def test_malformed_row_stops_the_run(self, tmp_path):
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, [case()[:5]])
        with pytest.raises(ValueError):
            filter_cases.run(str(src), str(dst), [filter_cases.CapSamePairErrorChange(5)])


class TestCommandLine:
    def test_writes_csv_and_reports(self, tmp_path, capsys):
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, [case(test="t%d" % i) for i in range(8)])
        filter_cases.main([str(src), str(dst), "--cap", "2"])
        assert len(read(dst)) - 1 == 2
        data = json.loads((tmp_path / "out.csv.report.json").read_text())
        assert [s["name"] for s in data["steps"]] == ["environment_errors", "non_repairs", "same_pair_error_change_cap"]
        assert data["steps"][2]["parameters"]["cap"] == 2
        assert (tmp_path / "out.csv.report.md").read_text().startswith("| Step |")
        assert ("Chain: 8 -> environment_errors: 8 -> non_repairs: 8 -> same_pair_error_change_cap: 2"
                in capsys.readouterr().out)

    def test_steps_option_selects_and_orders_steps(self, tmp_path, capsys):
        src = tmp_path / "in.csv"
        write(src, [case(test="t%d" % i) for i in range(8)])
        filter_cases.main([str(src), "--dry-run", "--steps", "same_pair_error_change_cap,non_repairs"])
        assert "Chain: 8 -> same_pair_error_change_cap: 5 -> non_repairs: 5" in capsys.readouterr().out

    def test_unknown_step_is_refused(self, tmp_path):
        src = tmp_path / "in.csv"
        write(src, [case()])
        with pytest.raises(SystemExit):
            filter_cases.main([str(src), "--dry-run", "--steps", "nonsense"])

    def test_refuses_to_overwrite(self, tmp_path):
        src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
        write(src, [case()])
        dst.write_text("precious")
        with pytest.raises(SystemExit):
            filter_cases.main([str(src), str(dst)])
        assert dst.read_text() == "precious"
        filter_cases.main([str(src), str(dst), "--force"])
        assert len(read(dst)) == 2

    def test_refuses_output_equal_to_input(self, tmp_path):
        src = tmp_path / "in.csv"
        write(src, [case()])
        with pytest.raises(SystemExit):
            filter_cases.main([str(src), str(src), "--force"])

    def test_output_needed_without_dry_run(self, tmp_path):
        src = tmp_path / "in.csv"
        write(src, [case()])
        with pytest.raises(SystemExit):
            filter_cases.main([str(src)])

    def test_dry_run_only_prints(self, tmp_path, capsys):
        src = tmp_path / "in.csv"
        write(src, [case(test="t%d" % i) for i in range(8)])
        filter_cases.main([str(src), "--dry-run"])
        assert ("Chain: 8 -> environment_errors: 8 -> non_repairs: 8 -> same_pair_error_change_cap: 5"
                in capsys.readouterr().out)
        assert [p.name for p in tmp_path.iterdir()] == ["in.csv"]

    def test_cap_must_be_positive(self, tmp_path):
        src = tmp_path / "in.csv"
        write(src, [case()])
        with pytest.raises(SystemExit):
            filter_cases.main([str(src), "--dry-run", "--cap", "0"])


def run_steps(tmp_path, rows, steps):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    write(src, rows)
    return filter_cases.run(str(src), str(dst), steps), read(dst)


class TestEnvironmentErrors:
    def kept(self, tmp_path, error, **kw):
        report, out = run_steps(tmp_path, [case(error=error, **kw)], [filter_cases.EnvironmentErrors()])
        return len(out) - 1, report

    def test_third_party_missing_module_is_dropped(self, tmp_path):
        n, report = self.kept(tmp_path, "ModuleNotFoundError: No module named 'mcp'")
        assert n == 0
        assert report["steps"][0]["details"]["dropped_by_reason"] == {"missing third-party module": 1}

    def test_own_module_named_in_the_repository_is_kept(self, tmp_path):
        n, _ = self.kept(tmp_path, "ModuleNotFoundError: No module named 'sqlit.db'", name="Maxteabag/sqlit")
        assert n == 1

    def test_own_module_edited_by_the_repair_is_kept(self, tmp_path):
        n, _ = self.kept(tmp_path, "ModuleNotFoundError: No module named 'pkg.old'", name="o/r",
                         broken="from pkg.old import f", fixed="from pkg.new import f")
        assert n == 1

    def test_module_named_in_the_test_path_is_kept(self, tmp_path):
        rows = [case(error="ImportError: No module named 'localhelpers'")]
        rows[0][2] = "tests/localhelpers/test_a.py"
        _report, out = run_steps(tmp_path, rows, [filter_cases.EnvironmentErrors()])
        assert len(out) - 1 == 1

    @pytest.mark.parametrize("error", [
        "requests.exceptions.ConnectionError: HTTPConnectionPool(host='x', port=1): Max retries exceeded",
        "httpx.ReadTimeout: Request timed out",
        "urllib.error.URLError: <urlopen error [Errno 111] Connection refused>",
        "agent_reach.transcribe.MissingDependency: ffprobe not found in PATH",
        "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'",
        "openai.OpenAIError: Missing credentials. Please pass an `api_key`",
        "OSError: pytest: reading from stdin while output is captured!  Consider using `-s`.",
        "PermissionError: [Errno 13] Permission denied",
    ])
    def test_environment_errors_are_dropped(self, tmp_path, error):
        n, _ = self.kept(tmp_path, error)
        assert n == 0

    @pytest.mark.parametrize("error", [
        "NameError: name 'main' is not defined",
        "AttributeError: 'Foo' object has no attribute 'bar'",
        "TypeError: demo() missing 1 required keyword-only argument: 'state_secret'",
        "TypeError: run_mypy_check() missing 1 required positional argument: 'python_executable'",
        "AssertionError: assert 'No data available' in 'DNS resolution failed for m.test: Name or service not known'",
        "AssertionError: 'token' not found in 'name: translate'",
        "AttributeError: <module 'docksurf_py.docker' from 'x'> has no attribute 'run'",
        "",
    ])
    def test_code_level_errors_are_kept(self, tmp_path, error):
        n, _ = self.kept(tmp_path, error)
        assert n == 1

    def test_file_not_found_is_kept_and_counted(self, tmp_path):
        n, report = self.kept(tmp_path, "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmpab12/book.epub'")
        assert n == 1
        assert report["steps"][0]["details"]["file_not_found_kept_not_dropped"] == 1


class TestNonRepairs:
    def kept(self, tmp_path, broken, fixed):
        report, out = run_steps(tmp_path, [case(broken=broken, fixed=fixed)], [filter_cases.NonRepairs()])
        return len(out) - 1, report["steps"][0]["details"]

    def test_normal_repair_is_kept(self, tmp_path):
        n, details = self.kept(tmp_path, "x = old()", "x = new()")
        assert n == 1 and details["cases_dropped"] == 0

    @pytest.mark.parametrize("fixed", ["@pytest.mark.skip\ndef test_x(): pass", "pytest.skip('later')", "pass",
                                        "self.skipTest('no')", "@pytest.mark.xfail\nx = 1"])
    def test_repair_that_adds_a_skip_is_dropped(self, tmp_path, fixed):
        n, details = self.kept(tmp_path, "x = old()", fixed)
        assert n == 0 and details["matches_per_reason (a case can match several)"]["adds skip/xfail/pass"] == 1

    def test_skip_already_in_the_broken_block_is_not_blamed_on_the_repair(self, tmp_path):
        n, _ = self.kept(tmp_path, "@pytest.mark.skip\nx = old()", "@pytest.mark.skip\nx = new()")
        assert n == 1

    def test_mostly_deletion_is_dropped(self, tmp_path):
        broken = "\n".join("assert value_%d == %d" % (i, i) for i in range(30))
        n, details = self.kept(tmp_path, broken, "assert value_0 == 0")
        assert n == 0 and details["matches_per_reason (a case can match several)"]["mostly deletion"] == 1

    def test_small_block_is_never_a_deletion(self, tmp_path):
        n, _ = self.kept(tmp_path, "assert a == 1\nassert b == 2", "assert a == 1")
        assert n == 1

    def test_huge_edit_is_dropped(self, tmp_path):
        broken = "\n".join("old_line_%d()" % i for i in range(30))
        fixed = "\n".join("new_line_%d()" % i for i in range(30))
        n, details = self.kept(tmp_path, broken, fixed)
        assert n == 0 and details["matches_per_reason (a case can match several)"]["more than 40 changed lines"] == 1

    def test_edit_of_exactly_the_limit_is_kept(self, tmp_path):
        broken = "\n".join("same_%d()" % i for i in range(30)) + "\n" + "\n".join("old_%d()" % i for i in range(20))
        fixed = "\n".join("same_%d()" % i for i in range(30)) + "\n" + "\n".join("new_%d()" % i for i in range(20))
        n, _ = self.kept(tmp_path, broken, fixed)
        assert n == 1                                    # 20 removed + 20 added = 40, not more than 40

    def test_one_case_matching_several_reasons_is_dropped_once(self, tmp_path):
        broken = "\n".join("assert value_%d == %d" % (i, i) for i in range(50))
        n, details = self.kept(tmp_path, broken, "pytest.skip('x')")
        assert n == 0 and details["cases_dropped"] == 1


class TestSteps:
    def test_report_has_before_and_after_for_every_step(self, tmp_path):
        rows = [case(test="t%d" % i) for i in range(7)] + [case(error="ModuleNotFoundError: No module named 'mcp'", test="e")]
        rows += [case(test="s", fixed="pytest.skip('x')")]
        report, out = run_steps(tmp_path, rows, [filter_cases.EnvironmentErrors(), filter_cases.NonRepairs(),
                                                  filter_cases.CapSamePairErrorChange(5)])
        steps = report["steps"]
        assert [(s["cases_before"], s["cases_after"]) for s in steps] == [(9, 8), (8, 7), (7, 5)]
        assert len(out) - 1 == 5 and report["final"]["cases"] == 5
        text = filter_cases.markdown(report)
        assert "environment_errors: 9 -> 8" in text and "non_repairs: 8 -> 7" in text
        assert "same_pair_error_change_cap: 7 -> 5" in text
        assert "Chain: 9 -> environment_errors: 8 -> non_repairs: 7 -> same_pair_error_change_cap: 5" in text

    def test_cleaning_runs_before_the_cap_so_good_cases_are_not_pushed_out(self, tmp_path):
        rows = [case(test="skipped%d" % i, fixed="pytest.skip('x')") for i in range(5)] + [case(test="good%d" % i) for i in range(3)]
        _report, out = run_steps(tmp_path, rows, [filter_cases.NonRepairs(), filter_cases.CapSamePairErrorChange(5)])
        assert sorted(r[6] for r in out[1:]) == ["good0", "good1", "good2"]
