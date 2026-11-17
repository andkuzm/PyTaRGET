"""Java plausibility (TaRBench rows): applying a candidate, JDK choice, verdict classification, and the rate."""
import json
from pathlib import Path

import pytest

from data_processing import java_plausibility as jp

HOMES = {"8": "/jdk8", "11": "/jdk11", "17": "/jdk17", "21": "/jdk21", "maven": "/mvn"}


class TestApplyCandidate:
    FILE = "class A {\n  void t() {\n    int a = 1;\n    int b = 2;\n    int c = 3;\n  }\n}"

    def row(self, *numbers):
        return {"hunk": {"targetChanges": [{"lineNo": n, "type": "ADD", "line": ""} for n in numbers]}}

    def test_one_line_is_replaced(self):
        out = jp.apply_candidate(self.FILE, self.row(4), "    int b = 20;")
        assert out.split("\n")[3] == "    int b = 20;" and len(out.split("\n")) == len(self.FILE.split("\n"))

    def test_several_target_lines_become_the_candidate_at_the_first_position(self):
        out = jp.apply_candidate(self.FILE, self.row(3, 4), "    int ab = 12;")
        assert out.split("\n")[2:5] == ["    int ab = 12;", "    int c = 3;", "  }"]
        assert len(out.split("\n")) == len(self.FILE.split("\n")) - 1

    def test_a_multi_line_candidate_is_inserted_as_is(self):
        out = jp.apply_candidate(self.FILE, self.row(4), "    int b = 2;\n    int d = 4;")
        assert "    int d = 4;" in out and out.count("\n") == self.FILE.count("\n") + 1

    def test_no_target_lines_means_no_result(self):
        assert jp.apply_candidate(self.FILE, {"hunk": {"targetChanges": []}}, "x") is None


class TestJavaVersion:
    POM = """<project xmlns="http://maven.apache.org/POM/4.0.0"><properties><maven.compiler.source>1.8</maven.compiler.source></properties></project>"""
    POM_PLUGIN = """<project xmlns="http://maven.apache.org/POM/4.0.0"><build><plugins><plugin><artifactId>maven-compiler-plugin</artifactId>
      <configuration><release>17</release></configuration></plugin></plugins></build></project>"""
    POM_PROP_REF = """<project xmlns="http://maven.apache.org/POM/4.0.0"><properties><jv>11</jv></properties><build><plugins><plugin>
      <artifactId>maven-compiler-plugin</artifactId><configuration><source>${jv}</source></configuration></plugin></plugins></build></project>"""

    def test_versions_come_from_properties_or_the_compiler_plugin(self):
        assert jp.detect_java_versions(self.POM) == ["1.8"]
        assert jp.detect_java_versions(self.POM_PLUGIN) == ["17"]
        assert jp.detect_java_versions(self.POM_PROP_REF) == ["11"]
        assert jp.detect_java_versions("<not xml") == []

    def test_the_home_is_the_smallest_available_version_that_is_enough(self):
        assert jp.pick_java_home(HOMES, ["1.8"])[0] == "8"
        assert jp.pick_java_home(HOMES, ["1.7"])[0] == "8"
        assert jp.pick_java_home(HOMES, ["12"])[0] == "17"
        assert jp.pick_java_home(HOMES, ["17", "1.8"])[0] == "17"
        assert jp.pick_java_home(HOMES, ["99"])[0] == "11"          # nothing is enough: the default
        assert jp.pick_java_home(HOMES, [])[0] == "11"
        assert jp.pick_java_home(HOMES, ["1.8"], forced="11")[0] == "11"

    def test_the_nearest_pom_above_the_test_is_used(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project/>")
        (tmp_path / "mod" / "src" / "test").mkdir(parents=True)
        (tmp_path / "mod" / "pom.xml").write_text("<project/>")
        test = tmp_path / "mod" / "src" / "test" / "T.java"
        assert jp.find_parent_pom(test, tmp_path) == tmp_path / "mod" / "pom.xml"
        (tmp_path / "other" / "t").mkdir(parents=True)
        assert jp.find_parent_pom(tmp_path / "other" / "t" / "T.java", tmp_path) == tmp_path / "pom.xml"


class TestClassify:
    OK = "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.1 s - in a.T"
    REL = "src/test/java/a/T.java"

    def run(self, code, log):
        return jp.classify(code, log, self.REL, "T", "testIt")

    def test_success_needs_exactly_one_passing_test(self):
        assert self.run(0, self.OK) == jp.SUCCESS
        assert self.run(0, self.OK.replace("Tests run: 1", "Tests run: 2")) == jp.TEST_NOT_EXECUTED
        assert self.run(0, "BUILD SUCCESS") == jp.TEST_NOT_EXECUTED

    def test_timeout(self):
        assert self.run(124, "") == jp.TIMEOUT

    def test_compile_error_in_the_test_file_is_valid_but_elsewhere_it_is_unrelated(self):
        log = f"[ERROR] COMPILATION ERROR :\n[ERROR] /work/proj/{self.REL}:[12,5] cannot find symbol\n"
        assert self.run(1, log) == jp.COMPILE_ERR
        assert self.run(1, "[ERROR] COMPILATION ERROR :\n[ERROR] /work/proj/src/main/java/B.java:[3,1] bad\n") == jp.UNRELATED_COMPILE_ERR

    def test_test_failures(self):
        assert self.run(1, "[ERROR]   T.testIt:41 expected:<1> but was:<2>") == jp.FAILURE
        assert self.run(1, "Tests run: 1, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.2 s") == jp.FAILURE
        assert self.run(1, "Tests run: 1, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 0.2 s") == jp.FAILURE

    def test_invalid_executions(self):
        assert self.run(1, "Could not resolve dependencies for project") == jp.DEPENDENCY_ERROR
        assert self.run(1, "java.lang.Exception: No tests found matching Method testIt") == jp.TEST_MATCH_FAILURE
        assert self.run(1, "something else entirely") == jp.UNKNOWN

    def test_method_names(self):
        assert jp.split_method("com.a.b.CountryTest.testFetchRequest()") == ("CountryTest", "testFetchRequest")


def test_the_rate_counts_a_case_when_any_candidate_succeeds_and_drops_invalid_cases():
    results = {
        "a": {"status": "evaluated", "verdicts": [{"status": "failure", "success": 0}, {"status": "success", "success": 1}]},
        "b": {"status": "evaluated", "verdicts": [{"status": "success", "success": 1}]},
        "c": {"status": "evaluated", "verdicts": [{"status": "failure", "success": 0}, {"status": "failure", "success": 0}]},
        "d": {"status": "evaluated", "verdicts": [{"status": "timeout", "success": 0}] * 4},               # invalid: dropped
        "e": {"status": "evaluated", "verdicts": []},
    }
    s = jp.summarize(results)
    assert s["cases"] == 3 and s["dropped_invalid"] == 1
    assert s["plausible@1"] == pytest.approx(33.33, abs=0.01) and s["plausible@5"] == pytest.approx(66.67, abs=0.01)


def test_remove_unnecessary_plugins(tmp_path):
    pom = tmp_path / "pom.xml"
    pom.write_text("<project><build><plugins><plugin><groupId>org.codehaus.mojo</groupId><artifactId>findbugs-maven-plugin</artifactId>"
                   "<version>1</version></plugin><plugin><groupId>x</groupId><artifactId>keep</artifactId></plugin></plugins></build></project>")
    jp.remove_unnecessary_plugins(pom)
    text = pom.read_text()
    assert "findbugs" not in text and "keep" in text


class TestControl:
    """evaluate_case with a fake build: the reference runs first and an environment that cannot pass it blames nobody."""

    @staticmethod
    def runner(tmp_path, statuses):
        project = tmp_path / "proj"
        (project / "src").mkdir(parents=True)
        (project / "src" / "T.java").write_text("a\nb\nc\n")
        runner = jp.JavaPlausibility.__new__(jp.JavaPlausibility)
        runner.log_dir = None
        runner.prepare = lambda repo, commit: project
        calls = []

        def execute(project_, row, original, candidate, rel, cls, method, rank=None):
            calls.append(candidate)
            return statuses.get(candidate, "failure"), "11", ""

        runner.execute = execute
        return runner, calls

    ROW = {"ID": "o/r:1", "aCommit": "abc", "aPath": "src/T.java", "name": "p.T.testIt()", "hunk": {"targetChanges": [{"lineNo": 2}]}}

    def test_a_failing_reference_means_reference_fails_and_no_candidate_runs(self, tmp_path):
        runner, calls = self.runner(tmp_path, {"REF": "compile_error"})
        out = runner.evaluate_case(self.ROW, ["x", "y"], "REF")
        assert out["status"] == "reference_fails" and out["control"] == "compile_error" and calls == ["REF"]

    def test_a_passing_reference_is_run_once_and_candidates_equal_to_it_reuse_that_verdict(self, tmp_path):
        runner, calls = self.runner(tmp_path, {"REF": "success", "good": "success"})
        out = runner.evaluate_case(self.ROW, ["bad", " REF ", "good", "bad"], "REF")
        assert [v["status"] for v in out["verdicts"]] == ["failure", "success", "success", "failure"]
        assert calls == ["REF", "bad", "good"]                    # the reference and the repeated candidate are not run again

    def test_without_control_a_candidate_equal_to_the_reference_counts_without_running(self, tmp_path):
        runner, calls = self.runner(tmp_path, {})
        out = runner.evaluate_case(self.ROW, ["REF"], "REF", control=False)
        assert out["verdicts"][0]["success"] == 1 and calls == []

    def test_the_summary_counts_reference_fails_apart(self):
        s = jp.summarize({"a": {"status": "reference_fails", "verdicts": []},
                          "b": {"status": "evaluated", "verdicts": [{"status": "success", "success": 1}]}})
        assert s["cases"] == 1 and s["reference_fails"] == 1 and s["plausible@1"] == 100.0


class TestUnion:
    """Incremental scoring: a second model only adds the candidates that were not judged yet; the control runs once."""

    def test_the_second_run_judges_only_new_candidates_and_never_repeats_the_control(self, tmp_path):
        runner, calls = TestControl.runner(tmp_path, {"REF": "success", "a": "success"})
        first = runner.evaluate_union(TestControl.ROW, ["a", "b"], "REF")
        assert first["status"] == "evaluated" and calls == ["REF", "a", "b"]
        assert first["cache"]["REF"][0] == "success" and first["cache"]["a"][0] == "success" and first["cache"]["b"][0] == "failure"
        calls.clear()
        second = runner.evaluate_union(TestControl.ROW, [" a ", "c", "REF", ""], "REF", prior=first)
        assert calls == ["c"] and set(second["cache"]) == {"REF", "a", "b", "c"}
        calls.clear()
        assert runner.evaluate_union(TestControl.ROW, ["a", "c"], "REF", prior=second)["status"] == "evaluated" and calls == []

    def test_a_failed_control_is_remembered_and_nothing_else_runs(self, tmp_path):
        runner, calls = TestControl.runner(tmp_path, {"REF": "dependency_error"})
        first = runner.evaluate_union(TestControl.ROW, ["a"], "REF")
        assert first["status"] == "reference_fails" and calls == ["REF"]
        calls.clear()
        again = runner.evaluate_union(TestControl.ROW, ["b"], "REF", prior=first)
        assert again["status"] == "reference_fails" and calls == []
