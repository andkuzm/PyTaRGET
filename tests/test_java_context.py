"""Java context (failure output of the broken run, definitions) and the case keys shared with the Python ablation code."""
from data_processing import context_ablation as ca
from data_processing import java_context as jc

LOG = """[INFO] Scanning for projects...
[INFO] Downloading from central: https://repo.maven.apache.org/maven2/foo
[ERROR] COMPILATION ERROR :
[ERROR] /gpfs/helios/home/u/work/repos/o__r/src/test/java/p/FooTest.java:[12,24] cannot find symbol
[ERROR]   symbol:   method plus(int,int)
[ERROR] /gpfs/helios/home/u/work/repos/o__r/src/test/java/p/FooTest.java:[12,24] cannot find symbol
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.8.0:testCompile (default-testCompile) on project r: Compilation failure
[ERROR]
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] -> [Help 1]
"""


def test_failure_text_keeps_error_lines_without_paths_noise_and_repeats():
    t = jc.failure_text(LOG)
    assert "cannot find symbol" in t and "symbol:   method plus(int,int)" in t and "src/test/java/p/FooTest.java:[12,24]" in t
    assert "/gpfs/" not in t and "Downloading" not in t and "Help 1" not in t and "full stack trace" not in t
    assert t.count("FooTest.java:[12,24]") == 1                       # the repeated error line is kept once


def test_failure_text_is_cut_to_head_and_tail():
    t = jc.failure_text("\n".join(f"[ERROR] line {i} " + "x" * 80 for i in range(200)), max_chars=500)
    assert len(t) <= 520 and "line 0" in t and "line 199" in t and "..." in t


def test_symbols_skip_keywords_and_test_boilerplate():
    assert jc.symbols_from_broken("assertEquals(3, new Foo().plus(1, 2));") == ["Foo", "plus"]


def test_java_index_has_main_types_and_methods_but_not_tests(tmp_path):
    (tmp_path / "src/main/java/p").mkdir(parents=True)
    (tmp_path / "src/test/java/p").mkdir(parents=True)
    (tmp_path / "src/main/java/p/Foo.java").write_text("package p;\npublic class Foo extends Bar {\n    public int add(int a, int b) throws IOException {\n        return a + b;\n    }\n    private static <T> List<T> items() {\n        return null;\n    }\n}\n")
    (tmp_path / "src/test/java/p/FooTest.java").write_text("class FooTest {\n    public void secret(int a) {}\n}\n")
    index = jc.index_java_source(tmp_path)
    assert "Foo" in index and "add" in index and "items" in index and "secret" not in index and "FooTest" not in index
    assert index["add"][0][1] == "public int add(int a, int b) throws IOException" and index["Foo"][0][1] == "public class Foo extends Bar"
    text = ca.definitions_context(["Foo", "add"], index)
    assert text.startswith("# src/main/java/p/Foo.java") and "int add(" in text


def test_broken_method_keeps_the_indentation_of_the_after_file():
    row = {"aSource": {"startLine": 2, "code": "x"}, "bSource": {"startLine": 2, "code": "@Test\n    public void t() {\n        old();\n    }"}}
    assert jc.broken_method_text(row, "class A {\n    @Test\n    public void t() {\n        fresh();\n    }\n}").split("\n")[0] == "    @Test"


def test_case_key_for_java_rows_is_the_id_and_python_rows_are_unchanged():
    assert ca.case_key({"ID": "o/r:12", "aCommit": "a"}) == "o/r:12"
    assert ca.case_key({"repository_name": "o/r", "broken_hash": "b", "test_name": "t", "ID": "ignored"}) == "o/r|b|t"
