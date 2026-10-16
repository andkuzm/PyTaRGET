"""
Plausibility of predicted test repairs: does the repaired test pass on the repaired version of the code?

For every case of a test split the reference repair is located inside the real test at `repaired_hash`, the
prediction is swapped in (re-indented to the reference's margin) and only that test is run with pytest.
A case counts only when the unmodified test passes in the same environment (the control), otherwise the failure
would say nothing about the prediction. Cases whose reference repair cannot be located (deleted lines, changes in
several places) are reported with a reason and left out.

WARNING: this clones third-party repositories and executes their code together with model-generated code.
Run it in a throw-away container or machine.

    python -m data_processing.plausibility_test test.json predictions.json --workdir plausibility --k 5
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from data_processing.metrics import normalize_indentation

DELETED = "# Deleted"  # what the Encoder writes as the reference when the repair only removes lines


@dataclass
class RepairSite:
    nodeid_parts: list  # e.g. ["TestFoo", "test_bar"]
    start: int  # 0-based line of the file where the reference repair begins
    length: int  # number of lines of the reference repair


def find_test_name(annotated_code):
    """The test's name: the first function defined in the annotated test context (imports have none)."""
    context = re.search(r"\[<TESTCONTEXT>](.*?)\[</TESTCONTEXT>]", annotated_code, re.DOTALL)
    for line in (context.group(1) if context else annotated_code).split("\n"):
        match = re.match(r"\s*(?:async\s+)?def\s+(\w+)\s*\(", line)
        if match:
            return match.group(1)
    return None


def iter_functions(node, chain=()):
    """Functions and methods with the names of their enclosing classes; nested functions are not descended into."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from iter_functions(child, chain + (child.name,))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, chain
        else:
            yield from iter_functions(child, chain)


def find_repair_site(source, test_name, reference):
    """Where `reference` (the reference repair) sits in the test `test_name`; None unless it is found exactly once."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.split("\n")
    block = [line.rstrip() for line in reference.strip("\n").rstrip().split("\n")]

    sites = []
    for node, chain in iter_functions(tree):
        if node.name != test_name:
            continue
        first = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
        body = [line.rstrip() for line in lines[first:node.end_lineno]]
        for i in range(len(body) - len(block) + 1):
            # the first line of a test lost its indentation when the dataset was built
            if all(body[i + j] == block[j] or (i + j == 0 and body[0].strip() == block[j].strip())
                   for j in range(len(block))):
                sites.append(RepairSite(list(chain) + [test_name], first + i, len(block)))
                break
    return sites[0] if len(sites) == 1 else None


def common_margin(lines):
    margins = [line[:len(line) - len(line.lstrip())] for line in lines if line.strip()]
    return os.path.commonprefix(margins) if margins else ""


def patch_source(source, site, prediction):
    """`source` with the reference repair replaced by `prediction`, keeping its relative indentation."""
    lines = source.split("\n")
    margin = common_margin(lines[site.start:site.start + site.length])
    new = [margin + line if line else line for line in normalize_indentation(prediction).split("\n")]
    return "\n".join(lines[:site.start] + new + lines[site.start + site.length:])


def verdict_from(returncode, output):
    if returncode == 124:
        return "timeout"
    if returncode == 0 and re.search(r"\d+ passed", output) and "failed" not in output:
        return "pass"  # a skipped or deselected test also exits with 0, that is not a pass
    return "fail" if returncode == 1 else "error"


def load_predictions(path):
    """{ID: (target, candidates)} from the json written by the testers (DataFrame.to_json or a list of records)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {int(data["ID"][k]): (data["target"][k], data["preds"][k]) for k in data["ID"]}
    return {int(r["ID"]): (r["target"], r["preds"]) for r in data}


def summarize(results):
    evaluated = [r for r in results.values() if r["status"] == "evaluated"]
    statuses = {}
    for r in results.values():
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    summary = {"cases": len(results), "evaluated": len(evaluated), "statuses": statuses}
    if evaluated:
        summary["plausible@1"] = round(100 * sum(r["verdicts"][0] == "pass" for r in evaluated) / len(evaluated), 2)
        summary["plausible@k"] = round(100 * sum("pass" in r["verdicts"] for r in evaluated) / len(evaluated), 2)
    return summary


class PlausibilityTester:
    def __init__(self, workdir, k=5, timeout=120, use_venv=True, install_requirements=True,
                 repo_url="https://github.com/{name}.git"):
        self.workdir = Path(workdir)
        self.k = k
        self.timeout = timeout
        self.use_venv = use_venv
        self.install_requirements = install_requirements
        self.repo_url = repo_url
        self.pythons = {}

    # ---- processes -------------------------------------------------------------------------------------------

    def run_cmd(self, cmd, cwd=None, timeout=None):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="ignore",
                                  timeout=timeout)
            return proc.returncode, proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired:
            return 124, ""

    # ---- repositories and environments -----------------------------------------------------------------------

    def prepare(self, name, sha):
        """Clone the repository once and check out `sha`; returns the repository directory."""
        repo_dir = self.workdir / "repos" / name.replace("/", "__")
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            code, out = self.run_cmd(["git", "clone", "--quiet", self.repo_url.format(name=name), str(repo_dir)])
            if code != 0:
                raise RuntimeError(f"clone failed: {out.strip()[-300:]}")
        code, out = self.run_cmd(["git", "checkout", "--quiet", "--force", sha], cwd=repo_dir)
        if code != 0:
            raise RuntimeError(f"checkout failed: {out.strip()[-300:]}")
        return repo_dir

    def python_for(self, name, repo_dir):
        """Interpreter for the repository: its own venv with pytest and the project installed (best effort)."""
        if not self.use_venv:
            return sys.executable
        if name in self.pythons:
            return self.pythons[name]
        venv_dir = self.workdir / "venvs" / name.replace("/", "__")
        python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            self.run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
            self.run_cmd([str(python), "-m", "pip", "install", "-q", "pytest"], timeout=900)
            if self.install_requirements:
                for requirements in sorted(repo_dir.glob("requirements*.txt")):
                    self.run_cmd([str(python), "-m", "pip", "install", "-q", "-r", str(requirements)], timeout=1800)
            if (repo_dir / "setup.py").exists() or (repo_dir / "pyproject.toml").exists():
                self.run_cmd([str(python), "-m", "pip", "install", "-q", "-e", "."], cwd=repo_dir, timeout=1800)
        self.pythons[name] = str(python)
        return self.pythons[name]

    def run_test(self, python, repo_dir, nodeid):
        cmd = [python, "-m", "pytest", "-x", "-q", "--disable-warnings", "-p", "no:cacheprovider", nodeid]
        return verdict_from(*self.run_cmd(cmd, cwd=repo_dir, timeout=self.timeout))

    # ---- cases -----------------------------------------------------------------------------------------------

    def evaluate_case(self, row, candidates):
        result = {"status": None, "nodeid": None, "verdicts": []}
        reference = row["output"]
        test_name = find_test_name(row["annotated_code"])
        if test_name is None:
            return {**result, "status": "no_test_name"}
        if reference.strip() == DELETED:
            return {**result, "status": "deleted_repair"}
        try:
            repo_dir = self.prepare(row["repository_name"], row["repaired_hash"])
        except RuntimeError as error:
            return {**result, "status": "setup_failed", "detail": str(error)}

        test_file = repo_dir / row["relative_path"]
        if not test_file.exists():
            return {**result, "status": "test_file_missing"}
        original = test_file.read_bytes()
        site = find_repair_site(original.decode("utf-8", errors="ignore"), test_name, reference)
        if site is None:
            return {**result, "status": "repair_site_not_found"}
        nodeid = "::".join([Path(row["relative_path"]).as_posix()] + site.nodeid_parts)
        result["nodeid"] = nodeid

        python = self.python_for(row["repository_name"], repo_dir)
        control = self.run_test(python, repo_dir, nodeid)
        if control != "pass":
            return {**result, "status": "reference_fails", "detail": control}

        known = {normalize_indentation(reference): "pass"}  # equal to the reference: the control already passed
        try:
            for candidate in candidates[:self.k]:
                key = normalize_indentation(candidate)
                if not key.strip():
                    verdict = "empty"
                elif key in known:
                    verdict = known[key]
                else:
                    test_file.write_text(patch_source(original.decode("utf-8"), site, candidate), encoding="utf-8")
                    verdict = self.run_test(python, repo_dir, nodeid)
                    known[key] = verdict
                result["verdicts"].append(verdict)
        finally:
            test_file.write_bytes(original)
        return {**result, "status": "evaluated"}

    def run(self, test_json, predictions_json, out_path):
        rows = json.loads(Path(test_json).read_text(encoding="utf-8"))
        predictions = load_predictions(predictions_json)
        out_path = Path(out_path)
        results = {}
        if out_path.exists():
            results = {int(k): v for k, v in json.loads(out_path.read_text(encoding="utf-8"))["results"].items()}
            print(f"Resuming, {len(results)} cases already done")

        todo = sorted((i for i in predictions if i not in results),
                      key=lambda i: (rows[i]["repository_name"], rows[i]["repaired_hash"]))
        for n, i in enumerate(todo, 1):
            target, candidates = predictions[i]
            row = rows[i]
            if target.rstrip() != row["output"].rstrip():
                results[i] = {"status": "id_mismatch", "nodeid": None, "verdicts": []}
            else:
                results[i] = self.evaluate_case(row, candidates)
            print(f"[{n}/{len(todo)}] case {i} {row['repository_name']}: {results[i]['status']} {results[i]['verdicts']}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"summary": summarize(results), "results": results}, indent=1),
                                encoding="utf-8")

        summary = summarize(results)
        print(summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_json", help="splits/test.json of the experiment")
    parser.add_argument("predictions_json", help="test_predictions.json or <model>_llm_test_predictions.json")
    parser.add_argument("--workdir", default="plausibility")
    parser.add_argument("--out", default=None, help="result file (default: <workdir>/plausibility.json)")
    parser.add_argument("--k", type=int, default=5, help="how many top candidates to execute per case")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per test run")
    parser.add_argument("--no-venv", action="store_true", help="use the current interpreter instead of a venv per repo")
    args = parser.parse_args()
    PlausibilityTester(args.workdir, args.k, args.timeout, use_venv=not args.no_venv).run(
        args.test_json, args.predictions_json, args.out or Path(args.workdir) / "plausibility.json")
