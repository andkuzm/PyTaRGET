"""
Plausibility of predicted Java test repairs (TaRBench rows): does the repaired test pass on the repaired version of the project?

Follows the method of the TaRGet paper (verdict names and rules are the same, so published verdicts can be compared with ours):
the project is checked out at the repaired commit (`aCommit`), the changed target lines of the test file (`hunk.targetChanges`) are
replaced by the candidate, and only that test runs with Maven on a JDK chosen from the project's pom (default 11, retried with 11 when
the verdict is invalid on another JDK). A candidate equal to the reference counts as a success without running (switch it off with
--no-trust-reference to audit the whole path). Verdicts: success / failure / compile_error are VALID; timeout, dependency_error and the
other statuses are INVALID (a case whose candidates are invalid for 75% or more is dropped from the rate, like in the paper).

WARNING: this clones third-party repositories and runs their build and test code. Run it in a throw-away session or a Slurm job.
Each repository is handled sequentially (its own clone, worktree cleaned between cases); repositories run in parallel threads.

  # check the method on the authors' own candidates (published verdicts are compared afterwards)
  python -m data_processing.java_plausibility --bench ~/references/TaRGet_Results_download/TaRGet_Results/Best_on_TaRBench \
      --published --ids <file with IDs> --max-rank 3 --out check.json --workdir <scratch>
  # score our predictions (position-keyed test_predictions.json next to the subset's splits/test.json)
  python -m data_processing.java_plausibility --split-dir ~/eval_results_java_llm/contrast300 --predictions <dir>/test_predictions.json \
      --out <dir>/java_plausibility.json --workdir <scratch>
"""
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUCCESS, FAILURE, COMPILE_ERR, TIMEOUT = "success", "failure", "compile_error", "timeout"
UNRELATED_COMPILE_ERR, EXPECTED_EXCEPTION_FAILURE = "unrelated_compile_error", "expected_exception_failure"
TEST_MATCH_FAILURE, UNRELATED_FAILURE, DEPENDENCY_ERROR = "test_match_failure", "unrelated_failure", "dependency_error"
TEST_NOT_EXECUTED, UNKNOWN, POM_NOT_FOUND = "test_not_executed", "unknown", "pom_not_found"
VALID = (SUCCESS, FAILURE, COMPILE_ERR)
INVALID_FOR_DROP = (TIMEOUT, DEPENDENCY_ERROR)
DEFAULT_JAVA = "11"
HOMES_FILE = Path.home() / "tools" / "java" / "java_homes.json"

MVN_SKIPS = ["-Djacoco.skip", "-Dcheckstyle.skip", "-Dspotless.apply.skip", "-Drat.skip", "-Denforcer.skip", "-Danimal.sniffer.skip",
             "-Dmaven.javadoc.skip", "-Dmaven.gitcommitid.skip", "-Dfindbugs.skip", "-Dwarbucks.skip", "-Dmodernizer.skip",
             "-Dimpsort.skip", "-Dpmd.skip", "-Dxjc.skip", "-Dair.check.skip-all", "-Dlicense.skip", "-Dremoteresources.skip"]
POM_NS = {"x": "http://maven.apache.org/POM/4.0.0"}
VERSION_PROPERTIES = ["java.version", "version.java", "javaVersion", "jdk.version", "jre.version", "compiler.level",
                      "maven.compiler.release", "maven.compiler.testRelease", "maven.compiler.source", "maven.compiler.target"]
PLUGIN_PATHS = {
    "maven-compiler-plugin": ["./x:configuration/x:release", "./x:configuration/x:source", "./x:configuration/x:target",
                              "./x:executions/x:execution/x:configuration/x:release", "./x:executions/x:execution/x:configuration/x:source",
                              "./x:executions/x:execution/x:configuration/x:target", "./x:configuration/x:jdkToolchain/x:version"],
    "maven-enforcer-plugin": ["./x:executions/x:execution/x:configuration/x:rules/x:requireJavaVersion/x:version"],
}


# ---- applying a candidate --------------------------------------------------------------------------------------------------

def apply_candidate(contents, row, candidate):
    """The file contents with the target lines of the repair replaced by `candidate` (None when the case has no target lines)."""
    changes = (row.get("hunk") or {}).get("targetChanges") or []
    if not changes:
        return None
    lines = contents.split("\n")
    first = changes[0]["lineNo"] - 1
    for change in sorted(changes, key=lambda c: c["lineNo"], reverse=True):
        index = change["lineNo"] - 1
        if 0 <= index < len(lines):
            del lines[index]
    lines.insert(first, candidate)
    return "\n".join(lines)


# ---- Java version -----------------------------------------------------------------------------------------------------------

def _is_float(text):
    try:
        float(text.strip())
        return True
    except (ValueError, AttributeError):
        return False


def _tag_value(root, tag):
    if tag is None or tag.text is None:
        return None
    match = re.search(r"\$\{(.+)\}", tag.text)
    if not match:
        if _is_float(tag.text):
            return tag.text.strip()
        match = re.search(r"\[(.+),.*", tag.text)
        return match.group(1) if match and _is_float(match.group(1)) else None
    prop = root.find(f"./x:properties/x:{match.group(1)}", POM_NS)
    return prop.text.strip() if prop is not None and _is_float(prop.text) else None


def detect_java_versions(pom_text):
    """Java versions a pom asks for (compiler plugin config, enforcer rule, version properties); [] when none or unparseable."""
    try:
        root = ET.fromstring(pom_text.strip())
    except ET.ParseError:
        return []
    def from_plugin(plugin, name):
        found = []
        for path in PLUGIN_PATHS[name]:
            value = _tag_value(root, plugin.find(path, POM_NS))
            if value is not None:
                found.append(value)
        return found
    for main in ("./x:build/x:pluginManagement/x:plugins/x:plugin/[x:artifactId='maven-compiler-plugin']",
                 "./x:build/x:plugins/x:plugin/[x:artifactId='maven-compiler-plugin']"):
        plugin = root.find(main, POM_NS)
        if plugin is not None:
            versions = from_plugin(plugin, "maven-compiler-plugin")
            if versions:
                return versions[:1]
    versions = []
    for name in PLUGIN_PATHS:
        for plugin in root.findall(f".//x:plugin/[x:artifactId='{name}']", POM_NS):
            versions.extend(from_plugin(plugin, name))
    if not versions:
        for prop in VERSION_PROPERTIES:
            node = root.find(f"./x:properties/x:{prop}", POM_NS)
            if node is not None and _is_float(node.text):
                versions.append(node.text.strip())
    return versions


def pick_java_home(homes, versions, forced=None):
    """JDK home: the forced version, else the smallest available one that is >= the highest version asked for (default 11)."""
    available = sorted(int(k) for k in homes if k.isdigit())
    if forced and str(forced) in homes:
        return str(forced), homes[str(forced)]
    wanted = [int(float(v.replace("1.", "", 1))) for v in versions if _is_float(v)]
    if not wanted:
        return DEFAULT_JAVA, homes[DEFAULT_JAVA]
    top = max(wanted)
    for version in available:
        if version >= top:
            return str(version), homes[str(version)]
    return DEFAULT_JAVA, homes[DEFAULT_JAVA]


def find_parent_pom(test_path, root):
    current = test_path.parent
    while True:
        if (current / "pom.xml").exists():
            return current / "pom.xml"
        if current == root or current == current.parent:
            return None
        current = current.parent


# ---- verdicts ---------------------------------------------------------------------------------------------------------------

def classify(returncode, log, test_rel_path, test_class, test_method):
    """Verdict status of one Maven run (same rules as the paper's code)."""
    if returncode == 124:
        return TIMEOUT
    if returncode == 0:
        for runs, failures, errors, skips in re.findall(r"^.*Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+).*Time elapsed.*$",
                                                         log, re.MULTILINE):
            if (int(runs), int(failures), int(errors), int(skips)) == (1, 0, 0, 0):
                return SUCCESS
        return TEST_NOT_EXECUTED
    if "COMPILATION ERROR" in log or "Compilation failure:" in log:
        if re.search(rf"^\[ERROR\]\s*/.+/{re.escape(str(test_rel_path))}:\[(\d+),\d+\].*$", log, re.MULTILINE):
            return COMPILE_ERR
    for regex in (rf"^\[ERROR\]\s*{test_class}\.{test_method}:(\d+).*$", rf"^\s*at .+{test_class}.{test_method}\({test_class}\.java:(\d+)\).*$"):
        if re.search(regex, log, re.MULTILINE):
            return FAILURE
    match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+).*Time elapsed.*", log)
    if match:
        if int(match.group(2)) > 0 or int(match.group(3)) > 0:
            return FAILURE
        if int(match.group(1)) == 0 or int(match.group(4)) > 0:
            return TEST_NOT_EXECUTED
    if "COMPILATION ERROR" in log or "Compilation failure:" in log:
        return UNRELATED_COMPILE_ERR
    if "java.lang.AssertionError: Expected exception:" in log:
        return EXPECTED_EXCEPTION_FAILURE
    if "java.lang.Exception: No tests found matching Method" in log:
        return TEST_MATCH_FAILURE
    if "<<< ERROR!" in log:
        return UNRELATED_FAILURE
    if "Could not resolve dependencies" in log or "Non-resolvable parent POM" in log:
        return DEPENDENCY_ERROR
    return UNKNOWN


def split_method(full_name):
    """com.pkg.Class.method(args) -> (Class, method); nested classes keep the innermost class name."""
    name = full_name.split("(")[0]
    parts = name.split(".")
    return parts[-2], parts[-1]


def remove_unnecessary_plugins(pom_path):
    """The three plugins the authors strip from poms (they break offline or unusual builds)."""
    if not pom_path.exists():
        return
    try:
        pom = pom_path.read_text()
    except OSError:
        return
    new = pom
    for group, artifact in ((r"org\.codehaus\.mojo", r"findbugs-maven-plugin"), (r"pl\.project13\.maven", r"git-commit-id-plugin"),
                            (r"io\.github\.git-commit-id", r"git-commit-id-maven-plugin")):
        match = re.search(rf"<plugin>\s*<groupId>{group}</groupId>\s*<artifactId>{artifact}</artifactId>.*?</plugin>", new, re.DOTALL)
        if match:
            new = new.replace(match.group(), "")
    if new != pom:
        pom_path.write_text(new)


# ---- running ----------------------------------------------------------------------------------------------------------------

def run_group(cmd, cwd, env, timeout):
    """Run a command in its own process group; a timeout kills the whole group. Returns (returncode, output)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode("utf-8", errors="ignore")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        return 124, out.decode("utf-8", errors="ignore")


class JavaPlausibility:
    def __init__(self, workdir, homes_file=HOMES_FILE, timeout=20 * 60, m2=None, log_dir=None, mvn_opts="-Xmx2g"):
        self.workdir = Path(workdir)
        self.homes = json.load(open(homes_file))
        self.mvn = str(Path(self.homes["maven"]) / "bin" / "mvn")
        self.timeout = timeout
        self.m2 = Path(m2) if m2 else self.workdir / "m2"
        self.log_dir = Path(log_dir) if log_dir else None
        self.mvn_opts = mvn_opts
        self.lock = threading.Lock()

    def git(self, repo_dir, *args, timeout=1800):
        code, out = run_group(["git", "-C", str(repo_dir), *args], None, os.environ.copy(), timeout)
        return code, out

    def prepare(self, repo, commit):
        """Clone once (partial clone: blobs on demand), then check out `commit` cleanly. Returns the project directory."""
        project = self.workdir / "repos" / repo.replace("/", "__")
        if not project.exists():
            project.parent.mkdir(parents=True, exist_ok=True)
            code, out = run_group(["git", "clone", "--quiet", "--filter=blob:none", f"https://github.com/{repo}.git", str(project)], None,
                                  os.environ.copy(), 3600)
            if code != 0:
                shutil.rmtree(project, ignore_errors=True)
                raise RuntimeError(f"clone failed: {out.strip()[-200:]}")
        self.git(project, "reset", "--hard", "--quiet")
        self.git(project, "clean", "-fdxq")
        code, out = self.git(project, "checkout", "--quiet", "--force", commit)
        if code != 0:
            self.git(project, "fetch", "--quiet", "origin", commit)
            code, out = self.git(project, "checkout", "--quiet", "--force", commit)
            if code != 0:
                raise RuntimeError(f"checkout failed: {out.strip()[-200:]}")
        return project

    def run_test(self, project, test_rel, test_class, test_method, forced_java=None):
        test_path = project / test_rel
        pom = find_parent_pom(test_path, project)
        if pom is None:
            return POM_NOT_FOUND, ""
        root_pom = project / "pom.xml"
        versions = detect_java_versions(root_pom.read_text(errors="ignore")) if root_pom.exists() else []
        java_version, java_home = pick_java_home(self.homes, versions, forced_java)
        remove_unnecessary_plugins(root_pom)
        remove_unnecessary_plugins(pom)
        self.m2.mkdir(parents=True, exist_ok=True)
        cmd = [self.mvn, "test", "-nsu", "-pl", str(pom.relative_to(project)), "--also-make", "-Dsurefire.failIfNoSpecifiedTests=false",
               "-DfailIfNoTests=false", "-Dmaven.test.skip=false", "-DskipTests=false", f"-Dtest={test_class}#{test_method}", "--batch-mode",
               *MVN_SKIPS, f"-Dmaven.repo.local={self.m2}"]
        env = os.environ.copy()
        env.update({"JAVA_HOME": java_home, "MAVEN_OPTS": self.mvn_opts})
        code, log = run_group(cmd, str(project), env, self.timeout)
        return classify(code, log, test_rel, test_class, test_method), f"{java_version}\n{log[-60000:]}"

    def execute(self, project, row, original, candidate, test_rel, test_class, test_method, rank=None):
        """Apply one candidate, run the test, restore the file. Returns (status, java version used, log)."""
        test_file = project / test_rel
        patched = apply_candidate(original, row, candidate)
        if patched is None:
            return "no_target_lines", None, ""
        try:
            test_file.write_text(patched)
            status, log = self.run_test(project, test_rel, test_class, test_method)
            java_used = log.split("\n", 1)[0]
            if status not in VALID and java_used != DEFAULT_JAVA:               # the paper retries invalid verdicts with Java 11
                status, log = self.run_test(project, test_rel, test_class, test_method, forced_java=DEFAULT_JAVA)
                java_used = DEFAULT_JAVA
        finally:
            test_file.write_text(original)
        if self.log_dir is not None:
            path = self.log_dir / row["ID"].replace("/", "__").replace(":", "_") / f"{'control' if rank is None else rank}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{status}\n{log}")
        return status, java_used, log

    def evaluate_case(self, row, candidates, target=None, trust_reference=True, max_rank=None, control=True):
        """
        Verdicts of the candidates of one case, verdicts cached per distinct candidate. With `control` the reference repair runs first:
        when it does not pass in this environment (a dependency that no longer exists, a floating version, another Maven) the case is
        reported as reference_fails and no candidate is judged, so the environment is never blamed on the model.
        Returns {"status": "evaluated" | "reference_fails" | ..., "control": status, "verdicts": [{rank, status, success, java}]}.
        """
        repo = row["ID"].split(":")[0]
        project = self.prepare(repo, row["aCommit"])
        test_rel = Path(row["aPath"])
        test_file = project / test_rel
        if not test_file.exists():
            return {"status": "test_file_missing", "verdicts": []}
        original = test_file.read_text()
        test_class, test_method = split_method(row["name"])
        cache, result = {}, {"status": "evaluated", "verdicts": []}
        if control and target is not None:
            status, java_used, _ = self.execute(project, row, original, target.strip(), test_rel, test_class, test_method)
            cache[target.strip()] = (status, java_used)
            result["control"] = status
            if status != SUCCESS:
                return {"status": "reference_fails", "control": status, "verdicts": []}
        for rank, candidate in enumerate(candidates[:max_rank]):
            candidate = candidate.strip()
            if not candidate:
                status, java_used = "empty", None
            elif candidate in cache:
                status, java_used = cache[candidate]
            elif trust_reference and target is not None and candidate == target.strip():
                status, java_used = SUCCESS, None
            else:
                status, java_used, _ = self.execute(project, row, original, candidate, test_rel, test_class, test_method, rank)
            cache[candidate] = (status, java_used)
            result["verdicts"].append({"rank": rank, "status": status, "success": int(status == SUCCESS), "java": java_used})
        return result

    def evaluate_union(self, row, candidates, target, prior=None):
        """
        Incremental scoring of many models on one case: judges only the candidates not judged before and runs the reference control
        once. `prior` is the record of an earlier run. Returns {"status": "evaluated" | "reference_fails", "control": status,
        "cache": {candidate text: [status, java]}}; the verdict of a model is the cache entry of its candidate.
        """
        prior = prior or {}
        cache = {k: tuple(v) for k, v in prior.get("cache", {}).items()}
        control = prior.get("control")
        wanted = []
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and candidate not in cache and candidate not in wanted:
                wanted.append(candidate)
        if control is not None and control != SUCCESS:
            return {"status": "reference_fails", "control": control, "cache": {k: list(v) for k, v in cache.items()}}
        if control is not None and not wanted:
            return {"status": "evaluated", "control": control, "cache": {k: list(v) for k, v in cache.items()}}
        project = self.prepare(row["ID"].split(":")[0], row["aCommit"])
        test_rel = Path(row["aPath"])
        test_file = project / test_rel
        if not test_file.exists():
            return {"status": "test_file_missing", "cache": {}}
        original = test_file.read_text()
        test_class, test_method = split_method(row["name"])
        if control is None:
            control, java_used, _ = self.execute(project, row, original, target.strip(), test_rel, test_class, test_method)
            cache[target.strip()] = (control, java_used)
            if control != SUCCESS:
                return {"status": "reference_fails", "control": control, "cache": {k: list(v) for k, v in cache.items()}}
            wanted = [c for c in wanted if c not in cache]
        for index, candidate in enumerate(wanted):
            status, java_used, _ = self.execute(project, row, original, candidate, test_rel, test_class, test_method, index)
            cache[candidate] = (status, java_used)
        return {"status": "evaluated", "control": control, "cache": {k: list(v) for k, v in cache.items()}}

    def run_union(self, cases, out_path, workers=2):
        """cases: [{"row", "key", "target", "candidates": [...]}]. The result file is read first and extended (incremental, resumable)."""
        out_path = Path(out_path)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        by_repo = {}
        for case in cases:
            by_repo.setdefault(case["row"]["ID"].split(":")[0], []).append(case)

        def save():
            with self.lock:
                tmp = out_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(results))
                tmp.replace(out_path)

        def work(item):
            repo, group = item
            ran = False
            for case in group:
                prior = results.get(case["key"])
                try:
                    outcome = self.evaluate_union(case["row"], case["candidates"], case["target"], prior)
                except Exception as error:
                    if prior is None:
                        outcome = {"status": "error", "detail": repr(error)[:300], "cache": {}}
                    else:
                        outcome = prior
                if prior is None or outcome != {k: prior.get(k) for k in outcome}:
                    ran = True
                results[case["key"]] = {"id": case["row"]["ID"], **outcome}
                save()
                print(f"{repo}: {case['row']['ID']} control {outcome.get('control')} judged {len(outcome.get('cache', {}))}", flush=True)
            if ran:
                shutil.rmtree(self.workdir / "repos" / repo.replace("/", "__"), ignore_errors=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, by_repo.items()))
        return results

    def run(self, cases, out_path, workers=2, max_rank=None, trust_reference=True, control=True):
        """cases: [{"row": TaRBench row, "key": result key, "preds": [...], "target": str}]. Resumable (the result file is read first)."""
        out_path = Path(out_path)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        by_repo = {}
        for case in cases:
            if case["key"] not in results:
                by_repo.setdefault(case["row"]["ID"].split(":")[0], []).append(case)

        def save():
            with self.lock:
                tmp = out_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(results))
                tmp.replace(out_path)

        def work(item):
            repo, group = item
            for case in group:
                try:
                    outcome = self.evaluate_case(case["row"], case["preds"], case.get("target"), trust_reference, max_rank, control)
                    results[case["key"]] = {"id": case["row"]["ID"], **outcome}
                except Exception as error:                      # one broken project must not cost the others
                    results[case["key"]] = {"id": case["row"]["ID"], "status": "error", "detail": repr(error)[:300], "verdicts": []}
                save()
                print(f"{repo}: {case['row']['ID']} -> {[v['status'] for v in results[case['key']]['verdicts']]}", flush=True)
            shutil.rmtree(self.workdir / "repos" / repo.replace("/", "__"), ignore_errors=True)    # the clone is deleted afterwards

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, by_repo.items()))
        return results


def summarize(results):
    """Plausible rate over evaluated cases: any success among the candidates; cases invalid for >= 75% (with more than one candidate)
    are dropped, cases whose reference fails in this environment are counted separately."""
    kept, dropped = [], 0
    reference_fails = sum(1 for v in results.values() if v.get("status") == "reference_fails")
    for value in results.values():
        if value.get("status") != "evaluated":
            continue
        vs = value.get("verdicts") or []
        if not vs:
            continue
        invalid = sum(1 for v in vs if v["status"] in INVALID_FOR_DROP)
        if len(vs) > 1 and invalid / len(vs) >= 0.75:
            dropped += 1
            continue
        kept.append(vs)
    n = len(kept)
    rate = lambda k: round(100 * sum(any(v["success"] for v in vs[:k]) for vs in kept) / n, 2) if n else None
    return {"cases": n, "reference_fails": reference_fails, "dropped_invalid": dropped, "plausible@1": rate(1), "plausible@5": rate(5), "plausible@k": rate(10 ** 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=20 * 60, help="seconds per Maven run (the paper used 30 min)")
    ap.add_argument("--max-rank", type=int, default=None)
    ap.add_argument("--no-trust-reference", action="store_true")
    ap.add_argument("--no-control", action="store_true", help="do not run the reference repair first (a case the environment cannot judge is then scored as failures)")
    ap.add_argument("--bench", help="TaRBench directory (test.json, test_predictions.json) for --published")
    ap.add_argument("--published", action="store_true", help="run the authors' own predictions")
    ap.add_argument("--ids", help="file with one case ID per line (published mode)")
    ap.add_argument("--split-dir", help="directory with splits/test.json (a subset); ID of a prediction = position in that file")
    ap.add_argument("--predictions", help="test_predictions.json of our runs (ID = position)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cases = []
    if args.published:
        rows = {r["ID"]: r for r in json.load(open(Path(args.bench) / "test.json"))}
        wanted = {l.strip() for l in open(args.ids)} if args.ids else set(rows)
        for p in json.load(open(Path(args.bench) / "test_predictions.json")):
            if p["ID"] in wanted:
                cases.append({"row": rows[p["ID"]], "key": p["ID"], "preds": p["preds"], "target": p["target"]})
    else:
        rows = json.load(open(Path(args.split_dir) / "splits" / "test.json"))
        data = json.load(open(args.predictions))
        records = [{"ID": data["ID"][k], "target": data["target"][k], "preds": data["preds"][k]} for k in data["ID"]] if isinstance(data, dict) else data
        for p in records:
            cases.append({"row": rows[int(p["ID"])], "key": str(p["ID"]), "preds": p["preds"], "target": p["target"]})
    if args.limit:
        cases = cases[:args.limit]
    runner = JavaPlausibility(args.workdir, timeout=args.timeout, log_dir=Path(args.out).with_suffix(".logs"))
    results = runner.run(cases, args.out, args.workers, args.max_rank, not args.no_trust_reference, not args.no_control)
    print(json.dumps(summarize(results)))


if __name__ == "__main__":
    main()
