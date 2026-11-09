"""
Plausibility of predicted test repairs: does the repaired test pass on the repaired version of the code?

For every case of a test split the reference repair is located inside the real test at `repaired_hash`, the
prediction is swapped in (re-indented to the reference's margin) and only that test is run with pytest.
A case counts only when the unmodified test passes in the same environment (the control), otherwise the failure
would say nothing about the prediction. Cases whose reference repair cannot be located (deleted lines, changes in
several places) are reported with a reason and left out.

WARNING: this clones third-party repositories and executes their code together with model-generated code.
Run it inside a Slurm job (or a throw-away container), never on a login node. What the program does about it:
  * every repository is handled by its own process in its own session and process group; a time limit kills the group
    (the test and everything it started), and the repository's clone and virtualenv are deleted afterwards;
  * test runs get a stripped environment: a private HOME and TMPDIR inside the repository's directory, no tokens or
    other variables of the caller, no core files, a cap on file sizes. This does NOT hide your files from code that
    uses absolute paths: it is the same user, so only a container would do that;
  * a test that hits the time limit FAILS: for a candidate the verdict is "fail", and a reference whose own control
    run times out leaves the case out as "reference_fails" (the environment cannot show anything then).

    python -m data_processing.plausibility_test test.json predictions.json --workdir /tmp/plaus --k 5 --workers 4
    python -m data_processing.plausibility_test --results ~/eval_results --models codet5p plbart codegen --workers 4
"""
import argparse
import ast
import json
import os
import random
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from data_processing.atomic_io import atomic_write_text, sigterm_as_keyboard_interrupt
from data_processing.metrics import normalize_indentation

REPO_TIMEOUT = int(os.environ.get("PLAUSIBILITY_REPO_TIMEOUT", 90 * 60))   # without a new result, per repository
CLONE_TIMEOUT = 900
KEPT_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "PLAUSIBILITY_TAG")       # nothing else of the caller reaches the tests
MAX_FILE_SIZE = 4 * 2**30
PLUGINS = ["pytest", "pytest-timeout", "pytest-mock", "pytest-cov"]   # many projects configure them (pytest-asyncio only on demand: its warnings break projects that turn warnings into errors)
# Missing modules a control run may ask for, installed only from this list (never a package named by an import error
# alone: that would let a repository pick what gets installed). {import name: PyPI name}
KNOWN_PACKAGES = {
    "yaml": "pyyaml", "cv2": "opencv-python-headless", "PIL": "pillow", "sklearn": "scikit-learn", "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv", "dateutil": "python-dateutil", "attr": "attrs", "numpy": "numpy", "pandas": "pandas",
    "scipy": "scipy", "requests": "requests", "httpx": "httpx", "aiohttp": "aiohttp", "fastapi": "fastapi",
    "pydantic": "pydantic", "click": "click", "rich": "rich", "tqdm": "tqdm", "jinja2": "jinja2", "flask": "flask",
    "mock": "mock", "responses": "responses", "freezegun": "freezegun", "toml": "toml", "tomli": "tomli",
    "anyio": "anyio", "aiofiles": "aiofiles", "websockets": "websockets", "uvicorn": "uvicorn", "starlette": "starlette",
    "markdown": "markdown", "lxml": "lxml", "cryptography": "cryptography", "psutil": "psutil", "tabulate": "tabulate",
    "openai": "openai", "anthropic": "anthropic", "tiktoken": "tiktoken", "feedparser": "feedparser",
    "pytest_asyncio": "pytest-asyncio", "pytest_mock": "pytest-mock", "hypothesis": "hypothesis", "typer": "typer",
    "sqlalchemy": "sqlalchemy", "networkx": "networkx", "matplotlib": "matplotlib", "pytz": "pytz", "six": "six",
    "tenacity": "tenacity", "loguru": "loguru", "orjson": "orjson", "ujson": "ujson", "yarl": "yarl",
}
# "Unknown config option: X" (a project that configures a plugin that is not installed) -> plugin
CONFIG_PLUGINS = {"timeout": "pytest-timeout", "asyncio_mode": "pytest-asyncio", "asyncio_default_fixture_loop_scope": "pytest-asyncio",
                  "mock_use_standalone_module": "pytest-mock", "env": "pytest-env", "xfail_strict": "pytest",
                  "django_find_project": "pytest-django", "pythonpath": "pytest"}
ROOT = Path(__file__).resolve().parent.parent
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


def junit_counts(report):
    """(tests, failures, errors, skipped) from the <testsuite> element of a junit file, None when there is none."""
    suite = re.search(r"<testsuite\b[^>]*>", report)
    if not suite:
        return None
    counts = {name: re.search(rf'\b{name}="(\d+)"', suite.group(0)) for name in ("tests", "failures", "errors", "skipped")}
    return tuple(int(m.group(1)) if m else 0 for m in counts.values())


def verdict_from(returncode, output, report=None):
    if returncode == 124:
        return "timeout"
    if returncode == 0:
        counts = junit_counts(report) if report else None
        if counts is not None:             # exactly the run test passed: a skipped or deselected test exits with 0 too
            tests, failures, errors, skipped = counts
            return "pass" if tests >= 1 and failures == errors == skipped == 0 else "fail"
        if re.search(r"\d+ passed", output) and "failed" not in output:
            return "pass"
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
    summary = {"cases": len(results), "evaluated": len(evaluated), "statuses": statuses,
               "candidate_timeouts": sum(r.get("timeouts", 0) for r in evaluated)}   # counted as failures
    if evaluated:
        if not any(r.get("first_rank") for r in evaluated):      # a run over the later candidates has no "first"
            summary["plausible@1"] = round(100 * sum(r["verdicts"][0] == "pass" for r in evaluated) / len(evaluated), 2)
        summary["plausible@k"] = round(100 * sum("pass" in r["verdicts"] for r in evaluated) / len(evaluated), 2)
    return summary




# ---- data -----------------------------------------------------------------------------------------------------------

def project_row(row):
    """Only what the evaluation needs: a test split is gigabytes of annotated code and logs."""
    recorded = row.get("test_name")   # written by searchers newer than PR #57, "name" or "Class.name"
    name = recorded.split(".")[-1] if isinstance(recorded, str) and recorded else find_test_name(row.get("annotated_code", ""))
    return {"repository_name": row["repository_name"], "repaired_hash": row["repaired_hash"],
            "relative_path": row["relative_path"], "output": row["output"], "test_name": name}


def iter_rows(path):
    """The cases of a splits/*.json one at a time: the encoder writes one case per line, and a 3 GB file must not be
    loaded as a whole. Any other layout of the json is read the usual way."""
    with open(path, encoding="utf-8") as f:
        if f.readline().strip() == "[":
            first = f.readline().strip()
            try:
                if first in ("", "]"):
                    return
                rows = [json.loads(first.rstrip(","))]
            except json.JSONDecodeError:
                rows = None
            if rows is not None:
                yield rows[0]
                for line in f:
                    line = line.strip()
                    if line and line != "]":
                        yield json.loads(line.rstrip(","))
                return
        f.seek(0)
        yield from json.load(f)


def build_environment(home, pip_cache=None):
    """The environment of every command that runs repository or model code."""
    env = {k: os.environ[k] for k in KEPT_ENVIRONMENT if k in os.environ}
    home = Path(home)
    (home / "tmp").mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), TMPDIR=str(home / "tmp"), XDG_CACHE_HOME=str(home / ".cache"),
               PIP_CACHE_DIR=str(pip_cache or home / ".cache" / "pip"), PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", GIT_TERMINAL_PROMPT="0", CI="1")
    return env


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_SIZE, MAX_FILE_SIZE))


def kill_group(process):
    """SIGKILL the process group a child leads (it was started with start_new_session)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def kill_tagged(tag):
    """SIGKILL every process of this user that carries PLAUSIBILITY_TAG=<tag> in its environment. A test run leads a
    session of its own (so that its timeout can kill it), which a kill of the repository's process group does not
    reach; the tag, inherited by everything the repository's process starts, does."""
    needle = f"PLAUSIBILITY_TAG={tag}".encode()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry.name}/environ", "rb") as f:
                if needle in f.read().split(b"\0"):
                    os.kill(int(entry.name), signal.SIGKILL)
        except OSError:
            pass


def remove_tree(path):
    """rmtree that also removes what a repository made read-only."""
    def fix(function, target, _):
        try:
            os.chmod(target, stat.S_IRWXU)
            os.chmod(os.path.dirname(target), stat.S_IRWXU)
            function(target)
        except OSError:
            pass
    shutil.rmtree(path, onerror=fix)


class PlausibilityTester:
    def __init__(self, workdir, k=5, timeout=120, use_venv=True, install_requirements=True,
                 repo_url="https://github.com/{name}.git", workers=1, repo_timeout=REPO_TIMEOUT, pip_cache=None, first_rank=0):
        self.workdir = Path(workdir)
        self.k = k
        self.first_rank = first_rank              # candidates before this rank were judged by an earlier run
        self.timeout = timeout
        self.use_venv = use_venv
        self.install_requirements = install_requirements
        self.repo_url = repo_url
        self.workers = workers
        self.repo_timeout = repo_timeout
        self.pip_cache = pip_cache
        self.pythons = {}
        self.installed = {}
        self.controls = {}
        self.last_output = ""
        self.last_full_output = ""
        self.env = None
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._active = set()

    # ---- processes -------------------------------------------------------------------------------------------

    def run_cmd(self, cmd, cwd=None, timeout=None):
        """Run a command in its own session; on a timeout the whole process group is killed. Returns (124, "")."""
        if self.env is None:
            self.env = build_environment(self.workdir / "home", self.pip_cache)
        proc = subprocess.Popen(cmd, cwd=cwd, env=self.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True, preexec_fn=_limit_resources)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            proc.communicate()
            return 124, ""
        kill_group(proc)                      # whatever the command left running
        return proc.returncode, stdout.decode("utf-8", "ignore") + "\n" + stderr.decode("utf-8", "ignore")

    # ---- repositories and environments -----------------------------------------------------------------------

    def prepare(self, name, sha):
        """Clone the repository once and check out `sha`; returns the repository directory."""
        repo_dir = self.workdir / "repos" / name.replace("/", "__")
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            code, out = self.run_cmd(["git", "clone", "--quiet", self.repo_url.format(name=name), str(repo_dir)],
                                     timeout=CLONE_TIMEOUT)
            if code != 0:
                raise RuntimeError(f"clone failed: {out.strip()[-300:] or 'timeout'}")
        code, out = self.run_cmd(["git", "checkout", "--quiet", "--force", sha], cwd=repo_dir, timeout=CLONE_TIMEOUT)
        if code != 0:
            raise RuntimeError(f"checkout failed: {out.strip()[-300:] or 'timeout'}")
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
            self.run_cmd([sys.executable, "-m", "venv", str(venv_dir)], timeout=600)
            self.run_cmd([str(python), "-m", "pip", "install", "-q", *PLUGINS], timeout=900)
            if self.install_requirements:
                for requirements in sorted(repo_dir.glob("requirements*.txt")):
                    self.run_cmd([str(python), "-m", "pip", "install", "-q", "-r", str(requirements)], timeout=1800)
            if (repo_dir / "setup.py").exists() or (repo_dir / "pyproject.toml").exists():
                self.run_cmd([str(python), "-m", "pip", "install", "-q", "-e", "."], cwd=repo_dir, timeout=1800)
        self.pythons[name] = str(python)
        return self.pythons[name]

    def run_test(self, python, repo_dir, nodeid):
        # The verdict comes from pytest's junit file, not from its text: a project may put -q in its addopts, which
        # hides the "1 passed" line (that made every test of such a project look like an error).
        junit = Path(tempfile.mkdtemp(prefix="junit_", dir=self.workdir)) / "result.xml"
        cmd = [python, "-m", "pytest", "-x", "-q", "--disable-warnings", "-p", "no:cacheprovider",
               f"--junit-xml={junit}", nodeid]
        code, output = self.run_cmd(cmd, cwd=repo_dir, timeout=self.timeout)
        self.last_output = output.strip()[-400:]
        self.last_full_output = output[-20000:]
        try:
            report = junit.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError:
            report = ""
        shutil.rmtree(junit.parent, ignore_errors=True)
        return verdict_from(code, output, report)

    def missing_packages(self, python, output):
        """Packages (from KNOWN_PACKAGES / CONFIG_PLUGINS only) that the output of a failed run asks for."""
        wanted = []
        for name in re.findall(r"No module named '([\w.]+)'", output):
            top = name.split(".")[0]
            if top in KNOWN_PACKAGES:
                wanted.append(KNOWN_PACKAGES[top])
        for option in re.findall(r"Unknown config option: (\w+)", output):
            if option in CONFIG_PLUGINS:
                wanted.append(CONFIG_PLUGINS[option])
        if re.search(r"async def functions are not natively supported|You need to install a suitable plugin", output):
            wanted.append("pytest-asyncio")
        return [w for w in dict.fromkeys(wanted) if w not in self.installed.setdefault(python, set())]

    def run_control(self, python, repo_dir, nodeid):
        """The control run; an 'error' that names known missing packages is run again after installing them."""
        verdict = self.run_test(python, repo_dir, nodeid)
        for _ in range(3):
            if verdict not in ("error", "fail") or not self.use_venv:
                break
            wanted = self.missing_packages(python, self.last_full_output)
            if not wanted:
                break
            self.run_cmd([python, "-m", "pip", "install", "-q", *wanted], cwd=repo_dir, timeout=900)
            self.installed[python].update(wanted)
            verdict = self.run_test(python, repo_dir, nodeid)
        return verdict

    # ---- cases -----------------------------------------------------------------------------------------------

    def evaluate_case(self, row, candidates, prior=None):
        """prior: [(candidate, verdict)] of an earlier run over the first `first_rank` candidates of this case. The case
        is then known to be run-able (its control passed): the control is not run again, and a candidate whose text
        was judged before keeps that verdict instead of being executed again."""
        result = {"status": None, "nodeid": None, "verdicts": []}
        reference = row["output"]
        recorded = row.get("test_name")
        test_name = recorded.split(".")[-1] if isinstance(recorded, str) and recorded else find_test_name(row.get("annotated_code", ""))
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
        control_key = (row["repository_name"], row["repaired_hash"], nodeid)   # the same case of another model
        if prior is not None and control_key not in self.controls:
            self.controls[control_key] = ("pass", "")
        if control_key not in self.controls:
            verdict = self.run_control(python, repo_dir, nodeid)
            self.controls[control_key] = (verdict, "" if verdict == "pass" else self.last_output)
        control, control_output = self.controls[control_key]
        if control != "pass":                 # a control that times out is a failure too: nothing can be judged
            return {**result, "status": "reference_fails", "detail": f"{control}: {control_output}" if control_output else control}

        known = {normalize_indentation(reference): "pass"}  # equal to the reference: the control already passed
        for earlier, earlier_verdict in prior or []:
            known.setdefault(normalize_indentation(earlier), earlier_verdict)
        timeouts = 0
        result["first_rank"] = self.first_rank
        try:
            for candidate in candidates[self.first_rank:self.k]:
                key = normalize_indentation(candidate)
                if not key.strip():
                    verdict = "empty"
                elif key in known:
                    verdict = known[key]
                else:
                    test_file.write_text(patch_source(original.decode("utf-8"), site, candidate), encoding="utf-8")
                    verdict = self.run_test(python, repo_dir, nodeid)
                    if verdict == "timeout":  # a repair that makes the test hang does not repair it
                        verdict, timeouts = "fail", timeouts + 1
                    known[key] = verdict
                result["verdicts"].append(verdict)
        finally:
            test_file.write_bytes(original)
        if timeouts:
            result["timeouts"] = timeouts
        return {**result, "status": "evaluated"}

    # ---- the whole run ---------------------------------------------------------------------------------------

    def run(self, test_json, predictions_json, out_path):
        """One experiment; see run_many."""
        return self.run_many([{"name": "run", "test_json": test_json, "predictions_json": predictions_json,
                               "out_path": out_path}])["run"]

    def run_many(self, tasks, max_repos=None, retry_statuses=()):
        """
        tasks: [{"name", "test_json", "predictions_json", "out_path"}], e.g. the three models of one experiment. The
        cases of all tasks are grouped by repository, so a repository is cloned and its environment built once, and
        the control run of a test is shared by the models. Each result file is written atomically and a run that is
        started again continues with the cases that have no result. Repositories run in parallel, `workers` at a time.
        """
        self.stop.clear()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.run_root = Path(tempfile.mkdtemp(prefix="run_", dir=self.workdir))
        self.pip_cache = self.pip_cache or str(self.run_root / "pip-cache")
        self.retry_statuses = set(retry_statuses)
        self.results = {t["name"]: {} for t in tasks}
        self.out_paths = {t["name"]: Path(t["out_path"]) for t in tasks}
        self._last_save = 0.0
        groups = {}
        for task in tasks:
            self._load_task(task, groups)
        order = sorted(groups, key=lambda repo: (-len(groups[repo]), repo))
        if max_repos:                                 # a pilot: a fixed random sample of the repositories
            order = sorted(random.Random(1).sample(order, min(max_repos, len(order))), key=lambda repo: (-len(groups[repo]), repo))
        total, finished = len(order), 0
        print(f"{sum(len(groups[repo]) for repo in order)} cases in {total} repositories to run, {self.workers} workers", flush=True)
        self._save(force=True)
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="repo-worker")
        try:
            with sigterm_as_keyboard_interrupt():
                for future in [pool.submit(self._run_repository, repo, groups[repo]) for repo in order]:
                    outcome = future.result()
                    finished += 1
                    print(f"[{finished}/{total}] {outcome}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted: stopping the repository processes", flush=True)
            self.stop.set()
            raise
        finally:
            self.stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            with self._lock:
                for proc in list(self._active):
                    kill_group(proc)
            self._save(force=True)
            kill_tagged_all = [e.name for e in self.run_root.iterdir()] if self.run_root.exists() else []
            for tag in kill_tagged_all:
                kill_tagged(tag)
            remove_tree(self.run_root)
        summaries = {name: summarize(results) for name, results in self.results.items()}
        for name, summary in summaries.items():
            print(name, summary, flush=True)
        return summaries

    def _load_task(self, task, groups):
        name, out_path = task["name"], Path(task["out_path"])
        predictions = load_predictions(task["predictions_json"])
        if out_path.exists():
            self.results[name] = {int(k): v for k, v in json.loads(out_path.read_text(encoding="utf-8"))["results"].items()
                                  if v.get("status") not in self.retry_statuses}     # these are run again
            print(f"{name}: resuming, {len(self.results[name])} cases already done", flush=True)
        todo = {i for i in predictions if i not in self.results[name]}
        earlier = {}
        if task.get("restrict_to"):               # candidates beyond the first_rank-th, only for the cases judged before
            earlier = {int(k): v for k, v in json.loads(Path(task["restrict_to"]).read_text(encoding="utf-8"))["results"].items()
                       if v.get("status") == "evaluated"}
            todo &= set(earlier)
            print(f"{name}: {len(todo)} cases evaluated by the earlier run to extend", flush=True)
        for i, row in enumerate(iter_rows(task["test_json"])):
            if i not in todo:
                continue
            row = project_row(row)
            target, candidates = predictions[i]
            if target.rstrip() != row["output"].rstrip():
                self.results[name][i] = {"status": "id_mismatch", "nodeid": None, "verdicts": []}
            else:
                item = {"model": name, "id": i, "row": row, "candidates": candidates[:self.k]}
                if i in earlier:
                    item["prior"] = list(zip(candidates[:self.first_rank], earlier[i]["verdicts"][:self.first_rank]))
                groups.setdefault(row["repository_name"], []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: (item["row"]["repaired_hash"], item["model"], item["id"]))

    def _save(self, force=False):
        with self._lock:
            if not force and time.time() - self._last_save < 20:
                return
            self._last_save = time.time()
            for name, results in self.results.items():
                atomic_write_text(self.out_paths[name],
                                  json.dumps({"summary": summarize(results), "results": results}, indent=1))

    def _record(self, model, case_id, result):
        with self._lock:
            self.results[model][case_id] = result

    def _run_repository(self, repo, items):
        """One repository in a child process of its own; its directory is deleted at the end, whatever happened."""
        if self.stop.is_set():
            return f"{repo}: skipped"
        repo_root = Path(tempfile.mkdtemp(prefix=repo.replace("/", "__") + "_", dir=self.run_root))
        payload, results_file, log_file = repo_root / "payload.json", repo_root / "results.jsonl", repo_root / "child.log"
        payload.write_text(json.dumps({
            "workdir": str(repo_root / "work"), "k": self.k, "timeout": self.timeout, "use_venv": self.use_venv,
            "install_requirements": self.install_requirements, "repo_url": self.repo_url,
            "pip_cache": self.pip_cache, "first_rank": self.first_rank, "items": items, "results_file": str(results_file)}), encoding="utf-8")
        tag = repo_root.name
        env = dict(os.environ, PYTHONPATH=str(ROOT), PLAUSIBILITY_TAG=tag)
        started, offset, reason = time.time(), 0, None
        last_progress = started            # the repository's limit counts the time without a new result, not the total
        pending = {(item["model"], item["id"]) for item in items}
        with open(log_file, "wb") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "data_processing.plausibility_test", "--child", str(payload)],
                                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        with self._lock:
            self._active.add(proc)
        try:
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    pass
                before = len(pending)
                offset = self._collect(results_file, offset, pending)
                if len(pending) < before:
                    last_progress = time.time()
                if self.stop.is_set():
                    reason = "stopped"
                    break
                if time.time() - last_progress > self.repo_timeout:
                    reason = "repo_timeout"
                    break
            if reason:
                kill_group(proc)
                proc.wait()
            kill_group(proc)
            kill_tagged(tag)
            offset = self._collect(results_file, offset, pending)
            if reason != "stopped" and pending:
                if reason == "repo_timeout":
                    status = {"status": "repo_timeout", "detail": f"no new result for {self.repo_timeout} s in the repository"}
                else:
                    tail = log_file.read_text(encoding="utf-8", errors="ignore")[-300:] if log_file.exists() else ""
                    status = {"status": "repo_crashed", "detail": f"exit status {proc.returncode}: {tail}"}
                for model, case_id in pending:
                    self._record(model, case_id, {"nodeid": None, "verdicts": [], **status})
            self._save()
        finally:
            with self._lock:
                self._active.discard(proc)
            kill_group(proc)
            kill_tagged(tag)
            remove_tree(repo_root)
        done = len(items) - len(pending)
        return f"{repo}: {done}/{len(items)} cases, {int(time.time() - started)} s" + (f" ({reason})" if reason else "")

    def _collect(self, results_file, offset, pending):
        """Take the finished cases from the child's result file (one json line each); returns the new offset."""
        if not results_file.exists():
            return offset
        with open(results_file, "rb") as f:
            f.seek(offset)
            data = f.read()
        end = data.rfind(b"\n") + 1               # a half-written last line waits for the next call
        for line in data[:end].splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._record(entry["model"], entry["id"], entry["result"])
            pending.discard((entry["model"], entry["id"]))
        if end:
            self._save()
        return offset + end


def child_main(payload_path):
    """The process of one repository: evaluates its cases and appends each result to a file as soon as it is known."""
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    tester = PlausibilityTester(payload["workdir"], payload["k"], payload["timeout"], payload["use_venv"],
                                payload["install_requirements"], payload["repo_url"], pip_cache=payload["pip_cache"],
                                first_rank=payload.get("first_rank", 0))
    with open(payload["results_file"], "a", encoding="utf-8") as out:
        for item in payload["items"]:
            try:
                result = tester.evaluate_case(item["row"], item["candidates"], item.get("prior"))
            except Exception as error:            # one broken case must not cost the others of the repository
                result = {"status": "error", "nodeid": None, "verdicts": [], "detail": repr(error)[:300]}
            out.write(json.dumps({"model": item["model"], "id": item["id"], "result": result}) + "\n")
            out.flush()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("test_json", nargs="?", help="splits/test.json of the experiment")
    parser.add_argument("predictions_json", nargs="?", help="test_predictions.json or <model>_llm_test_predictions.json")
    parser.add_argument("--results", help="results directory with <model>/<train size>/ (all models in one run)")
    parser.add_argument("--models", nargs="+", default=["codet5p", "plbart", "codegen"])
    parser.add_argument("--train-size", default="0.8")
    parser.add_argument("--workdir", default="plausibility", help="scratch directory (clones and venvs, deleted again)")
    parser.add_argument("--out", default=None, help="result file (default: <workdir>/plausibility.json)")
    parser.add_argument("--k", type=int, default=5, help="how many top candidates to execute per case")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per test run (a timeout is a failure)")
    parser.add_argument("--repo-timeout", type=int, default=REPO_TIMEOUT, help="seconds a repository may go without a new result")
    parser.add_argument("--workers", type=int, default=1, help="repositories in parallel")
    parser.add_argument("--repo-url", default="https://github.com/{name}.git", help="clone url, {name} = owner/repository")
    parser.add_argument("--no-venv", action="store_true", help="use the current interpreter instead of a venv per repo")
    parser.add_argument("--out-name", default="plausibility.json", help="result file name inside <model>/<train size>/")
    parser.add_argument("--first-rank", type=int, default=0, help="skip the first N candidates (judged by an earlier run)")
    parser.add_argument("--extend", default=None,
                        help="result file of an earlier run (inside <model>/<train size>/): only its evaluated cases, "
                             "candidates from --first-rank on; its verdicts are reused for identical candidates")
    parser.add_argument("--max-repos", type=int, default=None, help="pilot: only a random sample of N repositories")
    parser.add_argument("--retry-statuses", nargs="*", default=[],
                        help="statuses of an earlier run whose cases are run again, e.g. reference_fails setup_failed")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        child_main(args.child)
        return
    tester = PlausibilityTester(args.workdir, args.k, args.timeout, use_venv=not args.no_venv,
                                repo_url=args.repo_url, workers=args.workers, repo_timeout=args.repo_timeout,
                                first_rank=args.first_rank)
    if args.results:
        tasks = []
        for model in args.models:
            experiment = Path(args.results) / model / args.train_size
            tasks.append({"name": model, "test_json": experiment / "splits" / "test.json",
                          "predictions_json": experiment / "test_predictions.json",
                          "out_path": experiment / args.out_name,
                          **({"restrict_to": experiment / args.extend} if args.extend else {})})
        tester.run_many(tasks, max_repos=args.max_repos, retry_statuses=args.retry_statuses)
    else:
        tester.run(args.test_json, args.predictions_json, args.out or Path(args.workdir) / "plausibility.json")


if __name__ == "__main__":
    main()
