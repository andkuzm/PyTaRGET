import ast
import difflib
import os
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
import re
import coverage

from data_types.Broken_to_repaired import Broken_to_repaired
from py_parser import compile_and_run_test_python, TestVerdict, run_cmd, to_pytest_nodeid_part


def _flatten_dependency_group(groups_table, name, _seen=None):
    """Resolve a PEP 735 [dependency-groups] entry into a flat list of
    requirement strings, following {"include-group": "..."} references."""
    if _seen is None:
        _seen = set()
    if name in _seen or name not in groups_table:
        return []
    _seen.add(name)
    reqs = []
    for item in groups_table[name]:
        if isinstance(item, str):
            reqs.append(item)
        elif isinstance(item, dict) and "include-group" in item:
            reqs.extend(_flatten_dependency_group(groups_table, item["include-group"], _seen))
    return reqs


class RepositoryActions:
    def __init__(self, repository_name, repository_path, current_hash=None, previous_hash=None):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.current_hash = current_hash
        self.previous_hash = previous_hash
        self.visited_commits = set()
        self.commit_counter = 0
        self.repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

    def get_repository_name(self):
        return self.repository_name

    def get_repository_path(self):
        return self.repository_path

    def get_current_hash(self):
        return self.current_hash

    def set_current_hash(self, new_hash):
        self.current_hash = new_hash

    def clone_repository_last(self):
        repo_url = f"https://github.com/{self.repository_name}.git"
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        if os.path.exists(dest_dir):
            def handle_remove_readonly(func, path, exc_info):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(dest_dir, onerror=handle_remove_readonly)

        cmd = ["git", "clone", repo_url, dest_dir]
        clone_result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        if clone_result.returncode != 0:
            print(f"Failed to clone {repo_url}: {clone_result.stderr}")
            raise Exception(f"git clone failed for {self.repository_name}")

        # Editable install: a regular `pip install .` copies the package into
        # site-packages as it exists at HEAD *right now* and never touches
        # that copy again. Every later `git checkout <historical commit>` in
        # find_repaired_test_cases()/extract_and_annotate_code() only changes
        # files in dest_dir - the installed copy stays frozen at today's
        # HEAD. So "run the test at the parent/child commit" actually runs
        # today's HEAD code against historical test code, which fails almost
        # every genuine PASS check and silently zeroes out this repo's yield.
        # An editable install re-reads from dest_dir on every import, so it
        # tracks whatever commit is currently checked out. Fall back to a
        # regular install only if editable isn't supported for this repo.
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ, cwd=dest_dir)
        if result.returncode != 0:
            cmd = [sys.executable, "-m", "pip", "install", "."]
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ, cwd=dest_dir)
        if result.returncode != 0:
            print(f"Warning: pip install failed for {self.repository_name}:\n{result.stderr}")
            raise Exception("pip install failed; skipping repository")

        self._install_test_dependencies(dest_dir)

        hash_cmd = ["git", "rev-parse", "HEAD"]
        hash_result = subprocess.run(hash_cmd, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if hash_result.returncode == 0:
            latest_hash = hash_result.stdout.strip()
            self.set_current_hash(latest_hash)
            self.visited_commits.add(latest_hash)
        else:
            print("Error obtaining latest commit hash:", hash_result.stderr)
            raise Exception("Failed to obtain commit hash")

        return dest_dir

    def _install_test_dependencies(self, dest_dir):
        """`pip install .` only pulls a project's *runtime* dependencies from
        its packaging metadata - and plenty of repos (especially ones without
        proper install_requires/[project] metadata, relying on a bare
        requirements.txt instead) declare none at all there, so `pip install .`
        silently installs nothing. On top of that, most test suites also need
        test-only extras (e.g. fastapi's TestClient needs httpx, which only
        ships under the "all"/"standard" extras) or a dev/test requirements
        file. Either way the result is the same: imports fail during test
        collection, parent/child never reach a PASS verdict, and no repairs
        are ever found. This is best-effort: each candidate is installed
        independently and failures are swallowed, so a missing/broken extra
        never blocks the (already-successful) base install or the rest of the
        candidates.
        """
        extra_names = set()
        dependency_groups = {}
        pyproject = Path(dest_dir) / "pyproject.toml"
        if pyproject.exists():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                extra_names.update(data.get("project", {}).get("optional-dependencies", {}).keys())
                extra_names.update(data.get("tool", {}).get("poetry", {}).get("extras", {}).keys())
                dependency_groups = data.get("dependency-groups", {})
            except Exception:
                pass

        # PEP 735 [dependency-groups] (e.g. modern uv-managed projects) is a
        # separate mechanism from [project.optional-dependencies] and is not
        # installable via `pip install .[name]`. Prefer "test"-named groups
        # over "dev" (a "dev" group commonly pulls in large, unrelated
        # tooling - docs builders, browsers for e2e tests, type checkers -
        # that costs a lot of install time for no benefit here). But some
        # projects (e.g. langflow) never split a "tests" group out at all and
        # dump every dev/test tool - including the ones tests actually import,
        # like asgi-lifespan or pytest itself - into a single "dev" group. In
        # that case "dev" is the *only* place the test deps exist, so fall
        # back to it rather than silently getting nothing.
        test_group_reqs = set()
        for name in dependency_groups:
            if "test" in name.lower():
                test_group_reqs.update(_flatten_dependency_group(dependency_groups, name))
        if not test_group_reqs:
            for name in dependency_groups:
                if "dev" in name.lower():
                    test_group_reqs.update(_flatten_dependency_group(dependency_groups, name))
        if test_group_reqs:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *sorted(test_group_reqs)],
                    capture_output=True, text=True, env=os.environ, cwd=dest_dir, timeout=900
                )
            except Exception:
                pass

        keywords = ("test", "dev", "all", "full")
        for extra in sorted(extra_names):
            if any(k in extra.lower() for k in keywords):
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", f".[{extra}]"],
                        capture_output=True, text=True, env=os.environ, cwd=dest_dir, timeout=600
                    )
                except Exception:
                    pass

        requirements_candidates = [
            # Plain requirements.txt first: for repos with no install_requires
            # in their packaging metadata (see docstring), this is the *only*
            # place their actual runtime dependencies are declared at all.
            "requirements.txt",
            "requirements-test.txt", "requirements_test.txt", "test-requirements.txt",
            "requirements-tests.txt", "requirements-dev.txt", "requirements_dev.txt",
            "dev-requirements.txt", "requirements/test.txt", "requirements/tests.txt",
            "requirements/dev.txt", "requirements/base.txt", "requirements/requirements.txt",
            "tests/requirements.txt",
        ]
        for rel in requirements_candidates:
            req_file = Path(dest_dir) / rel
            if req_file.exists():
                self._pip_install_requirements_file(req_file, dest_dir)

    def _pip_install_requirements_file(self, req_file, dest_dir):
        """Install a requirements file, falling back to installing its lines
        one at a time if the batched install fails. pip resolves a
        requirements file as a whole and installs nothing at all if any
        single line is unresolvable (e.g. a stale pin, a platform-specific
        package); going line-by-line on failure salvages every dependency
        that *is* installable instead of losing all of them over one bad
        line."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                capture_output=True, text=True, env=os.environ, cwd=dest_dir, timeout=900
            )
        except Exception:
            return

        if result.returncode == 0:
            return

        try:
            lines = req_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "git+", "http://", "https://")):
                continue
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", line],
                    capture_output=True, text=True, env=os.environ, cwd=dest_dir, timeout=120
                )
            except Exception:
                continue

    def has_tests(self):
        test_patterns = [r"def\s+test_"]
        for root, dirs, files in os.walk(self.repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding="utf-8").splitlines()
                        for line in content:
                            if any(re.search(pattern, line) for pattern in test_patterns):
                                return True
                    except Exception as e:
                        continue
        return False


    def run_test_with_overridden_test_code(self, rel_path, test_method, overridden_test_code):
        """
        Temporarily replaces the test file's content with overridden_test_code,
        runs the specified test, and then restores the original content.
        If the change in test code is deemed unimportant (i.e. nearly identical to the current test code),
        the method returns a dummy success result immediately.
        """

        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        original_content = test_file_path.read_text(encoding="utf-8")

        line_range = self._get_method_line_range(rel_path, test_method)
        if line_range is None:
            return TestVerdict(TestVerdict.UNKNOWN, None, "Target test method not found")

        start, end = line_range

        try:
            lines = original_content.splitlines(keepends=True)
            new_lines = lines[:start] + [overridden_test_code + "\n"] + lines[end:]
            new_content = "".join(new_lines)
            test_file_path.write_text(new_content, encoding="utf-8")

            self.repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
            # Invalidate cached bytecode before running. The overridden test is
            # frequently the same byte-size as the child test it replaces and is
            # written within the same wall-clock second, so the .pyc header
            # (source mtime at second resolution + size) still matches the
            # stale, child-version bytecode. Without this, Python/pytest reuse
            # that stale bytecode and the override spuriously *passes*, causing
            # every genuine repaired case to be silently discarded.
            self._invalidate_bytecode_cache()
            result = compile_and_run_test_python(
                self.repo_dir, rel_path, test_method, self.repo_dir.parent
            )
        finally:
            test_file_path.write_text(original_content, encoding="utf-8")
            self._invalidate_bytecode_cache()

        return result

    def _invalidate_bytecode_cache(self):
        """Remove cached bytecode under the repo so freshly written test source
        is always recompiled (see run_test_with_overridden_test_code)."""
        for cache_dir in self.repo_dir.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        for pyc in self.repo_dir.rglob("*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass

    def find_repaired_test_cases(self):
        """
        BFS over the full commit DAG, following all parents of every merge commit.
        This surfaces repair commits that live on non-first-parent (feature-branch)
        history, which the previous linear HEAD^ walk missed entirely.

        For each (child, parent) pair:
          1. Collect test method source in the parent commit.
          2. Collect test method source in the child commit.
          3. For methods present in both that changed (difflib hunk-level):
             a. Parent must PASS.
             b. Child must PASS.
             c. Override (child source + parent test code) must FAIL.
             d. Record as a repaired test case.

        commit_counter is incremented globally across all branches; once
        MAX_COMMITS is reached the traversal stops.
        """
        MAX_COMMITS = 1000
        repaired_cases = set()

        # Git pathspecs without a leading "*/" only match at the repo root,
        # and "*/tests/*.py" needs a directory *before* "tests/" — so without
        # the bare "test_*.py" / "tests/*.py" / "test/*.py" variants below,
        # root-level test files (very common in small repos) never mark a
        # commit as relevant and their repairs are silently skipped.
        result = subprocess.run(
            ["git", "log", "--format=%H", "--diff-filter=M", "--",
             "test_*.py", "*/test_*.py",
             "*_test.py",
             "tests/*.py", "*/tests/*.py",
             "test/*.py", "*/test/*.py"],
            cwd=self.repo_dir, capture_output=True, text=True
        )
        relevant_commits = set(result.stdout.splitlines())

        queue = deque([self.current_hash])
        dest_dir = str(self.repo_dir)

        while queue:
            if self.commit_counter >= MAX_COMMITS:
                break

            child_commit = queue.popleft()

            proc = subprocess.run(
                ["git", "cat-file", "-p", child_commit],
                cwd=dest_dir, capture_output=True, text=True
            )
            if proc.returncode != 0:
                continue

            parents = [
                line.split()[1]
                for line in proc.stdout.splitlines()
                if line.startswith("parent ")
            ]
            if not parents:
                continue  # root commit

            for parent_commit in parents:
                if parent_commit in self.visited_commits:
                    continue
                if self.commit_counter >= MAX_COMMITS:
                    break

                self.visited_commits.add(parent_commit)
                self.commit_counter += 1
                queue.append(parent_commit)

                if child_commit not in relevant_commits and parent_commit not in relevant_commits:
                    continue

                # --- collect parent tests ---
                if not self.git_checkout_with_retry(dest_dir, parent_commit):
                    continue
                self.set_full_permissions()

                parent_methods = {}
                for file in self.list_test_files():
                    rel_path = str(file.relative_to(self.repo_dir))
                    for (_, test_method) in self.find_test_methods(rel_path):
                        code = self.extract_method_code(rel_path, test_method)
                        if code:
                            parent_methods[(rel_path, test_method)] = code

                # --- collect child tests ---
                if not self.git_checkout_with_retry(dest_dir, child_commit):
                    continue
                self.set_full_permissions()

                child_methods = {}
                for file in self.list_test_files():
                    rel_path = str(file.relative_to(self.repo_dir))
                    for (_, test_method) in self.find_test_methods(rel_path):
                        code = self.extract_method_code(rel_path, test_method)
                        if code:
                            child_methods[(rel_path, test_method)] = code

                # --- detect changed tests ---
                changed_tests = {
                    k for k in parent_methods
                    if k in child_methods
                       and self.is_test_method_changed(parent_methods[k], child_methods[k])
                }

                for key in changed_tests:
                    rel_path, test_method = key

                    # ---------- parent must PASS ----------
                    if not self.git_checkout_with_retry(dest_dir, parent_commit):
                        continue
                    self.set_full_permissions()
                    parent = compile_and_run_test_python(self.repo_dir, rel_path, test_method, self.repo_dir.parent)

                    if parent.status != TestVerdict.SUCCESS:
                        continue

                    # ---------- child must PASS ----------
                    if not self.git_checkout_with_retry(dest_dir, child_commit):
                        continue
                    self.set_full_permissions()
                    child = compile_and_run_test_python(self.repo_dir, rel_path, test_method, self.repo_dir.parent)

                    if child.status != TestVerdict.SUCCESS:
                        continue

                    # ---------- override must FAIL ----------
                    overridden = self.run_test_with_overridden_test_code(rel_path, test_method, parent_methods[key])

                    if overridden.status in (TestVerdict.FAILURE, TestVerdict.SYNTAX_ERR):
                        repaired_cases.add(
                            Broken_to_repaired(parent_commit, child_commit, test_method, rel_path, overridden.log)
                        )

        return repaired_cases

    def extract_and_annotate_code(self, broken_to_repaired_instance):
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        if not self.checkout_commit(broken_to_repaired_instance.broken, dest_dir):
            return "Error"
        self.current_hash = broken_to_repaired_instance.broken
        self.set_full_permissions()
        broken_test = self.extract_method_code(broken_to_repaired_instance.rel_path,
                                               broken_to_repaired_instance.test_name)

        if not self.checkout_commit(broken_to_repaired_instance.repaired, dest_dir):
            return "Error"
        self.current_hash = broken_to_repaired_instance.repaired
        self.set_full_permissions()
        repaired_test = self.extract_method_code(broken_to_repaired_instance.rel_path,
                                                 broken_to_repaired_instance.test_name)

        # Extract and annotate the source coverage
        source_code = self.extract_covered_source_coverage(
            broken_to_repaired_instance.rel_path,
            broken_to_repaired_instance.test_name,
            broken_to_repaired_instance.broken,
            broken_to_repaired_instance.repaired
        )

        if not source_code:
            return "Error"

        annotated_code = self.annotate_code(broken_test, repaired_test, source_code, broken_to_repaired_instance.rel_path)
        return annotated_code

    def annotate_code(self, broken_test, repaired_test, source_code, rel_path):
        broken_lines = broken_test.splitlines()
        repaired_lines = repaired_test.splitlines()

        diff_lines = list(
            difflib.unified_diff(
                broken_lines,
                repaired_lines,
                fromfile="Broken Test",
                tofile="Repaired Test",
                lineterm="",
                n=len(broken_lines) + len(repaired_lines),
            )
        )

        diff_lines = self.filter_diff_lines(diff_lines)
        if not diff_lines:
            return ""

        unchanged_before = []
        breakage_lines = []
        unchanged_after = []
        repaired_lines_only = []

        last_change_idx = -1
        for i, line in enumerate(diff_lines):
            if line.startswith("-") or line.startswith("+"):
                last_change_idx = i

        in_change_block = False

        for i, line in enumerate(diff_lines):
            if line.startswith("@@"):
                continue
            elif line.startswith("-"):
                breakage_lines.append(line[1:])
                in_change_block = True
            elif line.startswith("+"):
                repaired_lines_only.append(line[1:])
                in_change_block = True
            else:
                if not in_change_block:
                    unchanged_before.append(line)
                elif i > last_change_idx:
                    unchanged_after.append(line)
                else:
                    # inter-hunk context: present in both broken and repaired versions
                    breakage_lines.append(line)
                    repaired_lines_only.append(line)

        imports = self.extract_test_imports(rel_path)

        annotated_string = (
                "[<TESTCONTEXT>]\n"
                + imports + "\n\n"
                + "\n".join(unchanged_before) + "\n"
                + "[<BREAKAGE>]\n"
                + "\n".join(breakage_lines) + "\n"
                + "[</BREAKAGE>]\n"
                + "\n".join(unchanged_after) + "\n"
                + "[</TESTCONTEXT>]\n\n"
                + "[<REPAIREDTEST>]\n"
                + "\n".join(repaired_lines_only) + "\n"
                + "[</REPAIREDTEST>]\n\n"
                + "[<REPAIRCONTEXT>]\n"
                + source_code + "\n"
                + "[</REPAIRCONTEXT>]"
        )

        return annotated_string


    def checkout_commit(self, commit_hash, dest_dir):
        # set_full_permissions() chmods every tracked file to 0o777, which
        # flips the executable bit on ordinarily non-executable files and
        # leaves the working tree with mode-only "modifications". A plain
        # `git checkout` then refuses to switch commits ("local changes
        # would be overwritten"), so every checkout after the first one in a
        # sequence (e.g. the broken->repaired pair in
        # extract_and_annotate_code) fails. Reset first to discard that
        # self-inflicted noise, exactly like git_checkout_with_retry does.
        subprocess.run(["git", "reset", "--hard"], cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        cmd_checkout = ["git", "checkout", commit_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            print(f"Error checking out commit {commit_hash}: {proc_checkout.stderr}")
            return False
        return True

    def move_to_earlier_commit(self):
        """
        Moves the repository checkout to the parent commit of the current HEAD.
        Before moving, saves the current commit as previous_hash.
        Raises an exception if a cycle is detected.
        """

        MAX_COMMITS = 500
        if self.commit_counter >= MAX_COMMITS:
            return "Error"

        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        self.previous_hash = self.current_hash

        proc_parent = subprocess.run(
            ["git", "rev-parse", "HEAD^"],
            cwd=dest_dir,
            capture_output=True,
            text=True
        )
        if proc_parent.returncode != 0:
            return "Error"

        parent_hash = proc_parent.stdout.strip()

        if parent_hash in self.visited_commits:
            return "Error"
        self.visited_commits.add(parent_hash)

        if not self.git_checkout_with_retry(dest_dir, parent_hash):
            print(f"Error checking out parent commit {parent_hash}.")
            return "Error"

        self.current_hash = parent_hash
        self.commit_counter += 1
        self.set_full_permissions()
        return parent_hash

    def move_to_later_commit(self):
        """
        Moves the repository checkout back to the child commit.
        """

        if self.previous_hash is None:
            return "Error"

        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        if not self.git_checkout_with_retry(dest_dir, self.previous_hash):
            print("Error checking out child commit.")
            return "Error"

        self.current_hash = self.previous_hash
        self.previous_hash = None
        self.set_full_permissions()
        return self.current_hash

    def git_checkout_with_retry(self, dest_dir, target_hash, retries=5):
        for attempt in range(1, retries + 1):
            subprocess.run(["git", "reset", "--hard"], cwd=dest_dir)
            subprocess.run(["git", "clean", "-fdx"], cwd=dest_dir)

            proc = subprocess.run(["git", "checkout", target_hash], cwd=dest_dir, capture_output=True, text=True)

            if proc.returncode == 0:
                return True

            print(f"Checkout to {target_hash} failed (attempt {attempt}/{retries}): {proc.stderr}")
            time.sleep(2 ** attempt)

        return False

    def find_test_methods(self, test_rel_path):
        full_test_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / test_rel_path
        try:
            source = full_test_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {full_test_path}: {e}")
            return []
        try:
            tree = ast.parse(source)
        except Exception as e:
            print(f"Error parsing {full_test_path}: {e}")
            return []
        test_methods = []
        # Only module-level functions here; ast.walk() would also descend into
        # class bodies and re-add class methods a second time under their bare
        # (unqualified) name, producing bogus entries that can never be
        # collected by pytest. Class methods are handled separately below.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                if any("parametrize" in (ast.get_source_segment(source, d) or "") for d in node.decorator_list):
                    continue
                test_methods.append([test_rel_path, node.name])
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                inherits_testcase = any(
                    (isinstance(base, ast.Name) and base.id == "TestCase") or
                    (isinstance(base, ast.Attribute) and base.attr == "TestCase")
                    for base in node.bases
                )
                if inherits_testcase or node.name.startswith("Test"):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                            test_methods.append([test_rel_path, f"{node.name}.{item.name}"])
        return test_methods

    def extract_test_imports(self, rel_path):
        test_file_path = self.repo_dir / rel_path
        try:
            source = test_file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            return ""

        imports = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    imports.append(segment)

        return "\n".join(imports)

    def extract_executed_methods(self, source_code, rel_path): #fallback only
        """
        Extracts full methods that were executed during the test.
        Parses the executed source, finds the full method definitions from the full file,
        and returns them as a dictionary with keys in the format 'ClassName.method_name'.
        """

        executed_methods = {}
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        if not test_file_path.exists():
            return executed_methods

        try:
            full_source = test_file_path.read_text(encoding="utf-8")
            tree = ast.parse(full_source)
        except Exception:
            return executed_methods

        executed_lines = set(i for i, _ in enumerate(full_source.splitlines(), start=1)
                             if any(x.strip() for x in source_code.splitlines()))

        class Extractor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None

            def visit_ClassDef(self, node):
                prev = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev

            def visit_FunctionDef(self, node):
                start = node.lineno
                end = getattr(node, "end_lineno", start)

                if any(l in executed_lines for l in range(start, end + 1)):
                    name = f"{self.current_class}.{node.name}" if self.current_class else f"Global.{node.name}"
                    executed_methods[name] = ast.get_source_segment(full_source, node)

        Extractor().visit(tree)
        return executed_methods

    def extract_executed_methods_from_file(self, filename, executed_lines):
        executed_methods = {}

        try:
            full_source = Path(filename).read_text(encoding="utf-8")
            tree = ast.parse(full_source)
        except Exception:
            return executed_methods

        class Extractor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None

            def visit_ClassDef(self, node):
                prev = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev

            def visit_FunctionDef(self, node):
                start = node.lineno
                end = getattr(node, "end_lineno", start)

                if any(l in executed_lines for l in range(start, end + 1)):
                    name = f"{self.current_class}.{node.name}" if self.current_class else f"Global.{node.name}"
                    executed_methods[name] = ast.get_source_segment(full_source, node)

        Extractor().visit(tree)
        return executed_methods

    def extract_covered_source_coverage(self, rel_path, test_method, broken_hash, repaired_hash):
        dest_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        test_file_abs = str((self.repo_dir / rel_path).resolve())

        # --- BROKEN ---
        # reset first: set_full_permissions() leaves mode-only "modifications"
        # that make a plain `git checkout` refuse to switch commits (see
        # checkout_commit() above).
        subprocess.run(["git", "reset", "--hard"], cwd=dest_dir, capture_output=True, text=True)
        subprocess.run(["git", "checkout", broken_hash], cwd=dest_dir)
        broken_data = self.get_covered_source(rel_path, test_method, broken_hash)

        executed_methods_broken = {}
        if broken_data is not None:
            for filename in broken_data.measured_files():
                if not filename.endswith(".py") or os.path.abspath(filename) == test_file_abs:
                    continue
                lines = broken_data.lines(filename)
                if lines:
                    executed_methods_broken.update(
                        self.extract_executed_methods_from_file(filename, lines)
                    )

        # --- REPAIRED ---
        subprocess.run(["git", "reset", "--hard"], cwd=dest_dir, capture_output=True, text=True)
        subprocess.run(["git", "checkout", repaired_hash], cwd=dest_dir)
        repaired_data = self.get_covered_source(rel_path, test_method, repaired_hash)

        executed_methods_repaired = {}
        if repaired_data is not None:
            for filename in repaired_data.measured_files():
                if not filename.endswith(".py") or os.path.abspath(filename) == test_file_abs:
                    continue
                lines = repaired_data.lines(filename)
                if lines:
                    executed_methods_repaired.update(
                        self.extract_executed_methods_from_file(filename, lines)
                    )

        if not executed_methods_broken and not executed_methods_repaired:
            return ""

        result = []
        all_method_names = set(executed_methods_broken) | set(executed_methods_repaired)

        for method_name in sorted(all_method_names):
            broken_code = executed_methods_broken.get(method_name, "")
            repaired_code = executed_methods_repaired.get(method_name, "")

            diff = list(difflib.unified_diff(
                broken_code.splitlines(),
                repaired_code.splitlines(),
                lineterm=""
            ))
            diff = self.filter_diff_lines(diff)

            if diff:
                hunk = self.format_inline_diff(method_name, diff)
            else:
                hunk = f"[<HUNK>] {method_name}\n{repaired_code}\n[</HUNK>]"

            class_name = method_name.split(".")[0]
            result.append(f"class {class_name}:")
            result.append(hunk)

        return "\n\n".join(result)

    def format_inline_diff(self, method_name, diff):
        """
        Returns a string that wraps changes for a given method in [<HUNK>] tags.
        Consecutive added lines are grouped inside a single [<ADD>] block,
        and consecutive removed lines inside a single [<DEL>] block.
        Context lines are output as-is.
        """
        hunk_lines = [f"[<HUNK>]\n"]
        current_block = []  # Buffer for consecutive changed lines
        current_type = None  # '+' or '-' for the current block

        def flush_block():
            nonlocal current_block, current_type
            if current_block and current_type:
                block_text = "".join(current_block)
                if current_type == '+':
                    hunk_lines.append(f"[<ADD>]\n{block_text}[</ADD>]\n")
                elif current_type == '-':
                    hunk_lines.append(f"[<DEL>]\n{block_text}[</DEL>]\n")
                current_block = []
                current_type = None

        for line in diff:
            if line.startswith('@@'):
                # Skip hunk header lines.
                flush_block()
                continue
            if line.startswith('+') or line.startswith('-'):
                line_type = line[0]
                content = line[1:] + "\n"  # Append newline to preserve formatting.
                if current_type == line_type:
                    current_block.append(content)
                else:
                    flush_block()
                    current_type = line_type
                    current_block.append(content)
            else:
                # Context line: flush any pending block and output context line.
                flush_block()
                hunk_lines.append(line)
        flush_block()  # Flush any remaining block.
        hunk_lines.append("[</HUNK>]\n")
        return "\n".join(hunk_lines) if len(hunk_lines) > 2 else ""

    def get_covered_source(self, rel_path, test_method, commit_hash):
        self.repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        # Reset first: set_full_permissions() leaves mode-only "modifications"
        # that make a plain `git checkout` refuse to switch commits (see
        # checkout_commit() above).
        subprocess.run(["git", "reset", "--hard"], cwd=str(self.repo_dir), capture_output=True, text=True)

        # Checkout the specified commit.
        proc = subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=str(self.repo_dir),
            capture_output=True,
            text=True,
            env=os.environ
        )
        if proc.returncode != 0:
            print(f"Git checkout failed: {proc.stderr}")
            return None

        # Checking out broken/repaired commits in place can leave stale .pyc
        # files whose header (source mtime at second resolution + size) still
        # matches the just-checked-out source, so coverage would measure the
        # previous commit's bytecode and yield wrong/empty covered source.
        self._invalidate_bytecode_cache()

        # Ensure a .coveragerc file exists.
        coveragerc_path = self.repo_dir / ".coveragerc"
        if not coveragerc_path.exists():
            with open(coveragerc_path, "w", encoding="utf-8") as f:
                f.write("""[run]
                branch = True
                parallel = False
                """)

        # Set up the environment for the coverage subprocess.
        env = os.environ.copy()
        env["COVERAGE_PROCESS_START"] = str(coveragerc_path)
        env["PYTHONPATH"] = str(self.repo_dir)

        # Build the test node id (using the absolute path of the test file).
        test_file_path = self.repo_dir / rel_path
        nodeid = f"{test_file_path.as_posix()}::{to_pytest_nodeid_part(test_method)}"

        # Remove stale coverage files from previous runs before starting.
        for stale in self.repo_dir.glob(".coverage*"):
            try:
                stale.unlink()
            except Exception:
                pass

        # Run the test via coverage in parallel mode.
        cmd = [
            sys.executable, "-m", "coverage", "run", "--parallel-mode", "-m", "pytest",
            "--maxfail=1", "--disable-warnings", "--quiet", nodeid
        ]
        returncode, log = run_cmd(cmd, timeout=15 * 60, cwd=str(self.repo_dir), env=env)

        combine_cmd = [sys.executable, "-m", "coverage", "combine"]
        combine_return, combine_log = run_cmd(combine_cmd, timeout=15 * 60, cwd=str(self.repo_dir), env=env)

        if combine_return != 0:
            return None

        # Load the combined coverage data.
        cov_data_file = self.repo_dir / ".coverage"
        cov = coverage.Coverage(data_file=str(cov_data_file))
        cov.load()
        data = cov.get_data()

        return data

    def _get_method_line_range(self, rel_path, test_method):
        """Returns (start, end) as 0-indexed line indices for slicing, or None if not found."""
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        if not test_file_path.exists():
            return None
        try:
            source_code = test_file_path.read_text(encoding="utf-8")
            tree = ast.parse(source_code)
        except Exception:
            return None

        target = test_method.split(".")
        want_class = target[0] if len(target) == 2 else None
        want_method = target[-1]

        class RangeFinder(ast.NodeVisitor):
            def __init__(self):
                self.result = None

            def _range(self, node):
                decorator_lines = [d.lineno for d in node.decorator_list] if node.decorator_list else []
                start = min(decorator_lines + [node.lineno]) - 1
                end = node.end_lineno
                self.result = (start, end)

            def visit_FunctionDef(self, node):
                if want_class is None and node.name == want_method:
                    self._range(node)

            def visit_AsyncFunctionDef(self, node):
                if want_class is None and node.name == want_method:
                    self._range(node)

            def visit_ClassDef(self, node):
                if want_class and node.name == want_class:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == want_method:
                            self._range(item)

        finder = RangeFinder()
        finder.visit(tree)
        return finder.result

    def extract_method_code(self, rel_path, test_method):
        """
        Supports:
          test_func
          ClassName.test_func
        """

        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        if not test_file_path.exists():
            return ""

        try:
            source_code = test_file_path.read_text(encoding="utf-8")
            lines = source_code.splitlines()
            tree = ast.parse(source_code)
        except Exception:
            return ""

        target = test_method.split(".")
        want_class = target[0] if len(target) == 2 else None
        want_method = target[-1]

        class FunctionExtractor(ast.NodeVisitor):
            def __init__(self):
                self.found = ""

            def _extract_with_decorators(self, node):
                # skip parametrized tests (produce unstable repairs)
                for d in node.decorator_list:
                    dec_src = ast.get_source_segment(source_code, d)
                    if dec_src and "parametrize" in dec_src:
                        return ""

                # include decorators if present
                decorator_lines = [d.lineno for d in node.decorator_list] if node.decorator_list else []
                start = min(decorator_lines + [node.lineno]) - 1
                end = node.end_lineno
                return "\n".join(lines[start:end])

            def visit_FunctionDef(self, node):
                if want_class is None and node.name == want_method:
                    self.found = self._extract_with_decorators(node)

            def visit_AsyncFunctionDef(self, node):
                if want_class is None and node.name == want_method:
                    self.found = self._extract_with_decorators(node)

            def visit_ClassDef(self, node):
                if want_class and node.name == want_class:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == want_method:
                            self.found = self._extract_with_decorators(item)

        fx = FunctionExtractor()
        fx.visit(tree)

        return fx.found.strip() if fx.found else ""

    def filter_diff_lines(self, diff_lines, strip_markers=False):
        """
        Filters out diff header lines from a list of diff lines.

        If strip_markers is True, also removes the leading '+' or '-' from changed lines.
        Otherwise, leaves changed lines intact.
        """
        filtered = []
        for line in diff_lines:
            if line.startswith('@@') or line.startswith('---') or line.startswith('+++'):
                continue  # Skip header lines
            if strip_markers and (line.startswith('+') or line.startswith('-')):
                filtered.append(line[1:])
            else:
                filtered.append(line)
        return filtered

    def is_test_method_changed(self, parent_code, child_code):
        parent = [l.rstrip() for l in parent_code.splitlines()]
        child = [l.rstrip() for l in child_code.splitlines()]

        diff = self.filter_diff_lines(list(difflib.unified_diff(parent, child, lineterm="")))

        def meaningful(line):
            c = line[1:].strip()
            return c and not c.startswith("#")

        return any(line.startswith(('+', '-')) and meaningful(line) for line in diff)

    def list_test_files(self):
        self.repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        files_paths = []
        test_patterns = [r"def\s+test_"]
        for root, dirs, files in os.walk(self.repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding="utf-8").splitlines()
                        if any(re.search(pattern, line) for pattern in test_patterns for line in content):
                            files_paths.append(file_path)
                    except Exception as e:
                        continue
        return files_paths

    def set_full_permissions(self):
        os.chmod(self.repository_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for root, dirs, files in os.walk(self.repository_path):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
