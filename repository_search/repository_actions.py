import ast
import difflib
import os
import shutil
import stat
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
import re
import coverage

from data_types.Broken_to_repaired import Broken_to_repaired
from py_parser import compile_and_run_test_python, TestVerdict, run_cmd

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
            print(f"Repository already exists at {dest_dir}. Removing it...")
            def handle_remove_readonly(func, path, exc_info):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(dest_dir, onerror=handle_remove_readonly)

        print(f"Cloning repository from {repo_url} to {dest_dir}...")
        cmd = ["git", "clone", repo_url, dest_dir]
        subprocess.run(cmd, capture_output=True, text=True, env=os.environ)

        cmd = [sys.executable, "-m", "pip", "install", "."]
        print(subprocess.run(cmd, capture_output=True, text=True, env=os.environ, cwd=dest_dir))

        hash_cmd = ["git", "rev-parse", "HEAD"]
        hash_result = subprocess.run(hash_cmd, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if hash_result.returncode == 0:
            latest_hash = hash_result.stdout.strip()
            self.set_current_hash(latest_hash)
            self.visited_commits.add(latest_hash)
            print(f"Current commit hash set to: {self.current_hash}")
        else:
            print("Error obtaining latest commit hash:", hash_result.stderr)
            raise Exception("Failed to obtain commit hash")

        return dest_dir

    def has_tests(self):
        print("Checking if tests exist")
        test_patterns = [r"def\s+test_"]
        for root, dirs, files in os.walk(self.repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding="utf-8").splitlines()
                        for line in content:
                            if any(re.search(pattern, line) for pattern in test_patterns):
                                print(f"Test file found: {file_path}")
                                return True
                    except Exception as e:
                        continue
        print("Tests not found")
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

        # Extract imports once
        imports = self.extract_test_imports(rel_path)

        # Replace only the method body
        current_method_code = self.extract_method_code(rel_path, test_method)
        if not current_method_code:
            return TestVerdict(TestVerdict.UNKNOWN, None, "Target test method not found")

        patched_test_code = imports + "\n\n" + overridden_test_code

        try:
            new_content = original_content.replace(current_method_code, patched_test_code)
            test_file_path.write_text(new_content, encoding="utf-8")

            self.repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
            result = compile_and_run_test_python(
                self.repo_dir, rel_path, test_method, self.repo_dir.parent
            )
        finally:
            test_file_path.write_text(original_content, encoding="utf-8")

        return result

    def find_repaired_test_cases(self):
        """
        Revised approach iterating over commit pairs:
          The method iterates over commit pairs (child and its immediate parent) until it reaches
          the beginning of the commit history or a stop condition from move_to_earlier_commit.
          For each pair:
             1. In the parent commit, extract test methods' source code.
             2. Switch back to the child commit and extract test methods' source code.
             3. For each test method present in both commits, if the parent's and child's code differ
                (determined at hunk-level using difflib), then:
                 a. Verify the test passes in the parent commit.
                 b. Verify the test passes in the child commit.
                 c. Run the parent's test code on the child's source using run_test_with_overridden_test_code.
                 d. If the override test fails, record it as a repaired test case.
          After processing the pair, update the current commit to the parent commit for the next iteration.
        Returns:
          A set of Broken_to_repaired objects representing detected repaired test cases.
        """
        repaired_cases = set()

        while True:
            child_commit = self.current_hash

            parent_commit = self.move_to_earlier_commit()
            if parent_commit == "Error":
                break

            print(f"Processing commit pair: Parent: {parent_commit} | Child: {child_commit}")

            # --- collect parent tests ---
            parent_methods = {}
            for file in self.list_test_files():
                rel_path = str(file.relative_to(self.repo_dir))
                for (_, test_method) in self.find_test_methods(rel_path):
                    code = self.extract_method_code(rel_path, test_method)
                    if code:
                        parent_methods[(rel_path, test_method)] = code

            # --- back to child ---
            if self.move_to_later_commit() == "Error":
                break

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
                if self.move_to_earlier_commit() == "Error":
                    continue
                parent = compile_and_run_test_python(self.repo_dir, rel_path, test_method, self.repo_dir.parent)

                if parent.status != TestVerdict.SUCCESS:
                    print(f"Skipping {key}: parent does not PASS ({parent.status})")
                    self.move_to_later_commit()
                    continue

                # ---------- child must PASS ----------
                if self.move_to_later_commit() == "Error":
                    continue
                child = compile_and_run_test_python(self.repo_dir, rel_path, test_method, self.repo_dir.parent)

                if child.status != TestVerdict.SUCCESS:
                    print(f"Skipping {key}: child does not PASS ({child.status})")
                    continue

                # ---------- override must FAIL ----------
                overridden = self.run_test_with_overridden_test_code(rel_path, test_method, parent_methods[key])

                if overridden.status == TestVerdict.FAILURE:
                    print(f"Repaired test detected: {key}")
                    repaired_cases.add(
                        Broken_to_repaired(parent_commit, self.current_hash, test_method, rel_path, overridden.log)
                    )

            # --- move history pointer backward ---
            dest_dir = str(self.repo_dir)
            if not self.git_checkout_with_retry(dest_dir, parent_commit):
                break

            self.current_hash = parent_commit

        return repaired_cases

    def extract_and_annotate_code(self, broken_to_repaired_instance):
        print("Attempting to extract and annotate code")
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        # Checkout to the parent's commit (broken test).
        if not self.checkout_commit(broken_to_repaired_instance.broken, dest_dir):
            return "Error"
        self.current_hash = broken_to_repaired_instance.broken
        self.set_full_permissions()
        print(f"Repository at broken commit: {self.current_hash}")
        broken_test = self.extract_method_code(broken_to_repaired_instance.rel_path,
                                               broken_to_repaired_instance.test_name)

        # Checkout to the child's commit (repaired test).
        if not self.checkout_commit(broken_to_repaired_instance.repaired, dest_dir):
            return "Error"
        self.current_hash = broken_to_repaired_instance.repaired
        self.set_full_permissions()
        print(f"Repository at repaired commit: {self.current_hash}")
        repaired_test = self.extract_method_code(broken_to_repaired_instance.rel_path,
                                                 broken_to_repaired_instance.test_name)

        # Extract and annotate the source coverage
        source_code = self.extract_covered_source_coverage(
            broken_to_repaired_instance.rel_path,
            broken_to_repaired_instance.test_name,
            broken_to_repaired_instance.broken,
            broken_to_repaired_instance.repaired
        )

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

        in_change_block = False

        for line in diff_lines:
            if line.startswith("@@"):
                continue
            elif line.startswith("-"):
                breakage_lines.append(line[1:])
                in_change_block = True
            elif line.startswith("+"):
                repaired_lines_only.append(line[1:])
                in_change_block = True
            else:
                if in_change_block:
                    unchanged_after.append(line)
                else:
                    unchanged_before.append(line)

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

        # if self.commit_counter >= 300:
        #     print("Commit counter limit reached.")
        #     return "Error"

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
        print(f"Parent commit hash: {parent_hash}")

        if not self.git_checkout_with_retry(dest_dir, parent_hash):
            print(f"Error checking out parent commit {parent_hash}. Ending iteration.")
            return "Error"

        self.current_hash = parent_hash
        self.commit_counter += 1
        self.set_full_permissions()

        print(
            f"Repository is now at commit: {self.current_hash}, previously invoked move_to_earlier_commit: {self.commit_counter} times")
        return parent_hash

    def move_to_later_commit(self):
        """
        Moves the repository checkout back to the child commit.
        """

        if self.previous_hash is None:
            print("Either attempting to reverse second time, or first commit")
            return "Error"

        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        if not self.git_checkout_with_retry(dest_dir, self.previous_hash):
            print("Error checking out child commit.")
            return "Error"

        self.current_hash = self.previous_hash
        self.previous_hash = None
        self.set_full_permissions()

        print(f"Repository is now at commit: {self.current_hash}")
        return self.current_hash

    def git_checkout_with_retry(self, dest_dir, target_hash, retries=1000):
        print("retrying")
        for attempt in range(1, retries + 1):
            subprocess.run(["git", "reset", "--hard"], cwd=dest_dir)
            subprocess.run(["git", "clean", "-fdx"], cwd=dest_dir)

            proc = subprocess.run(["git", "checkout", target_hash], cwd=dest_dir, capture_output=True, text=True)

            if proc.returncode == 0:
                return True

            print(f"Checkout to {target_hash} failed (attempt {attempt}): {proc.stderr}")

            time.sleep(1)

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
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                test_methods.append([test_rel_path, node.name])
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                inherits_testcase = any(
                    (isinstance(base, ast.Name) and base.id == "TestCase") or
                    (isinstance(base, ast.Attribute) and base.attr == "TestCase")
                    for base in node.bases
                )
                if inherits_testcase:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
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

        # --- BROKEN ---
        subprocess.run(["git", "checkout", broken_hash], cwd=dest_dir)
        broken_data = self.get_covered_source(rel_path, test_method, broken_hash)

        executed_methods_broken = {}
        for filename in broken_data.measured_files():
            if not filename.endswith(".py") or "tests" in filename:
                continue
            lines = broken_data.lines(filename)
            if lines:
                executed_methods_broken.update(
                    self.extract_executed_methods_from_file(filename, lines)
                )

        # --- REPAIRED ---
        subprocess.run(["git", "checkout", repaired_hash], cwd=dest_dir)
        repaired_data = self.get_covered_source(rel_path, test_method, repaired_hash)

        executed_methods_repaired = {}
        for filename in repaired_data.measured_files():
            if not filename.endswith(".py") or "tests" in filename:
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
            return ""

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
        nodeid = f"{test_file_path.as_posix()}::{test_method}"

        # Run the test via coverage in parallel mode.
        cmd = [
            sys.executable, "-m", "coverage", "run", "--parallel-mode", "-m", "pytest",
            "--maxfail=1", "--disable-warnings", "--quiet", nodeid
        ]
        returncode, log = run_cmd(cmd, timeout=15 * 60, cwd=str(self.repo_dir), env=env)
        print("pytest/coverage run returned:", returncode)
        print("Log output:", log)

        # Combine coverage data from subprocesses.
        combine_cmd = [sys.executable, "-m", "coverage", "combine"]
        combine_return, combine_log = run_cmd(combine_cmd, timeout=15 * 60, cwd=str(self.repo_dir), env=env)
        print("Coverage combine returned:", combine_return)
        print("Coverage combine log:", combine_log)

        # Load the combined coverage data.
        cov_data_file = self.repo_dir / ".coverage"
        cov = coverage.Coverage(data_file=str(cov_data_file))
        cov.load()
        data = cov.get_data()

        return data

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
        print("Listing test files")
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
