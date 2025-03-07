import ast
import difflib
import os
import shutil
import stat
import subprocess
import sys
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

        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
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
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        test_patterns = [r"def\s+test_"]
        for root, dirs, files in os.walk(repo_dir):
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
        """
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        original_content = test_file_path.read_text(encoding="utf-8")
        try:
            test_file_path.write_text(overridden_test_code, encoding="utf-8")
            repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
            result = compile_and_run_test_python(repo_dir, rel_path, test_method, repo_dir.parent)
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
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        while True:
            # Save current commit as child commit for this pair.
            child_commit = self.current_hash

            # Attempt to move to the parent commit.
            parent_commit = self.move_to_earlier_commit()
            if parent_commit == "Error":
                print("No further commits or stop condition reached. Ending iteration.")
                break

            print(f"Processing commit pair: Parent: {parent_commit} | Child: {child_commit}")

            # --- Extract test methods in the parent commit ---
            parent_methods = {}
            for file in self.list_test_files():
                rel_path = str(file.relative_to(repo_dir))
                test_methods = self.find_test_methods(rel_path)
                for (_, test_method) in test_methods:
                    code = self.extract_method_code(rel_path, test_method)
                    if code:
                        parent_methods[(rel_path, test_method)] = code

            # --- Switch back to the child commit and extract test methods ---
            if self.move_to_later_commit() == "Error":
                print("Error moving back to child commit.")
                break

            child_methods = {}
            for file in self.list_test_files():
                rel_path = str(file.relative_to(repo_dir))
                test_methods = self.find_test_methods(rel_path)
                for (_, test_method) in test_methods:
                    code = self.extract_method_code(rel_path, test_method)
                    if code:
                        child_methods[(rel_path, test_method)] = code

            # --- Identify test methods that have changed ---
            changed_tests = set()
            for key in parent_methods:
                if key in child_methods:
                    if self.is_test_method_changed(parent_methods[key], child_methods[key]):
                        changed_tests.add(key)
                    else:
                        print(f"Test {key} unchanged; skipping.")
                else:
                    print(f"Test {key} not found in child commit; skipping.")

            # --- For each changed test, verify test results and run override ---
            for key in changed_tests:
                rel_path, test_method = key

                # Verify the test passes in the parent commit.
                if self.move_to_earlier_commit() == "Error":
                    print("Error moving to parent commit for verification.")
                    continue
                parent_result = compile_and_run_test_python(repo_dir, rel_path, test_method, repo_dir.parent)
                if parent_result.status != TestVerdict.SUCCESS:
                    print(f"Test {key} does not pass in parent commit; skipping.")
                    if self.move_to_later_commit() == "Error":
                        continue
                    continue

                # Switch back to the child commit.
                if self.move_to_later_commit() == "Error":
                    print("Error moving back to child commit for verification.")
                    continue
                child_result = compile_and_run_test_python(repo_dir, rel_path, test_method, repo_dir.parent)
                if child_result.status != TestVerdict.SUCCESS:
                    print(f"Test {key} does not pass in child commit; skipping.")
                    continue

                # Run the override: inject parent's test code into child's source.
                result = self.run_test_with_overridden_test_code(rel_path, test_method, parent_methods[key])
                if result.status != TestVerdict.SUCCESS:
                    print(f"Repaired test detected: {key}")
                    repaired_case = Broken_to_repaired(parent_commit, self.current_hash, test_method, rel_path)
                    repaired_cases.add(repaired_case)

            # --- Update current commit to parent's commit for the next iteration ---
            checkout_cmd = ["git", "checkout", parent_commit]
            proc = subprocess.run(checkout_cmd, cwd=str(repo_dir), capture_output=True, text=True, env=os.environ)
            if proc.returncode != 0:
                print(f"Error checking out parent commit {parent_commit}. Ending iteration.")
                break
            self.current_hash = parent_commit
            print(f"Updated current commit to {self.current_hash} for next iteration.")

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

        annotated_code = self.annotate_code(broken_test, repaired_test, source_code)
        return annotated_code

    def annotate_code(self, broken_test, repaired_test, source_code):
        print("Annotating code")
        broken_lines = broken_test.splitlines()
        repaired_lines = repaired_test.splitlines()
        diff_lines = list(
            difflib.unified_diff(broken_lines, repaired_lines, fromfile="Broken Test", tofile="Repaired Test",
                                 lineterm=""))

        unchanged_before = []
        breakage_lines = []
        unchanged_after = []
        repaired_lines_only = []
        in_change_block = False
        change_done = False

        for line in diff_lines:
            if line.startswith('@@'):
                if in_change_block:
                    change_done = True
                in_change_block = True
            elif in_change_block and not change_done:
                if line.startswith('-'):
                    breakage_lines.append(line[1:])
                elif line.startswith('+'):
                    repaired_lines_only.append(line[1:])
                else:
                    unchanged_after.append(line)
            elif not in_change_block:
                unchanged_before.append(line)
            else:
                unchanged_after.append(line)

        annotated_string = (
                "[<TESTCONTEXT>]" + "\n"
                + "\n".join(unchanged_before) + "\n"
                + "[<BREAKAGE>]" + "\n"
                + "\n".join(breakage_lines) + "\n"
                + "[</BREAKAGE>]" + "\n"
                + "\n".join(unchanged_after) + "\n"
                + "[</TESTCONTEXT>]" + "\n\n"
                + "[<REPAIREDTEST>]" + "\n"
                + "\n".join(repaired_lines_only) + "\n"
                + "[</REPAIREDTEST>]" + "\n\n"
                + "[<REPAIRCONTEXT>]" + "\n"
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
        print(str(self.commit_counter)+"th commit")
        if self.commit_counter >= 300:
            print("Commit counter limit reached. Stopping further processing of commits in this repository.")
            return "Error"

        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        print(f"Current repository directory: {dest_dir}")

        # Save the current commit as the child commit (for later reversal)
        self.previous_hash = self.current_hash

        # Get the parent commit hash (HEAD^)
        cmd_parent = ["git", "rev-parse", "HEAD^"]
        proc_parent = subprocess.run(cmd_parent, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_parent.returncode != 0:
            return "Error"
        parent_hash = proc_parent.stdout.strip()

        # Check for a cycle:
        # if parent_hash == self.current_hash:
        #     raise Exception("Cycle detected: parent commit is the same as current commit.")
        # if parent_hash in self.visited_commits:
        #     raise Exception("Cycle detected: commit has already been visited.")
        # self.visited_commits.add(parent_hash)

        print(f"Parent commit hash: {parent_hash}")

        # Checkout the parent commit
        cmd_checkout = ["git", "checkout", parent_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"
        self.current_hash = parent_hash
        self.commit_counter += 1  # Increment commit counter for every successful commit move.
        self.set_full_permissions()
        print(f"Repository is now at commit: {self.current_hash}, processed commits: {self.commit_counter}")
        return parent_hash

    def move_to_later_commit(self):
        """
        Moves the repository checkout back to the child commit.
        """
        if self.previous_hash is None:
            print("Either attempting to reverse second time, or first commit")
            return "Error"
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        print(f"Current repository directory: {dest_dir}")
        cmd_checkout = ["git", "checkout", self.previous_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            print("Error checking out child commit:", proc_checkout.stderr)
            return "Error"
        self.current_hash = self.previous_hash
        self.previous_hash = None  # Clear previous hash after moving back.
        self.set_full_permissions()
        print(f"Repository is now at commit: {self.current_hash}")
        return self.current_hash

    def find_test_methods(self, test_rel_path):
        print("Finding test methods")
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

    def compare_ast_similarity(self, ast1, ast2):
        print("Checking changeset similarity")
        dump1 = ast.dump(ast1, annotate_fields=False)
        dump2 = ast.dump(ast2, annotate_fields=False)
        return SequenceMatcher(None, dump1, dump2).ratio()

    def extract_covered_source_coverage(self, rel_path, test_method, broken_hash, repaired_hash):
        print("Extracting dynamic covered source code for both versions")

        # Extract covered source for both versions.
        broken_source = self.get_covered_source(rel_path, test_method, broken_hash)
        repaired_source = self.get_covered_source(rel_path, test_method, repaired_hash)

        # Split each source into classes (and their methods)
        broken_classes = self.split_into_classes(broken_source)
        repaired_classes = self.split_into_classes(repaired_source)

        result = []
        # Gather all class names (including 'Global' if any code is outside classes)
        all_class_names = set(broken_classes.keys()).union(repaired_classes.keys())
        for class_name in sorted(all_class_names):
            # Get dictionaries of methods for the given class (default to empty dict if not present)
            broken_methods = broken_classes.get(class_name, {})
            repaired_methods = repaired_classes.get(class_name, {})
            # Union of all method signatures in this class
            all_method_names = set(broken_methods.keys()).union(repaired_methods.keys())
            class_hunks = []
            for method_name in sorted(all_method_names):
                broken_code = broken_methods.get(method_name, '')
                repaired_code = repaired_methods.get(method_name, '')
                diff = list(difflib.unified_diff(broken_code.splitlines(), repaired_code.splitlines(), lineterm=""))
                formatted_hunk = self.format_inline_diff(method_name, diff)
                if formatted_hunk:
                    class_hunks.append(formatted_hunk)
            if class_hunks:
                result.append(f'class {class_name}:')
                result.extend(class_hunks)
        return "\n\n".join(result)

    def split_into_classes(self, source_code):
        """
        Splits the source code into a dictionary organized by class.
        Each key is the class name (or 'Global' if code is outside any class), and its value is a dict mapping
        method signatures to the corresponding method body.
        """
        classes = {}
        current_class = 'Global'
        current_method = None
        current_lines = []

        # Initialize the global group.
        if current_class not in classes:
            classes[current_class] = {}

        for line in source_code.splitlines():
            # Detect class definition.
            class_match = re.match(r'^\s*class\s+(\w+)', line)
            if class_match:
                # Finish any pending method before switching class.
                if current_method:
                    classes[current_class][current_method] = "\n".join(current_lines)
                    current_method = None
                    current_lines = []
                # Switch to the new class.
                current_class = class_match.group(1)
                if current_class not in classes:
                    classes[current_class] = {}
                continue

            # Detect a method definition.
            method_match = re.match(r'^\s*def\s+(\w+)', line)
            if method_match:
                # If we were collecting a previous method, save it.
                if current_method:
                    classes[current_class][current_method] = "\n".join(current_lines)
                current_method = line.strip()
                current_lines = [line]
            else:
                # If within a method, continue collecting its lines.
                if current_method:
                    current_lines.append(line)
        # Save any method still in progress.
        if current_method:
            classes[current_class][current_method] = "\n".join(current_lines)
        return classes

    def format_inline_diff(self, method_name, diff):
        """
        Returns a string that wraps changes for a given method in [<HUNK>] tags.
        Within the hunk, added lines are wrapped in [<ADD>] tags and removed lines in [<DEL>] tags.
        """
        hunk_lines = [f"[<HUNK>] {method_name}"]
        inside_hunk = False

        for line in diff:
            if line.startswith('@@'):
                inside_hunk = True
            elif inside_hunk:
                if line.startswith('+'):
                    hunk_lines.append(f"[<ADD>]{line[1:]}[</ADD>]")
                elif line.startswith('-'):
                    hunk_lines.append(f"[<DEL>]{line[1:]}[</DEL>]")
                else:
                    hunk_lines.append(line)
        hunk_lines.append("[</HUNK>]")
        return "\n".join(hunk_lines) if len(hunk_lines) > 2 else ""

    def get_covered_source(self, rel_path, test_method, commit_hash):
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        proc = subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            env=os.environ
        )
        if proc.returncode != 0:
            print(f"Git checkout failed: {proc.stderr}")
            return ""

        test_file_path = repo_dir / rel_path
        nodeid = f"{test_file_path.as_posix()}::{test_method}"
        cmd = [
            "python", "-m", "coverage", "run", "--parallel-mode", "-m", "pytest",
            "--maxfail=1", "--disable-warnings", "--quiet", nodeid
        ]
        returncode, log = run_cmd(cmd, timeout=15 * 60, cwd=str(repo_dir), env=os.environ)
        combine_cmd = ["python", "-m", "coverage", "combine"]
        combine_return, combine_log = run_cmd(combine_cmd, timeout=15 * 60, cwd=str(repo_dir), env=os.environ)

        cov_data_file = repo_dir / ".coverage"
        cov = coverage.Coverage(data_file=str(cov_data_file))
        cov.load()
        data = cov.get_data()

        covered_source = ""
        for filename in data.measured_files():
            executed_lines = data.lines(filename)
            if executed_lines:
                try:
                    with open(filename, encoding="utf-8") as f:
                        source_lines = f.readlines()
                    executed_source = "".join(source_lines[i - 1] for i in sorted(executed_lines))
                    covered_source += f"{executed_source}\n"
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
        return covered_source

    def extract_method_code(self, rel_path, test_method):
        print("Extracting method code")
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        if not test_file_path.exists():
            print(f"Error: Test file {test_file_path} does not exist.")
            return ""
        try:
            source_code = test_file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {test_file_path}: {e}")
            return ""
        try:
            tree = ast.parse(source_code)
        except Exception as e:
            print(f"Error parsing {test_file_path}: {e}")
            return ""
        class FunctionExtractor(ast.NodeVisitor):
            def __init__(self, method_name):
                self.method_name = method_name
                self.found_code = ""
            def visit_FunctionDef(self, node):
                if node.name == self.method_name:
                    self.found_code = ast.get_source_segment(source_code, node)
                self.generic_visit(node)
        extractor = FunctionExtractor(test_method)
        extractor.visit(tree)
        if not extractor.found_code:
            print(f"Warning: Test method {test_method} not found in {test_file_path}")
        return extractor.found_code.strip() if extractor.found_code else ""

    def extract_method_ast(self, rel_path, test_method):
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        if not test_file_path.exists():
            print(f"Error: Test file {test_file_path} does not exist.")
            return None
        try:
            source_code = test_file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {test_file_path}: {e}")
            return None
        try:
            tree = ast.parse(source_code)
        except Exception as e:
            print(f"Error parsing {test_file_path}: {e}")
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == test_method:
                return node
        print(f"Warning: Test method {test_method} not found in {test_file_path}")
        return None

    def compare_ast(self, ast1, ast2):
        print("Comparing ASTs")
        if ast1 is None or ast2 is None:
            return False
        return ast.dump(ast1, annotate_fields=False) == ast.dump(ast2, annotate_fields=False)

    # def is_test_method_changed(self, parent_code, child_code):
    #     # Normalize each line by stripping trailing whitespace.
    #     parent_lines = [line.rstrip() for line in parent_code.splitlines()]
    #     child_lines = [line.rstrip() for line in child_code.splitlines()]
    #     diff = list(difflib.unified_diff(parent_lines, child_lines, lineterm=""))
    #     return len(diff) > 0

    def is_test_method_changed(self, parent_code, child_code):
        # Normalize each line by stripping trailing whitespace.
        parent_lines = [line.rstrip() for line in parent_code.splitlines()]
        child_lines = [line.rstrip() for line in child_code.splitlines()]

        # Get the diff between the parent and child code.
        diff = list(difflib.unified_diff(parent_lines, child_lines, lineterm=""))

        # Flag to track if we are in a contiguous block of changes
        in_change_block = False
        last_change_line = None

        for line in diff:
            if line.startswith('+') or line.startswith('-'):
                if in_change_block:
                    # If we're already in a change block, continue (adjacent changes)
                    pass
                else:
                    if last_change_line is not None and line[0] != last_change_line[0]:
                        # If we are switching between addition and removal (non-adjacent change)
                        return False
                    # Start a new change block
                    in_change_block = True
                last_change_line = line
            else:
                # Non-change line; end of the current block
                if in_change_block:
                    in_change_block = False
                    last_change_line = None

        return True

    def list_test_files(self):
        print("Listing test files")
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        files_paths = []
        test_patterns = [r"def\s+test_"]
        for root, dirs, files in os.walk(repo_dir):
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

    def install_dependencies(self):
        repo_path = Path(self.repository_path) / self.repository_name.split("/")[-1]
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            print(f"Installing dependencies from {req_file}")
            try:
                subprocess.run(["pip", "install", "-r", str(req_file)], check=True, env=os.environ)
            except subprocess.CalledProcessError as e:
                print(f"Error installing requirements from {req_file}: {e}")
        else:
            print("No requirements.txt found.")
        setup_file = repo_path / "setup.py"
        if setup_file.exists():
            print("Found setup.py; installing package...")
            try:
                subprocess.run(["pip", "install", "."], cwd=str(repo_path), check=True, env=os.environ)
            except subprocess.CalledProcessError as e:
                print(f"Error installing package via setup.py: {e}")
        else:
            print("No setup.py found.")
