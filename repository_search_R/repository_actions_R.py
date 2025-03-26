import ast
import difflib
import os
import re
import stat
import subprocess
from pathlib import Path

from data_types.Broken_to_repaired import Broken_to_repaired
from repository_search.py_parser import compile_and_run_test_python, TestVerdict


class RepositoryActions:
    def __init__(self, repository_name, repository_path, current_hash, out_path, previous_hash=None):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.current_hash = current_hash
        self.out_path = out_path
        self.previous_hash = previous_hash
        self.visited_commits = set()
        self.commit_counter = 0

    def checkout_commit(self, commit_hash, dest_dir):
        cmd_checkout = ["git", "checkout", commit_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            print(f"Error checking out commit {commit_hash}: {proc_checkout.stderr}")
            return False
        print("current hash is", self.current_hash)
        return True

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
                print("test methods: "+str(test_methods))
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
                    repaired_case = Broken_to_repaired(parent_commit, self.current_hash, test_method, rel_path, result.log)
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
        print("moving to parent commit")
        # Checkout the parent commit
        cmd_checkout = ["git", "checkout", parent_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"
        self.current_hash = parent_hash
        self.commit_counter += 1  # Increment commit counter for every successful commit move.
        self.set_full_permissions()
        print(f"Repository is now at commit: {self.current_hash}, previously invoked move_to_earlier_commit: {self.commit_counter} times")
        return parent_hash

    def run_test_with_overridden_test_code(self, rel_path, test_method, overridden_test_code):
        """
        Temporarily replaces the test file's content with overridden_test_code,
        runs the specified test, and then restores the original content.
        If the change in test code is deemed unimportant (i.e. nearly identical to the current test code),
        the method returns a dummy success result immediately.
        """
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path
        original_content = test_file_path.read_text(encoding="utf-8")

        # Extract the current test code from the file.
        current_test_code = self.extract_method_code(rel_path, test_method)

        # Compute similarity ratio between current and overridden test code.
        similarity = difflib.SequenceMatcher(None, current_test_code, overridden_test_code).ratio()
        # If similarity is very high (e.g. over 98%), consider the change unimportant.
        if similarity > 0.98:
            print(f"Change is unimportant (similarity {similarity:.2f}); skipping override test execution.")
            return TestVerdict(status=TestVerdict.SUCCESS, error_lines="test skipped due to high similarity")

        try:
            # Write the overridden (parent's) test code.
            test_file_path.write_text(overridden_test_code, encoding="utf-8")
            repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
            result = compile_and_run_test_python(repo_dir, rel_path, test_method, repo_dir.parent)
        finally:
            # Always restore the original content.
            test_file_path.write_text(original_content, encoding="utf-8")
        return result

    def move_to_later_commit(self):
        """
        Moves the repository checkout back to the child commit.
        """
        if self.previous_hash is None:
            print("Either attempting to reverse second time, or first commit")
            return "Error"
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        print(f"Current repository directory: {dest_dir}")
        print("moving to child commit")
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

    def set_full_permissions(self):
        os.chmod(self.repository_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for root, dirs, files in os.walk(self.repository_path):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

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

    def extract_executed_methods(self, source_code, rel_path):
        """
        Extracts full methods that were executed during the test.
        Parses the executed source, finds the full method definitions from the full file,
        and returns them as a dictionary with keys in the format 'ClassName.method_name'.
        """
        executed_methods = {}
        test_file_path = Path(self.repository_path) / self.repository_name.split("/")[-1] / rel_path

        if not test_file_path.exists():
            print(f"Warning: Test file {test_file_path} does not exist.")
            return executed_methods

        try:
            full_source = test_file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {test_file_path}: {e}")
            return executed_methods

        try:
            tree = ast.parse(full_source)
        except Exception as e:
            print(f"Error parsing {test_file_path}: {e}")
            return executed_methods

        # Normalize executed lines (trim whitespace and skip empties)
        executed_lines = set(line.strip() for line in source_code.splitlines() if line.strip())

        # Helper to determine if a method is "executed" based on a threshold ratio.
        def is_method_executed(method_source):
            method_lines = [line.strip() for line in method_source.splitlines() if line.strip()]
            if not method_lines:
                return False
            count = 0
            for line in method_lines:
                for executed in executed_lines:
                    # Either exact match or one is contained in the other.
                    if line == executed or line in executed or executed in line:
                        count += 1
                        break
            ratio = count / len(method_lines)
            return ratio >= 0.3  # threshold: at least 30% of lines are found

        class FunctionExtractor(ast.NodeVisitor):
            def __init__(self):
                self.methods = {}
                self.current_class = "Global"

            def visit_ClassDef(self, node):
                previous_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = previous_class

            def visit_FunctionDef(self, node):
                method_source = ast.get_source_segment(full_source, node)
                if method_source and is_method_executed(method_source):
                    class_key = self.current_class if self.current_class else "Global"
                    self.methods[f"{class_key}.{node.name}"] = method_source
                self.generic_visit(node)

        extractor = FunctionExtractor()
        extractor.visit(tree)
        return extractor.methods

    def is_test_method_changed(self, parent_code, child_code):
        # Normalize each line by stripping trailing whitespace.
        parent_lines = [line.rstrip() for line in parent_code.splitlines()]
        child_lines = [line.rstrip() for line in child_code.splitlines()]

        # Get the unified diff and filter out header lines.
        diff = list(difflib.unified_diff(parent_lines, child_lines, lineterm=""))
        diff = self.filter_diff_lines(diff)

        # Count the number of contiguous blocks of change lines.
        block_count = 0
        in_block = False
        has_functional_change = False

        def is_functional_line(line):
            # Strip diff markers and check if the line is meaningful code.
            line_content = line[1:].strip()  # Remove leading + or -
            return (
                    line_content
                    and not line_content.startswith("#")  # Ignore comments
                    and not (line_content.startswith(('"""', "'''")) and line_content.endswith(('"""', "'''")))
            # Ignore full-line docstrings
            )

        for line in diff:
            if line.startswith('+') or line.startswith('-'):
                if not in_block:
                    block_count += 1
                    in_block = True
                # Check for functional change in each changed line
                if is_functional_line(line):
                    has_functional_change = True
            else:
                in_block = False

        # Return True only if there's exactly one contiguous block of *functional* changes.
        return block_count == 1 and has_functional_change

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