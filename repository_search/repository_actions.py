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
    def __init__(self, repository_name, repository_path, current_hash=None, previous_hash = None):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.current_hash = current_hash
        self.previous_hash = previous_hash

    def get_repository_name(self):
        return self.repository_name
    def get_repository_path(self):
        return self.repository_path
    def get_current_hash(self):
        return self.current_hash
    def set_current_hash(self, new_hash):
        self.current_hash = new_hash

    def clone_repository_last(self):
        """
        Clones the repository from GitHub at its latest commit.
        Sets self.current_hash to the latest commit hash.
        Assumes repository_name is in the form "owner/repository".
        """
        # Construct the GitHub repository URL
        repo_url = f"https://github.com/{self.repository_name}.git"
        # Destination directory based on repository name (without owner)
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        if os.path.exists(dest_dir):
            print(f"Repository already exists at {dest_dir}. Removing it...")

            # If the destination already exists, remove it for a fresh clone
            def handle_remove_readonly(func, path, exc_info):
                # Change the permissions and then call the removal function again
                os.chmod(path, stat.S_IWRITE)
                func(path)

            shutil.rmtree(dest_dir, onerror=handle_remove_readonly)

        print(f"Cloning repository from {repo_url} to {dest_dir}...")
        cmd = ["git", "clone", repo_url, dest_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)

        sys.path = [p for p in sys.path if self.repository_path not in p]

        # Prepare the three paths to add.
        repo_root = str(Path(self.repository_path).resolve())
        repo_folder = str((Path(self.repository_path) / self.repository_name.split("/")[-1]).resolve())
        nested_repo = str((Path(self.repository_path) / self.repository_name.split("/")[-1] /
                           self.repository_name.split("/")[-1]).resolve())

        # Add the paths to sys.path.
        for p in [repo_root, repo_folder, nested_repo]:
            if p not in sys.path:
                sys.path.insert(0, p)
                print(f"Added repository path to sys.path: {p}")


        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        print(subprocess.run(cmd, capture_output=True, text=True, env=os.environ, cwd=dest_dir))

        # Update current_hash with the latest commit hash from the cloned repository
        hash_cmd = ["git", "rev-parse", "HEAD"]
        hash_result = subprocess.run(hash_cmd, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if hash_result.returncode == 0:
            latest_hash = hash_result.stdout.strip()
            self.set_current_hash(latest_hash)
            print(f"Current commit hash set to: {self.current_hash}")
        else:
            print("Error obtaining latest commit hash:", hash_result.stderr)
            raise Exception("Failed to obtain commit hash")

        return dest_dir

    def has_tests(self):
        """
        Checks if the repository contains Python test files.
        This function recursively walks through the repository directory,
        looking for .py files that contain test-related keywords such as
        "import pytest", "from unittest", or function definitions that start with "def test_".
        """
        print("checking if tests exist")
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        # test_patterns = [r"import\s+pytest", r"from\s+unittest", r"def\s+test_", r"def\s+testing_", r"def\s+tests_"]
        # test_patterns = [r"def\s+test_", r"import\s+pytest", r"from\s+unittest"]
        test_patterns = [r"def\s+test_"]
        # Recursively iterate through all .py files in the repository
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding="utf-8").splitlines()
                        for line in content:
                            # Ensure line is not commented out
                            if any(re.search(pattern, line) for pattern in test_patterns):
                                print(f"Test file found: {file_path}")
                                return True
                    except Exception as e:
                        continue
        print("tests not found")
        return False

    def has_broken_to_repaired_test(self):
        test_files = self.list_test_files()
        working_test_methods = set()
        for path_to_file in test_files:
            test_methods = self.find_test_methods(path_to_file)
            for rel_path, test_method in test_methods:
                out = compile_and_run_test_python(Path(self.repository_path) / self.repository_name.split("/")[-1], rel_path, test_method, Path(self.repository_name).parent)
                if out.status == TestVerdict.SUCCESS:
                    working_test_methods.add((rel_path, test_method))
                elif (rel_path, test_method) in working_test_methods:
                    if self.check_if_test_changed(rel_path, test_method):
                        print("broken to repaired test found")
                        return True
                    else:
                        print("broken to repaired test not found, continuing")
                        continue
        while not self.move_to_earlier_commit().__eq__("Error"):
            print("has broken to repaired test loop")
            print("currently working test methods: "+working_test_methods.__str__())
            for path_to_file in test_files:
                test_methods = self.find_test_methods(path_to_file)
                for rel_path, test_method in test_methods:
                    out = compile_and_run_test_python(Path(self.repository_path) / self.repository_name.split("/")[-1], rel_path, test_method, Path(self.repository_name).parent)
                    if out.status == TestVerdict.SUCCESS:
                        working_test_methods.add((rel_path, test_method))
                    elif (rel_path, test_method) in working_test_methods:
                        if self.check_if_test_changed(rel_path, test_method):
                            print("broken to repaired test found")
                            return True
                        else:
                            print("broken to repaired test found, continuing")
                            continue
        print("broken to repaired tests are not found")
        return False


    def extract_broken_to_repaired_list(self):
        self.move_to_later_commit()
        print("extracting broken to repaired tests list")
        failing_tests = set() #rel_path, method_name, failing_hash, fixed_hash
        test_files = self.list_test_files()
        working_test_methods = set()
        for path_to_file in test_files:
            test_methods = self.find_test_methods(path_to_file)
            for rel_path, test_method in test_methods:
                out = compile_and_run_test_python(Path(self.repository_path) / self.repository_name.split("/")[-1], rel_path, test_method, Path(self.repository_name).parent)
                if out.status == TestVerdict.SUCCESS:
                    working_test_methods.add((rel_path, test_method))
                elif (rel_path, test_method) in working_test_methods:
                    if self.check_if_test_changed(rel_path, test_method):
                        failing_tests.add(Broken_to_repaired(self.current_hash, self.previous_hash, test_method, rel_path))
                        working_test_methods.remove((rel_path, test_method))
                        print("broken test found")
                        continue
                    else:
                        print("broken test not found, continuing")
                        continue
                else:
                    if (rel_path, test_method) in working_test_methods:
                        working_test_methods.remove((rel_path, test_method))
                        print("working test not found, continuing")
        while not self.move_to_earlier_commit().__eq__("Error"):
            print("extracting broken to repaired tests list outer loop")
            for path_to_file in test_files:
                print("extracting broken to repaired tests list inner loop")
                test_methods = self.find_test_methods(path_to_file)
                for rel_path, test_method in test_methods:
                    out = compile_and_run_test_python(Path(self.repository_path) / self.repository_name.split("/")[-1], rel_path, test_method, Path(self.repository_name).parent)
                    if out.status == TestVerdict.SUCCESS:
                        working_test_methods.add((rel_path, test_method))
                    elif (rel_path, test_method) in working_test_methods:
                        if self.check_if_test_changed(rel_path, test_method):
                            failing_tests.add(Broken_to_repaired(self.current_hash, self.previous_hash, test_method, rel_path))
                            working_test_methods.remove((rel_path, test_method))
                            print("broken test found")
                            continue
                        else:
                            print("broken test not found, continuing")
                            continue
                    else:
                        if (rel_path, test_method) in working_test_methods:
                            working_test_methods.remove((rel_path, test_method))
                            print("working test not found, continuing")
        print("broken to repair tests list extraction attempt finished")
        return failing_tests

    def extract_and_annotate_code(self, broken_to_repaired_instance):
        """
        Checkouts to both commits (broken and repaired) for a given test,
        extracts the test code and the relevant source code from the repaired commit,
        annotates the code (as in the reference paper), and returns the annotated versions.

        Parameters:
          broken_to_repaired_instance: An instance of Broken_to_repaired containing:
                                       - broken: commit hash when test was broken.
                                       - repaired: commit hash when test was fixed.
                                       - test_name: name of the test method.
                                       - rel_path: relative path to the test file.

        Returns:
          A tuple (annotated_source_code, annotated_test_code) or raises an error if checkout fails.
        """
        # Determine repository directory (assuming repository is cloned under repository_path/repo_folder)
        print("attempting to extract code")
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        # Checkout to the broken commit
        cmd_checkout = ["git", "checkout", broken_to_repaired_instance.broken]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"

        # Update the current hash to the broken commit
        self.current_hash = broken_to_repaired_instance.broken
        self.set_full_permissions()
        print(f"Repository is now at broken commit: {self.current_hash}")

        # Extract the broken test code from the broken commit
        broken_test = self.extract_method_code(broken_to_repaired_instance.rel_path,
                                               broken_to_repaired_instance.test_name)

        # Checkout to the repaired commit
        cmd_checkout = ["git", "checkout", broken_to_repaired_instance.repaired]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"

        # Update the current hash to the repaired commit
        self.current_hash = broken_to_repaired_instance.repaired
        self.set_full_permissions()
        print(f"Repository is now at repaired commit: {self.current_hash}")

        # Extract the repaired test code from the repaired commit
        repaired_test = self.extract_method_code(broken_to_repaired_instance.rel_path,
                                                 broken_to_repaired_instance.test_name)
        # Extract the relevant source code from the repaired commit
        source_code = self.extract_covered_source_coverage(broken_to_repaired_instance.rel_path,
                                                           broken_to_repaired_instance.test_name)

        # Annotate the extracted code as per the reference paper
        annotated_code = self.annotate_code(repaired_test, broken_test, source_code)

        # Return the annotated code (could be a tuple or a custom object)
        return annotated_code

    def annotate_code(self, repaired_test, broken_test, source_code):
        """
        Given the repaired test code, the broken test code, and the related source code,
        this method returns a single annotated string that contains:

          - A TESTCONTEXT section that includes:
              - The broken test code wrapped in [<BREAKAGE>] and [</BREAKAGE>]
              - The repaired test code wrapped in [<REPAIREDTEST>] and [</REPAIREDTEST>]
              - A diff between the broken and repaired test code, with hunks annotated using:
                    [<HUNK>] ... [</HUNK>] for diff hunk headers,
                    [<DEL>] ... [</DEL>] for deleted lines, and
                    [<ADD>] ... [</ADD>] for added lines.
          - A REPAIRCONTEXT section that includes:
              - The related source code wrapped in [<HUNK>] and [</HUNK>]

        Returns:
            str: A single annotated string containing all the information.
        """
        # Generate a unified diff between broken_test and repaired_test.
        print("attempting to annotate code")
        broken_lines = broken_test.splitlines()
        repaired_lines = repaired_test.splitlines()
        diff_lines = list(
            difflib.unified_diff(broken_lines, repaired_lines, fromfile="Broken Test", tofile="Repaired Test", lineterm=""))

        # Annotate the diff: wrap hunk headers, deletions, and additions.
        annotated_diff = ""
        for line in diff_lines:
            if line.startswith('@@'):
                annotated_diff += f"\n[<HUNK>]{line}[</HUNK>]\n"
            elif line.startswith('-'):
                annotated_diff += f"[<DEL>]{line}[</DEL>]\n"
            elif line.startswith('+'):
                annotated_diff += f"[<ADD>]{line}[</ADD>]\n"
            else:
                annotated_diff += line + "\n"

        # Construct the final annotated string.
        annotated_string = (
                "[<TESTCONTEXT>]\n"
                "[<BREAKAGE>]\n" +
                broken_test + "\n"
                              "[</BREAKAGE>]\n"
                              "[<REPAIREDTEST>]\n" +
                repaired_test + "\n"
                                "[</REPAIREDTEST>]\n"
                                "[<TESTDIFF>]\n" +
                annotated_diff + "\n"
                                 "[</TESTDIFF>]\n"
                                 "[</TESTCONTEXT>]\n\n"
                                 "[<REPAIRCONTEXT>]\n"
                                 "[<HUNK>]\n" +
                source_code + "\n"
                              "[</HUNK>]\n"
                              "[</REPAIRCONTEXT>]\n"
        )

        return annotated_string

    def move_to_earlier_commit(self):
        """
        Moves the repository checkout to the parent commit of the current HEAD.
        Updates the current_hash accordingly.
        """
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        print(f"Current repository directory: {dest_dir}")

        # Get the parent commit hash (HEAD^)
        cmd_parent = ["git", "rev-parse", "HEAD^"]
        proc_parent = subprocess.run(cmd_parent, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_parent.returncode != 0:
            return "Error"

        parent_hash = proc_parent.stdout.strip()
        print(f"Parent commit hash: {parent_hash}")

        # Checkout the parent commit
        cmd_checkout = ["git", "checkout", parent_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"

        # Update the current hash to the parent commit
        self.current_hash = parent_hash
        self.set_full_permissions()
        print(f"Repository is now at commit: {self.current_hash}")
        return parent_hash

    def move_to_later_commit(self):
        """
        Moves the repository checkout to the child commit of the current HEAD.
        Updates the current_hash accordingly.
        """

        if self.previous_hash is None:
            print("either attempting to reverse second time, or first commit")
            return "Error"
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        print(f"Current repository directory: {dest_dir}")

        # Get the parent commit hash (HEAD^)
        cmd_checkout = ["git", "checkout", self.previous_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            print("Error checking out child commit:", proc_checkout.stderr)
            return "Error"

        # Update the current hash to the child commit
        self.current_hash = self.previous_hash
        self.previous_hash = None
        self.set_full_permissions()
        print(f"Repository is now at commit: {self.current_hash}")
        return self.current_hash

    def find_test_methods(self, test_rel_path):
        """
        Inspects a Python file at the given test_rel_path and returns a list of test methods.
        Each element in the returned list is a list: [test_rel_path, test_method],
        where test_rel_path is the file repository_path relative to full_test_path, and test_method is the name
        of the test function. For methods in a class, the name is returned in the form "ClassName.method_name".

        Parameters:
          test_rel_path (str or Path): The repository_path to the Python source file.
          full_test_path (str or Path): The base directory to which the relative repository_path should be computed.

        Returns:
          list: A list of lists, each containing [test_rel_path, test_method].
        """
        print("fnding test methods")
        # print(self.repository_path +"/"+ self.repository_name.split("/")[-1]+"/"+test_rel_path.)
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

        # Compute the relative repository_path from full_test_path

        test_methods = []

        # Collect module-level test functions (e.g., def test_example(): ...)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                # We add the method with the relative file repository_path.
                test_methods.append([test_rel_path, node.name])

        # Collect methods defined inside classes that inherit from unittest.TestCase
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                inherits_testcase = False
                for base in node.bases:
                    # Check if the class inherits from TestCase either directly or as an attribute (e.g., unittest.TestCase)
                    if (isinstance(base, ast.Name) and base.id == "TestCase") or \
                            (isinstance(base, ast.Attribute) and base.attr == "TestCase"):
                        inherits_testcase = True
                if inherits_testcase:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            # Append with the format "ClassName.method_name"
                            test_methods.append([test_rel_path, f"{node.name}.{item.name}"])

        return test_methods

    def compare_ast_similarity(self, ast1, ast2):
        """
        Compares two AST nodes by dumping them (ignoring field annotations)
        and computing a similarity ratio.

        Returns:
            float: A similarity ratio between 0 and 1.
        """

        print("checking changeset similarity")
        dump1 = ast.dump(ast1, annotate_fields=False)
        dump2 = ast.dump(ast2, annotate_fields=False)
        return SequenceMatcher(None, dump1, dump2).ratio()

    def extract_covered_source_coverage(self, rel_path, test_method):
        """
        Uses coverage.py to run the specified test and returns a dictionary mapping file paths
        (as strings) to the set of line numbers executed in those files during the test run.

        This method dynamically measures coverage across the entire repository (not just a single source file).

        Parameters:
            rel_path (str): Relative path (from repository root) to the test file.
            test_method (str): Name of the test method (e.g., 'test_example').

        Returns:
            dict: A dictionary where keys are file paths and values are sets of executed line numbers.
                  If no coverage data is found, returns an empty dictionary.
        """
        print("Attempting to extract dynamic coverage data for the entire repository...")

        # Determine the repository directory
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        if not repo_dir.exists():
            print(f"Warning: Repository directory {repo_dir} not found.")
            return {}

        # Initialize coverage measurement over the entire repository
        cov = coverage.Coverage(source=[str(repo_dir)])
        cov.start()

        # Build the pytest nodeid for the test (file and test method)
        test_file_path = repo_dir / rel_path
        nodeid = f"{test_file_path.as_posix()}::{test_method}"

        # Run the test using the same run_cmd helper (assumed to be available)
        returncode, log = run_cmd(["pytest", "--maxfail=1", "--disable-warnings", "--quiet", nodeid],
                                  timeout=15 * 60, cwd=str(Path(self.repository_path) / self.repository_name.split("/")[-1]), env=os.environ)

        cov.stop()
        cov.save()

        # Retrieve coverage data: create a dict mapping file paths to sets of executed lines.
        data = cov.get_data()
        coverage_dict = {}
        for filename in data.measured_files():
            executed_lines = data.lines(filename)
            if executed_lines:
                coverage_dict[filename] = set(executed_lines)

        return coverage_dict

    def check_if_test_changed(self, rel_path, test_method):
        """
        Determines whether a test repair is "pure"—i.e., the test method itself has changed
        significantly (AST similarity < 95%), while the underlying source code that the test covers remains
        essentially unchanged (coverage similarity >= 95%).

        Returns:
          bool: True if the test changed and the covered source code is at least 95% similar between commits.
        """

        print("checking changeset similarity")
        # Save current commit hash.
        current_commit = self.current_hash

        # Extract AST of the test method in the current commit.
        original_test_ast = self.extract_method_ast(rel_path, test_method)
        if original_test_ast is None:
            return False

        # Extract coverage fingerprint of the relevant source in the current commit.
        original_coverage = self.extract_covered_source_coverage(rel_path, test_method)

        # Move to the later commit (you must implement move_to_later_commit).
        if self.move_to_later_commit() == "Error":
            return False

        # Extract AST of the test method in the later commit.
        new_test_ast = self.extract_method_ast(rel_path, test_method)
        # Extract coverage fingerprint of the relevant source in the later commit.
        new_coverage = self.extract_covered_source_coverage(rel_path, test_method)

        # Move back to the original commit.
        self.move_to_earlier_commit()

        # Compare ASTs for the test method.
        test_similarity = self.compare_ast(original_test_ast, new_test_ast)

        # Compute Jaccard similarity for coverage fingerprints.
        if not original_coverage or not new_coverage:
            print("Coverage data missing; cannot determine pure test repair.")
            return False

        intersection = original_coverage.intersection(new_coverage)
        union = original_coverage.union(new_coverage)
        coverage_similarity = len(intersection) / len(union) if union else 1.0
        print(f"Source coverage similarity for {test_method}: {coverage_similarity:.2f}")

        # - Source coverage similarity is at least 0.95 (i.e. the executed source remains nearly the same)
        return not test_similarity and coverage_similarity >= 0.95

    def extract_method_code(self, rel_path, test_method):
        """
        Extracts the source code of a given test method from a specified file.

        Parameters:
            rel_path (str): The relative repository_path to the test file.
            test_method (str): The name of the test method to extract.

        Returns:
            str: The extracted method code as a string, or an empty string if not found.
        """

        print("extracting method code")
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
            """AST visitor class to extract the function source code."""

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
        """
        Extracts the AST of a given test method from a specified file.

        Parameters:
            rel_path (str): The relative repository_path to the test file.
            test_method (str): The name of the test method to extract.

        Returns:
            ast.AST: The AST of the function, or None if not found.
        """
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
                return node  # Return the AST of the test function

        print(f"Warning: Test method {test_method} not found in {test_file_path}")
        return None

    def compare_ast(self, ast1, ast2):
        """
        Compares two AST nodes to determine if they are structurally the same.

        Parameters:
            ast1 (ast.AST): The AST of the first function.
            ast2 (ast.AST): The AST of the second function.

        Returns:
            bool: True if the ASTs are equivalent, False if they have structural differences.
        """

        print("checking changeset similarity")
        if ast1 is None or ast2 is None:
            return False  # One of the ASTs is missing, meaning the method changed

        return ast.dump(ast1, annotate_fields=False) == ast.dump(ast2, annotate_fields=False)

    def list_test_files(self):
        print("attempting to get a list of test files")
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        files_paths = []
        # test_patterns = [r"import\s+pytest", r"from\s+unittest", r"def\s+test_", r"def\s+testing_", r"def\s+tests_"]
        test_patterns = [r"def\s+test_"]
        # Recursively iterate through all .py files in the repository
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding="utf-8").splitlines()
                        for line in content:
                            # Ensure line is not commented out
                            if any(re.search(pattern, line) for pattern in test_patterns):
                                files_paths.append(file_path)
                    except Exception as e:
                        continue
        return files_paths

    def set_full_permissions(self):
        """
        Recursively sets full permissions (read, write, execute) for all files and directories
        in the given directory.
        """
        os.chmod(self.repository_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for root, dirs, files in os.walk(self.repository_path):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Full access

            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    def install_dependencies(self):
        """
        Attempts to install the required dependencies for a repository.

        It first checks for a requirements.txt file in the repository root.
        If found, it runs:
            pip install -r requirements.txt

        Additionally, if a setup.py file is present, it installs the package via:
            pip install .

        Parameters:
            repo_dir (str or Path): The path to the cloned repository.
        """
        repo_path = Path(self.repository_path) / self.repository_name.split("/")[-1]

        # Check for requirements.txt and install dependencies
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            print(f"Installing dependencies from {req_file}")
            try:
                subprocess.run(
                    ["pip", "install", "-r", str(req_file)],
                    check=True, env=os.environ
                )
            except subprocess.CalledProcessError as e:
                print(f"Error installing requirements from {req_file}: {e}")
        else:
            print("No requirements.txt found.")

        # Optionally, check for setup.py and install the package if needed
        setup_file = repo_path / "setup.py"
        if setup_file.exists():
            print("Found setup.py; installing package...")
            try:
                subprocess.run(
                    ["pip", "install", "."],
                    cwd=str(repo_path),
                    check=True, env=os.environ
                )
            except subprocess.CalledProcessError as e:
                print(f"Error installing package via setup.py: {e}")
        else:
            print("No setup.py found.")

# test = RepositoryActions("andkuzm/fourth_task", "C:\\Users\\kandr\\PycharmProjects\\PyTaRGET\\test")
# test.clone_repository_last()
# print(test.current_hash)
# test.move_to_earlier_commit()
# print(test.current_hash)
# print(test.find_test_methods("src/assets/t.py"))