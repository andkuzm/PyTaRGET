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

        sys.path = [p for p in sys.path if self.repository_path not in p]

        repo_root = str(Path(self.repository_path).resolve())
        repo_folder = str((Path(self.repository_path) / self.repository_name.split("/")[-1]).resolve())
        nested_repo = str((Path(self.repository_path) / self.repository_name.split("/")[-1] / self.repository_name.split("/")[-1]).resolve())

        for p in [repo_root, repo_folder, nested_repo]:
            if p not in sys.path:
                sys.path.insert(0, p)
                print(f"Added repository path to sys.path: {p}")

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
        Implements the new approach over a single parent-child pair:
          1. Checkout the parent commit and record passing tests (with their source).
          2. Checkout back to the child commit and record passing tests.
          3. For tests passing in both commits, override the child’s test file with the parent’s test code.
             If the parent’s test now fails on the child’s source, record the case.
        Returns:
          A set of Broken_to_repaired objects representing detected repaired test cases.
        """
        repaired_cases = set()
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        # Move to parent commit.
        parent_commit = self.move_to_earlier_commit()
        if parent_commit == "Error":
            return repaired_cases

        print(f"At parent commit: {parent_commit}")
        parent_passed = {}
        for file in self.list_test_files():
            rel_path = str(file.relative_to(repo_dir))
            test_methods = self.find_test_methods(rel_path)
            for (rp, test_method) in test_methods:
                out = compile_and_run_test_python(repo_dir, rel_path, test_method, repo_dir.parent)
                if out.status == TestVerdict.SUCCESS:
                    parent_passed[(rel_path, test_method)] = self.extract_method_code(rel_path, test_method)

        # Move back to child commit.
        if self.move_to_later_commit() == "Error":
            return repaired_cases

        print(f"At child commit: {self.current_hash}")
        child_passed = {}
        for file in self.list_test_files():
            rel_path = str(file.relative_to(repo_dir))
            test_methods = self.find_test_methods(rel_path)
            for (rp, test_method) in test_methods:
                out = compile_and_run_test_python(repo_dir, rel_path, test_method, repo_dir.parent)
                if out.status == TestVerdict.SUCCESS:
                    child_passed[(rel_path, test_method)] = self.extract_method_code(rel_path, test_method)

        # For tests that passed in both commits, run parent's test code against child's source.
        for key, parent_code in parent_passed.items():
            if key in child_passed:
                result = self.run_test_with_overridden_test_code(key[0], key[1], parent_code)
                if result.status != TestVerdict.SUCCESS:
                    print(f"Repaired test detected: {key}")
                    repaired_case = Broken_to_repaired(parent_commit, self.current_hash, key[1], key[0])
                    repaired_cases.add(repaired_case)

        return repaired_cases

    def extract_and_annotate_code(self, broken_to_repaired_instance):
        print("Attempting to extract and annotate code")
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        # Checkout to the parent's commit (broken test).
        cmd_checkout = ["git", "checkout", broken_to_repaired_instance.broken]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"
        self.current_hash = broken_to_repaired_instance.broken
        self.set_full_permissions()
        print(f"Repository at broken commit: {self.current_hash}")
        broken_test = self.extract_method_code(broken_to_repaired_instance.rel_path, broken_to_repaired_instance.test_name)

        # Checkout to the child's commit (repaired test).
        cmd_checkout = ["git", "checkout", broken_to_repaired_instance.repaired]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"
        self.current_hash = broken_to_repaired_instance.repaired
        self.set_full_permissions()
        print(f"Repository at repaired commit: {self.current_hash}")
        repaired_test = self.extract_method_code(broken_to_repaired_instance.rel_path, broken_to_repaired_instance.test_name)
        source_code = self.extract_covered_source_coverage(broken_to_repaired_instance.rel_path, broken_to_repaired_instance.test_name)
        annotated_code = self.annotate_code(repaired_test, broken_test, source_code)
        return annotated_code

    def annotate_code(self, repaired_test, broken_test, source_code):
        print("Annotating code")
        broken_lines = broken_test.splitlines()
        repaired_lines = repaired_test.splitlines()
        diff_lines = list(difflib.unified_diff(broken_lines, repaired_lines, fromfile="Broken Test", tofile="Repaired Test", lineterm=""))
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

        annotated_string = (
            "[<TESTCONTEXT>]\n"
            "[<BREAKAGE>]\n" + broken_test + "\n[</BREAKAGE>]\n"
            "[<REPAIREDTEST>]\n" + repaired_test + "\n[</REPAIREDTEST>]\n"
            "[<TESTDIFF>]\n" + annotated_diff + "\n[</TESTDIFF>]\n"
            "[</TESTCONTEXT>]\n\n"
            "[<REPAIRCONTEXT>]\n"
            "[<HUNK>]\n" + source_code + "\n[</HUNK>]\n"
            "[</REPAIRCONTEXT>]\n"
        )
        return annotated_string

    def move_to_earlier_commit(self):
        """
        Moves the repository checkout to the parent commit of the current HEAD.
        Before moving, saves the current commit as previous_hash.
        Raises an exception if a cycle is detected.
        """
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
        if parent_hash == self.current_hash:
            raise Exception("Cycle detected: parent commit is the same as current commit.")
        if parent_hash in self.visited_commits:
            raise Exception("Cycle detected: commit has already been visited.")
        self.visited_commits.add(parent_hash)

        print(f"Parent commit hash: {parent_hash}")

        # Checkout the parent commit
        cmd_checkout = ["git", "checkout", parent_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True, env=os.environ)
        if proc_checkout.returncode != 0:
            return "Error"
        self.current_hash = parent_hash
        self.set_full_permissions()
        print(f"Repository is now at commit: {self.current_hash}")
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

    def extract_covered_source_coverage(self, rel_path, test_method):
        print("Extracting dynamic coverage data")
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        if not repo_dir.exists():
            print(f"Warning: Repository directory {repo_dir} not found.")
            return {}
        cov = coverage.Coverage(source=[str(repo_dir)])
        cov.start()
        test_file_path = repo_dir / rel_path
        nodeid = f"{test_file_path.as_posix()}::{test_method}"
        returncode, log = run_cmd(["pytest", "--maxfail=1", "--disable-warnings", "--quiet", nodeid],
                                  timeout=15 * 60, cwd=str(repo_dir), env=os.environ)
        cov.stop()
        cov.save()
        data = cov.get_data()
        coverage_dict = {}
        for filename in data.measured_files():
            executed_lines = data.lines(filename)
            if executed_lines:
                coverage_dict[filename] = set(executed_lines)
        return coverage_dict

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
