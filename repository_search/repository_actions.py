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
        print(broken_lines, "broken lines in annotated code")
        print(repaired_lines, "repaired lines in annotated code")

        # Compute the diff
        diff_lines = list(
            difflib.unified_diff(broken_lines, repaired_lines, fromfile="Broken Test", tofile="Repaired Test",
                                 lineterm="")
        )
        diff_lines = self.filter_diff_lines(diff_lines)

        if not diff_lines:
            return ""

        print(diff_lines)

        # Initialize sections
        unchanged_before = []
        breakage_lines = []
        unchanged_after = []
        repaired_lines_only = []

        # Track where we are in the diff
        in_change_block = False

        for line in diff_lines:
            if line.startswith('@@'):
                continue  # Skip the hunk header, no need to track
            elif line.startswith('-'):
                breakage_lines.append(line[1:])  # Capture broken lines
                in_change_block = True
            elif line.startswith('+'):
                repaired_lines_only.append(line[1:])  # Capture repaired lines
                in_change_block = True
            else:
                # Handle context lines
                if in_change_block:
                    unchanged_after.append(line)
                else:
                    unchanged_before.append(line)

        # Build the annotated output
        annotated_string = (
                "[<TESTCONTEXT>]\n"
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

    def compare_ast_similarity(self, ast1, ast2):
        print("Checking changeset similarity")
        dump1 = ast.dump(ast1, annotate_fields=False)
        dump2 = ast.dump(ast2, annotate_fields=False)
        return SequenceMatcher(None, dump1, dump2).ratio()

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

    def extract_covered_source_coverage(self, rel_path, test_method, broken_hash, repaired_hash):
        print("Extracting dynamic covered source code for both versions")

        # Extract the full executed source for both commits.
        broken_source = self.get_covered_source(rel_path, test_method, broken_hash)
        print("broken source ecsc 503: " + broken_source)
        repaired_source = self.get_covered_source(rel_path, test_method, repaired_hash)
        print("broken source ecsc 505: "+ repaired_source)
        if not broken_source and not repaired_source:
            print("No coverage data found for either version.")
            return ""

        executed_methods_broken = self.extract_executed_methods(broken_source, rel_path)
        executed_methods_repaired = self.extract_executed_methods(repaired_source, rel_path)
        print("executed methods in ecsc 511", executed_methods_broken)
        print("executed methods in ecsc 512", executed_methods_repaired)

        # Fallback: if executed methods are empty, use the full method source from the file.
        if not executed_methods_broken:
            fallback = self.extract_method_code(rel_path, test_method)
            if fallback:
                executed_methods_broken = {"fallback ecsc 520" + test_method: fallback}
        if not executed_methods_repaired:
            fallback = self.extract_method_code(rel_path, test_method)
            if fallback:
                executed_methods_repaired = {"fallback ecsc 522" + test_method: fallback}

        result = []
        # Compute the union of all method keys.
        all_method_names = set(executed_methods_broken.keys()).union(executed_methods_repaired.keys())

        for method_name in sorted(all_method_names):
            broken_code = executed_methods_broken.get(method_name, '')
            print("broken code: ecsc 531 " + broken_code)
            repaired_code = executed_methods_repaired.get(method_name, '')
            print("repaired code: ecsc 533 " + repaired_code)
            diff = list(difflib.unified_diff(broken_code.splitlines(), repaired_code.splitlines(), lineterm=""))
            print("raw diff: " + " ".join(diff))
            diff = self.filter_diff_lines(diff)
            print("filtered diff: " + " ".join(diff))
            if diff:
                formatted_hunk = self.format_inline_diff(method_name, diff)
            else:
                # Even if there is no diff, output the full repaired method code wrapped in a hunk.
                formatted_hunk = f"[<HUNK>] {method_name}\n{repaired_code}\n[</HUNK>]"
            # Prepend the class header based on the method key (format: "Class.method")
            class_name = method_name.split(".")[0]
            result.append(f'class {class_name}:')
            result.append(formatted_hunk)

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
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        # Checkout the specified commit.
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

        # Ensure a .coveragerc file exists.
        coveragerc_path = repo_dir / ".coveragerc"
        if not coveragerc_path.exists():
            with open(coveragerc_path, "w", encoding="utf-8") as f:
                f.write("""[run]
    parallel = True
    branch = True
    concurrency = thread
    """)

        # Set up the environment for the coverage subprocess.
        env = os.environ.copy()
        env["COVERAGE_PROCESS_START"] = str(coveragerc_path)
        env["PYTHONPATH"] = str(repo_dir)

        # Build the test node id (using the absolute path of the test file).
        test_file_path = repo_dir / rel_path
        nodeid = f"{test_file_path.as_posix()}::{test_method}"

        # Run the test via coverage in parallel mode.
        cmd = [
            "python", "-m", "coverage", "run", "--parallel-mode", "-m", "pytest",
            "--maxfail=1", "--disable-warnings", "--quiet", nodeid
        ]
        returncode, log = run_cmd(cmd, timeout=15 * 60, cwd=str(repo_dir), env=env)
        print("pytest/coverage run returned:", returncode)
        print("Log output:", log)

        # Combine coverage data from subprocesses.
        combine_cmd = ["python", "-m", "coverage", "combine"]
        combine_return, combine_log = run_cmd(combine_cmd, timeout=15 * 60, cwd=str(repo_dir), env=env)
        print("Coverage combine returned:", combine_return)
        print("Coverage combine log:", combine_log)

        # Load the combined coverage data.
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
                    # Coverage reports lines 1-indexed.
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
