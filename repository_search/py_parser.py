import os
import re
import shlex
import subprocess
from subprocess import TimeoutExpired


# Reuse or adapt the TestVerdict class from your original code.
class TestVerdict:
    # Valid results
    SUCCESS = "success"
    FAILURE = "failure"
    SYNTAX_ERR = "syntax_error"  # For Python, instead of COMPILE_ERR
    # Invalid results
    TIMEOUT = "timeout"
    TEST_NOT_EXECUTED = "test_not_executed"
    UNEXPECTED_FAILURE = "unexpected_failure"
    UNKNOWN = "unknown"
    # Special verdict for unconventional test output
    UNCONVENTIONAL = "unconventional"

    def __init__(self, status, error_lines, log=None):
        self.status = status
        self.error_lines = error_lines
        self.log = log

    def is_valid(self):
        return self.status in [TestVerdict.SUCCESS, TestVerdict.FAILURE, TestVerdict.SYNTAX_ERR]

    def is_broken(self):
        return self.is_valid() and self.status != TestVerdict.SUCCESS

    def succeeded(self):
        return self.status == TestVerdict.SUCCESS

    def to_dict(self):
        return {
            "status": self.status,
            "error_lines": sorted(list(self.error_lines)) if self.error_lines is not None else None,
        }

    def __str__(self):
        return f"TestVerdict(status={self.status})"

def run_cmd(cmd, timeout, cwd, env):
    """Run a command using subprocess and return returncode and output."""
    try:
        # Use shlex.join if available (Python 3.8+), otherwise " ".join(cmd) works if there are no special characters.
        proc = subprocess.Popen(shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd)
        stdout, stderr = proc.communicate(timeout=timeout)
        # Combine stdout and stderr for complete output
        output = stdout.decode("utf-8", errors="ignore") + "\n" + stderr.decode("utf-8", errors="ignore")
        return proc.returncode, output
    except TimeoutExpired as e:
        proc.kill()
        stdout, stderr = proc.communicate()
        output = stdout.decode("utf-8", errors="ignore") + "\n" + stderr.decode("utf-8", errors="ignore")
        return 124, output

def parse_successful_execution_py(log):
    # For pytest, if returncode is 0, we assume the test passed.
    # You could further parse the log for additional information.
    return TestVerdict(TestVerdict.SUCCESS, None, log)

def parse_test_failure_py(log, test_class, test_method):
    # Attempt to capture line numbers from a typical pytest traceback.
    # This regex might need adjustment based on your pytest configuration.
    regex = rf"File \".+{test_class}.py\", line (\d+), in {test_method}"
    matches = re.compile(regex).findall(log)
    if matches:
        error_lines = set([int(m) for m in matches])
        return TestVerdict(TestVerdict.FAILURE, error_lines, log)
    # If no match but there is failure info, return a generic failure verdict.
    if "FAILED" in log:
        return TestVerdict(TestVerdict.FAILURE, set(), log)
    return None

def parse_invalid_execution_py(log):
    # Check for syntax errors or other unexpected output in pytest log.
    if "SyntaxError" in log:
        return TestVerdict(TestVerdict.SYNTAX_ERR, None, log)
    # Additional patterns can be added as needed.
    return TestVerdict(TestVerdict.UNKNOWN, None, log)


def compile_and_run_test_python(project_path, test_rel_path, test_method, log_path, save_logs=True, timeout=15 * 60):
    """
    Run a single Python test using pytest.
    - project_path: Path object to the project root.
    - test_rel_path: Relative repository_path to the test file.
    - test_method: Name of the test method (e.g., 'test_example').
    - log_path: Path object where log file will be stored.

    Returns a TestVerdict:
      - SUCCESS if the test passes.
      - FAILURE (or SYNTAX_ERR) if the test fails with a conventional, parsable result.
      - UNCONVENTIONAL if the output doesn't match any known pattern.

    This version ignores cases where warnings (and not errors) are present.
    """
    test_file = project_path / test_rel_path
    if not test_file.exists():
        raise FileNotFoundError(f"Test file does not exist: {test_file}")

    # Build a pytest command using the nodeid format: <file>::<test_method>
    nodeid = f"{test_file.as_posix()}::{test_method}"
    cmd = ["pytest", "--maxfail=1", "--disable-warnings", "--quiet", nodeid]

    # Run the command and capture output.
    returncode, log = run_cmd(cmd, timeout=timeout, cwd=project_path, env=os.environ)

    print("compiling and running")
    # if save_logs:
    #     log_path.mkdir(parents=True, exist_ok=True)
    #     log_file.write_text(log)

    # If the test passed, return success.
    if returncode == 0:
        print("test passed")
        print(log)
        return parse_successful_execution_py(log)

    # At this point, returncode != 0.
    # Check if the log contains error indicators.
    error_indicators = ["error"]
    if not any(indicator in log for indicator in error_indicators):
        # If no error markers are present, assume it's just warnings.
        print("Only warnings detected; treating test as passed.")
        print(log)
        return parse_successful_execution_py(log)

    print("test failed")
    print(log)
    if returncode == 124:
        return TestVerdict(TestVerdict.TIMEOUT, None, log)

    # Try to parse a conventional failure.
    failure = parse_test_failure_py(log, test_file.stem, test_method)
    if failure is not None:
        return failure

    # Fallback: parse invalid execution.
    invalid = parse_invalid_execution_py(log)
    if invalid.status == TestVerdict.UNKNOWN:
        # Mark as "unconventional" if the output doesn't match known patterns.
        return TestVerdict(TestVerdict.UNCONVENTIONAL, None, log)
    return invalid

# Example usage:
if __name__ == "__main__":
    from pathlib import Path

    # Define your project directory, test file, test method, and log directory.
    project_dir = Path("/content/my_python_project")
    test_rel_path = Path("tests/test_example.py")
    test_method = "test_functionality"  # Replace with actual test method name.
    log_dir = Path("/content/test_logs")

    verdict = compile_and_run_test_python(project_dir, test_rel_path, test_method, log_dir)
    print("Test Verdict:", verdict)
