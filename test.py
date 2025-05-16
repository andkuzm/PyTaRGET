import os
import subprocess
from pathlib import Path

import coverage


def run_cmd(cmd, timeout, cwd, env):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        return -1, str(e)


class DummyRepository:
    def __init__(self, repository_name, repository_path):
        self.repository_name = repository_name
        self.repository_path = repository_path

    def extract_covered_source_coverage(self, rel_path, test_method):
        print("Extracting dynamic covered source code")
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]
        if not repo_dir.exists():
            print(f"Warning: Repository directory {repo_dir} not found.")
            return ""

        # Ensure a .coveragerc file exists
        coveragerc_path = repo_dir / ".coveragerc"
        if not coveragerc_path.exists():
            with open(coveragerc_path, "w", encoding="utf-8") as f:
                f.write("""[run]
parallel = True
branch = True
concurrency = thread
""")

        env = os.environ.copy()
        env["COVERAGE_PROCESS_START"] = str(coveragerc_path)
        env["PYTHONPATH"] = str(repo_dir)  # Ensure modules can be discovered

        test_file_path = repo_dir / rel_path
        nodeid = f"{test_file_path.as_posix()}::{test_method}"

        # Run the test with coverage
        cmd = [
            "python", "-m", "coverage", "run", "--parallel-mode", "-m", "pytest",
            "--maxfail=1", "--disable-warnings", "--quiet", nodeid
        ]
        returncode, log = run_cmd(cmd, timeout=15 * 60, cwd=str(repo_dir), env=env)
        print("pytest/coverage run returned:", returncode)
        print("Log output:", log)

        # Combine coverage data
        combine_cmd = ["python", "-m", "coverage", "combine"]
        run_cmd(combine_cmd, timeout=15 * 60, cwd=str(repo_dir), env=env)

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


# Setup the test repository
def setup_toy_repo(base_dir):
    repository_name = "owner/toyproject"
    repo_folder = Path(base_dir) / repository_name.split("/")[-1]
    tests_folder = repo_folder / "tests"
    tests_folder.mkdir(parents=True, exist_ok=True)

    test_file = tests_folder / "test_sample.py"
    test_file.write_text(
        """
class MyClass:
    def old_method():
        print('Old Method')

def test_example():
    instance = MyClass()
    instance.old_method()
    assert True
        """,
        encoding="utf-8"
    )

    return repository_name, str(base_dir)


if __name__ == "__main__":
    toy_base = Path("toy_repo")
    toy_base.mkdir(exist_ok=True)
    repository_name, repository_path = setup_toy_repo(toy_base)

    repo = DummyRepository(repository_name, repository_path)
    covered_source = repo.extract_covered_source_coverage("", "test_example")

    print("Covered source collected:\n", covered_source)

# You can run this script and see the output, which should show the executed method! 🚀
