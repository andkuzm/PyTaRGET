import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import requests
import time

PROCESS_TIMEOUT = 90 * 60   # 90 minutes


# def run_processor(full_name, repository_path, out_path):
#     processor = main_repository_miner.Main(full_name, repository_path, out_path)
#     processor.process_repository()

class GitHubSearch:

    def __init__(self, github_token, repository_path, out_path):
        self.repository_path = repository_path
        self.github_token = github_token
        self.out_path = out_path
        self.processed_file = Path("processed_repositories.txt")

    def get_latest_commit(self, full_name):
        """
        Retrieves the latest commit hash for the repository using the GitHub API.
        """
        commits_url = f"https://api.github.com/repos/{full_name}/commits"
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        params = {"per_page": 1}
        response = requests.get(commits_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and data:
                return data[0].get("sha", "")
        else:
            print(f"Error fetching latest commit for {full_name}: {response.status_code} {response.text}")
        return ""

    def reinstall_pytest(self):
        """
        Removes pytest + plugins and reinstalls a clean version.
        """
        print("Resetting pytest installation...")

        cmds = [
            [sys.executable, "-m", "pip", "uninstall", "-y", "pytest"],
            #[sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            [sys.executable, "-m", "pip", "install", "pytest"]
        ]

        for cmd in cmds:
            print("Running:", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print("Command failed:", proc.stderr)

        print("pytest reset complete.")

    def run_pytest_trace(self):
        """
        Runs pytest trace check, with timeout to avoid hangs.
        """
        return subprocess.run(
            [sys.executable, "-m", "pytest", "--trace-config"],
            capture_output=True,
            text=True,
            cwd=".",
            timeout=60*5
        )

    def run_pytest_check(self, last_repo):
        """Validates pytest works, auto-recovers once on failure."""

        print("Running pytest --trace-config...")

        try:
            result = self.run_pytest_trace()
        except subprocess.TimeoutExpired:
            print("pytest timed out — forcing reinstall + retry.")
            self.reinstall_pytest()
            result = self.run_pytest_trace()

        print("Exit code:", result.returncode)

        if result.returncode in (0, 5):
            print("pytest OK.")
            return

        # ---- First failure ----
        print(f"\npytest failed after processing {last_repo}. Attempting automatic recovery...")
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)

        # Reset environment
        self.reinstall_pytest()

        time.sleep(2)

        # ---- Retry ----
        print("Re-running pytest...")
        retry = self.run_pytest_trace()

        if retry.returncode in (0, 5):
            print("pytest OK after recovery.")
            return

        # ---- Final failure ----
        print("\npytest still failing AFTER recovery. This is treated as a fatal error.")
        print("Output:\n", retry.stdout)
        print("Errors:\n", retry.stderr)
        sys.exit(1)

    def find_and_process_repositories(self, stars=50, size_start=1000, size_end=10000):
        """
        Searches GitHub for repositories that are either unlicensed or have a public non-commercial license,
        extracts their full names (in "username/repository" format), and processes each repository.
        After processing, the repository's full name and its latest commit hash are appended to the processed file.
        """
        processed_repos = set()
        if self.processed_file.exists():
            try:
                with self.processed_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        # Split on '|' delimiter if available, so that both plain full names
                        # and "full_name|commit" lines are correctly recognized.
                        repo_full_name = line.split("|")[0].strip()
                        processed_repos.add(repo_full_name)
            except Exception as e:
                print(f"Error reading {self.processed_file}: {e}")

        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'

        queries = [
            f"license:mit language:python stars:>={stars} size:>={size_start} size:<{size_end}", #size:>=1000 size:<10000
            f"license:apache-2.0 language:python stars:>={stars} size:>={size_start} size:<{size_end}", # size:<1000
            f"license:Unlicense language:python stars:>={stars} size:>={size_start} size:<{size_end}"
        ]
        base_url = "https://api.github.com/search/repositories"

        for query in queries:
            page = 1
            while True:
                print("page:", page, "running query:", query)
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 30,
                    "page": page
                }
                response = requests.get(base_url, headers=headers, params=params)
                if response.status_code != 200:
                    print(f"GitHub API error: {response.status_code} {response.text}")
                    break
                data = response.json()
                items = data.get("items", [])
                if not items:
                    break
                for repo in items:
                    full_name = repo.get("full_name")  # format: "username/repository_name"
                    if full_name in processed_repos:
                        continue
                    print(f"Processing repository: {full_name}")
                    success = self.process_repository_with_timeout(full_name)

                    if not success:
                        print(f"Repository {full_name} failed or timed out. Marking as processed and continuing.")

                    latest_commit = self.get_latest_commit(full_name)

                    with self.processed_file.open("a", encoding="utf-8") as f:
                        f.write(f"{full_name}|{latest_commit}\n")

                    # Also update the in-memory set.
                    processed_repos.add(full_name)

                    self.run_pytest_check(full_name)
                if 'next' not in response.links:
                    print("no 'next' link found")
                    break
                page += 1
                time.sleep(2)  # be respectful of rate limits

        print("Finished processing repositories.")

    def process_repository_with_timeout(self, full_name):
        """
        Runs the repository miner in an isolated virtualenv with a hard timeout.
        Returns True if processing finished, False if timed out or crashed.
        """

        # temp venv folder
        venv_dir = tempfile.mkdtemp(prefix=f"repo_venv_{full_name.replace('/', '_')}_")
        create_virtualenv(venv_dir)

        def target():
            run_processor_in_venv(full_name, self.repository_path, self.out_path, venv_dir)

        p = multiprocessing.Process(target=target)

        start = time.time()
        p.start()
        p.join(PROCESS_TIMEOUT)

        timed_out = False

        if p.is_alive():
            print(f"Timeout while processing {full_name}. Killing process.")
            p.terminate()
            p.join()
            timed_out = True

        duration = time.time() - start
        print(f"Finished {full_name} in {int(duration)} seconds.")

        # always clean up venv
        try:
            shutil.rmtree(venv_dir)
        except Exception as e:
            print(f"Warning: failed to delete venv for {full_name}: {e}")

        if timed_out:
            return False

        return p.exitcode == 0

def create_virtualenv(venv_path):
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(venv_path)

def run_processor_in_venv(full_name, repository_path, out_path, venv_path):
    python_bin = (
        os.path.join(venv_path, "bin", "python")
        if os.name != "nt"
        else os.path.join(venv_path, "Scripts", "python.exe")
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())

    # Upgrade tooling
    subprocess.run(
        [python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        check=False,
    )

    # Install what repository_actions needs (single call, fail if broken)
    subprocess.check_call(
        [python_bin, "-m", "pip", "install", "pytest", "coverage"],
        env=env,
    )

    # Run miner
    subprocess.check_call(
        [
            python_bin,
            "-m",
            "main_repository_miner",
            full_name,
            repository_path,
            out_path,
        ],
        env=env,
    )

def clone_environment_to_venv(python_bin):
    # 1. freeze current environment
    reqs = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True
    )

    # 2. install into venv
    subprocess.run(
        [python_bin, "-m", "pip", "install", "-r", "-"],
        input=reqs,
        text=True,
        check=False
    )

if __name__ == "__main__":
    searcher = GitHubSearch(
        github_token="",
        repository_path="",
        out_path=""
    )
    for i in range(0, 1000000, 100):
        searcher.find_and_process_repositories(size_start=i, size_end=i+100)