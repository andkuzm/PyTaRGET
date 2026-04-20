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


class GitHubSearch:

    def __init__(self, github_token, repository_path, out_path):
        self.repository_path = repository_path
        self.github_token = github_token
        self.out_path = out_path
        self.processed_file = Path(__file__).resolve().parent / "processed_repositories.txt"
        print("Will write annotated test cases into:" + self.out_path + "/annotated_cases.csv")
        print("Will read already processed repos from:" + str(self.processed_file))

    def get_latest_commit(self, full_name):
        """
        Retrieves the latest commit hash for the repository using the GitHub API.
        """
        commits_url = f"https://api.github.com/repos/{full_name}/commits"
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'Bearer {self.github_token}'
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
            [sys.executable, "-m", "pip", "install", "pytest"]
        ]

        for cmd in cmds:
            print("Running:", " ".join(cmd))
            proc = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
            if proc.returncode != 0:
                print("Command failed:", proc.stderr)

        print("pytest reset complete.")

    def run_pytest_trace(self):
        """
        Runs pytest trace check, with timeout to avoid hangs.
        """
        return subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            cwd=".",
            timeout=60*5
        )

    def run_pytest_check(self, last_repo):
        """Validates pytest works, auto-recovers once on failure."""

        print("Running pytest --version")

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

    def _rate_limit_sleep(self, response):
        """Sleeps based on X-RateLimit headers to stay within GitHub's search rate limit."""
        remaining = int(response.headers.get("X-RateLimit-Remaining", 10))
        reset = int(response.headers.get("X-RateLimit-Reset", 0))
        if remaining <= 2:
            wait = max(0, reset - int(time.time())) + 1
            print(f"Rate limit low ({remaining} remaining). Sleeping {wait}s until reset...")
            time.sleep(wait)
        else:
            time.sleep(2)

    def _github_search_request(self, headers, params, retries=5):
        """
        Makes a GitHub Search API request with rate-limit-aware retry logic.
        On 429/403 rate-limit responses, sleeps until the reset time and retries.
        On 5xx errors, retries with exponential backoff.
        Returns the Response object on success, or None after exhausting retries.
        """
        base_url = "https://api.github.com/search/repositories"
        for attempt in range(retries):
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code == 200:
                return response

            if response.status_code in (403, 429):
                retry_after = response.headers.get("Retry-After")
                reset = response.headers.get("X-RateLimit-Reset")
                if retry_after:
                    wait = int(retry_after)
                elif reset:
                    wait = max(0, int(reset) - int(time.time())) + 1
                else:
                    wait = 60 * (2 ** attempt)
                print(f"Rate limited ({response.status_code}). Waiting {wait}s (attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
                continue

            print(f"GitHub API error: {response.status_code} {response.text}")
            if attempt < retries - 1:
                backoff = 10 * (2 ** attempt)
                print(f"Retrying in {backoff}s...")
                time.sleep(backoff)

        print("Exhausted retries for GitHub API request.")
        return None

    def _search_size_range(self, query_base, size_start, size_end, headers, processed_repos):
        """
        Recursively searches repositories in the size range [size_start, size_end) KB.

        GitHub's Search API caps results at 1,000 per query. When total_count hits that
        cap, this method splits the range in half and recurses, ensuring no repos are
        missed due to truncation.
        """
        query = f"{query_base} size:>={size_start} size:<{size_end}"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": 1,
        }

        print(f"Checking size range [{size_start}, {size_end}) KB ...")
        response = self._github_search_request(headers, params)
        if response is None:
            return

        data = response.json()
        total_count = data.get("total_count", 0)

        if total_count == 0:
            return

        # GitHub caps returned results at 1000; subdivide to avoid silently missing repos
        if total_count >= 1000 and size_end - size_start > 1:
            mid = (size_start + size_end) // 2
            print(f"  total_count={total_count} >= 1000; splitting into [{size_start},{mid}) and [{mid},{size_end})")
            self._search_size_range(query_base, size_start, mid, headers, processed_repos)
            self._search_size_range(query_base, mid, size_end, headers, processed_repos)
            return

        # Process all pages in this range
        print(f"  {total_count} repos in [{size_start}, {size_end}) KB — paginating...")
        page = 1
        while True:
            if page > 1:
                response = self._github_search_request(headers, {**params, "page": page})
                if response is None:
                    print(
                        f"Warning: GitHub API returned no response at page {page} of size range "
                        f"[{size_start}, {size_end}). Successfully retrieved {page - 1} page(s) before failure."
                    )
                    break
                data = response.json()

            items = data.get("items", [])
            if not items:
                break

            for repo in items:
                full_name = repo.get("full_name")
                if full_name in processed_repos:
                    continue
                processed_repos.add(full_name)
                print(f"Processing repository: {full_name}")
                success = self.process_repository_with_timeout(full_name)
                if not success:
                    print(f"Repository {full_name} failed or timed out. Marking as processed and continuing.")

                latest_commit = self.get_latest_commit(full_name)
                with self.processed_file.open("a", encoding="utf-8") as f:
                    f.write(f"{full_name}|{latest_commit}\n")

                self.run_pytest_check(full_name)

            if 'next' not in response.links:
                break
            page += 1
            self._rate_limit_sleep(response)

    def find_and_process_repositories(self, stars=50, size_start=0, size_end=1_000_000):
        """
        Searches GitHub for Python repositories with open-source licenses and processes each.

        Uses adaptive size-range splitting to work around the GitHub Search API's hard
        1,000-result cap per query: when a range returns >= 1,000 results it is split in
        half recursively, so no repos are silently skipped.

        After processing, each repository's full name and latest commit hash are appended
        to the processed file so interrupted runs can resume without reprocessing.
        """
        processed_repos = set()
        if self.processed_file.exists():
            try:
                with self.processed_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        repo_full_name = line.split("|")[0].strip()
                        processed_repos.add(repo_full_name)
            except Exception as e:
                print(f"Error reading {self.processed_file}: {e}")

        headers = {}
        if self.github_token:
            headers['Authorization'] = f'Bearer {self.github_token}'

        license_filters = [
            f"license:mit language:python stars:>={stars}",
            f"license:apache-2.0 language:python stars:>={stars}",
            f"license:Unlicense language:python stars:>={stars}",
        ]

        for query_base in license_filters:
            self._search_size_range(query_base, size_start, size_end, headers, processed_repos)

        print("Finished processing repositories.")

    def process_repository_with_timeout(self, full_name):
        """
        Runs the repository miner in an isolated virtualenv with a hard timeout.
        Returns True if processing finished, False if timed out or crashed.
        """

        venv_dir = tempfile.mkdtemp(prefix=f"repo_venv_{full_name.replace('/', '_')}_")
        create_virtualenv(venv_dir)

        p = multiprocessing.Process(
            target=run_processor_in_venv,
            args=(full_name, self.repository_path, self.out_path, venv_dir)
        )

        start = time.time()
        p.start()
        p.join(PROCESS_TIMEOUT)

        timed_out = False

        if p.is_alive():
            print(f"Timeout while processing {full_name}. Killing process.")
            p.terminate()
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join()
            timed_out = True

        duration = time.time() - start
        print(f"Finished {full_name} in {int(duration)} seconds.")

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

    python_bin = os.path.join(
        venv_path,
        "Scripts" if os.name == "nt" else "bin",
        "python"
    )

    env = os.environ.copy()
    root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(root)

    miner = root / "repository_search" / "main_repository_miner.py"

    subprocess.run([python_bin, "-m", "pip", "install", "pytest", "coverage"], timeout=600, check=False)

    try:
        subprocess.check_call([
            python_bin, "-u", str(miner),
            full_name, repository_path, out_path
        ], timeout=PROCESS_TIMEOUT + 60, env=env)
    except subprocess.TimeoutExpired:
        print(f"Subprocess timed out while processing {full_name}.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed for {full_name} with exit code {e.returncode}.")
        raise


if __name__ == "__main__":
    searcher = GitHubSearch(
        github_token="",
        repository_path="",
        out_path=""
    )
    searcher.find_and_process_repositories(size_start=0, size_end=1_000_000)
