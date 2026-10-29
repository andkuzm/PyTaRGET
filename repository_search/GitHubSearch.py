import csv
import multiprocessing
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import venv
import warnings
from pathlib import Path

import requests
import time

try:
    import processed_log  # run as a script, or with repository_search/ on the path
except ImportError:
    from repository_search import processed_log  # imported as repository_search.GitHubSearch

PROCESS_TIMEOUT = 90 * 60   # 90 minutes

# Must match the header written by main_repository_miner.Main.save_case.
CSV_HEADER = ["repository_name", "annotated_code", "relative_path",
              "broken_hash", "repaired_hash", "outdated_test_log",
              "test_name", "broken_date", "repaired_date"]

# Bump this to the latest merged PR number whenever a change lands that
# affects mining/repair-detection behavior (repository_actions.py,
# main_repository_miner.py, py_parser.py, or the search logic here). It's
# baked into the output CSV/blacklist/clone-dir names below so a run's
# filenames always show exactly which fixes produced it - no need to
# remember whether `git pull` happened before starting a run, and a repo
# blacklisted by an older, buggier revision gets retried automatically
# under a new one instead of staying skipped forever.
PIPELINE_FIX_REVISION = "pr59"


class GitHubSearch:

    def __init__(self, github_token, version=None, cwd=None,
                 repository_path=None, out_path=None, workers=1, max_repositories=None):
        self.github_token = github_token
        # workers: how many repositories are processed at the same time (each in its own clone directory, venv and
        # process group). max_repositories: stop handing out repositories after this many (None = no limit).
        self.workers = max(1, int(workers))
        self.max_repositories = max_repositories
        self._lock = threading.Lock()
        self._pytest_lock = threading.Lock()
        self._pool = None
        self._free_slots = None
        self._futures = []
        self._active = set()
        self._dispatched = 0
        self._abort = False
        self._interrupted = False

        if version is not None:
            self.version = version
            self.cwd = Path(cwd) if cwd else Path.cwd()
            run_id = f"{version}_{PIPELINE_FIX_REVISION}"
            self.repository_path = str(self.cwd / f"repos_{run_id}")
            self.out_path = str(self.cwd)
            self.output_csv = self.cwd / f"annotated_cases_{run_id}.csv"
            self.processed_file = self.cwd / f"processed_repositories_{run_id}.txt"
            print(f"Pipeline fix revision: {PIPELINE_FIX_REVISION}")
            print(f"Output CSV:  {self.output_csv}")
            print(f"Blacklist:   {self.processed_file}")
            print(f"Clone dir:   {self.repository_path}")
        else:
            warnings.warn(
                "GitHubSearch: 'repository_path' and 'out_path' are deprecated. "
                "Use 'version' (and optionally 'cwd') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.repository_path = repository_path or ""
            self.out_path = out_path or ""
            self.output_csv = Path(out_path) / "annotated_cases.csv" if out_path else Path("annotated_cases.csv")
            self.processed_file = Path(__file__).resolve().parent / "processed_repositories.txt"

    def ensure_output_csv_initialized(self):
        """Write the output CSV with just the header row up front, before any
        searching happens, so a broken path/permission would surface
        immediately instead of silently discarding hours of search results
        later (the miner only writes the header lazily, alongside the first
        found case)."""
        if self.output_csv.exists() and self.output_csv.stat().st_size > 0:
            return
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_ALL)
            writer.writerow(CSV_HEADER)
        print(f"Initialized output CSV (header only): {self.output_csv}")

    def get_latest_commit(self, full_name):
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
        return ""

    def reinstall_pytest(self):
        cmds = [
            [sys.executable, "-m", "pip", "uninstall", "-y", "pytest"],
            [sys.executable, "-m", "pip", "install", "pytest"]
        ]
        for cmd in cmds:
            proc = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"pytest reinstall command failed: {proc.stderr.strip()}")

    def run_pytest_trace(self):
        return subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            cwd=".",
            timeout=60*5
        )

    def run_pytest_check(self, last_repo):
        try:
            result = self.run_pytest_trace()
        except subprocess.TimeoutExpired:
            self.reinstall_pytest()
            result = self.run_pytest_trace()

        if result.returncode in (0, 5):
            return

        self.reinstall_pytest()
        time.sleep(2)

        retry = self.run_pytest_trace()
        if retry.returncode in (0, 5):
            return

        print(f"\npytest still failing after recovery (last repo: {last_repo}). Exiting.")
        print("Output:\n", retry.stdout)
        print("Errors:\n", retry.stderr)
        sys.exit(1)

    def _rate_limit_sleep(self, response):
        remaining = int(response.headers.get("X-RateLimit-Remaining", 10))
        reset = int(response.headers.get("X-RateLimit-Reset", 0))
        if remaining <= 2:
            wait = max(0, reset - int(time.time())) + 1
            print("Pausing for rate limit...")
            time.sleep(wait)
        else:
            time.sleep(2)

    def _github_search_request(self, headers, params, retries=5):
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
                print("Pausing for rate limit...")
                time.sleep(wait)
                continue

            print(f"GitHub API error: {response.status_code}")
            if attempt < retries - 1:
                time.sleep(10 * (2 ** attempt))

        print("Exhausted retries for GitHub API request.")
        return None

    def _search_size_range(self, query_base, size_start, size_end, headers, processed_repos):
        query = f"{query_base} size:>={size_start} size:<{size_end}"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": 1,
        }

        response = self._github_search_request(headers, params)
        if response is None:
            return

        data = response.json()
        total_count = data.get("total_count", 0)

        if total_count == 0:
            return

        if total_count >= 1000 and size_end - size_start > 1:
            mid = (size_start + size_end) // 2
            self._search_size_range(query_base, size_start, mid, headers, processed_repos)
            self._search_size_range(query_base, mid, size_end, headers, processed_repos)
            return

        page = 1
        while True:
            if page > 1:
                response = self._github_search_request(headers, {**params, "page": page})
                if response is None:
                    print(f"Warning: API gave no response at page {page}, stopping pagination.")
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
                self._dispatch(full_name)

            if 'next' not in response.links:
                break
            page += 1
            self._rate_limit_sleep(response)

    def find_and_process_repositories(self, stars=10, size_start=0, size_end=1_000_000):
        self._run_count = 0
        self._dispatched = 0
        self._abort = False
        self._interrupted = False

        self.ensure_output_csv_initialized()

        processed_repos = set()
        unfinished = []
        try:
            entries = processed_log.read_entries(self.processed_file)
            processed_repos = set(entries)  # finished and unfinished alike: the search pass must not run them again
            unfinished = [entry for entry in entries.values() if entry.in_progress]
        except Exception as e:
            print(f"Error reading {self.processed_file}: {e}")

        if processed_repos:
            print(f"Resuming: {len(processed_repos) - len(unfinished)} repositories already processed, "
                  f"{len(unfinished)} unfinished.")
        else:
            print("Starting fresh (no previously processed repositories found).")

        headers = {}
        if self.github_token:
            headers['Authorization'] = f'Bearer {self.github_token}'

        license_filters = [
            f"license:mit language:python stars:>={stars}",
            f"license:apache-2.0 language:python stars:>={stars}",
            f"license:Unlicense language:python stars:>={stars}",
            f"license:bsd-2-clause language:python stars:>={stars}",
            f"license:bsd-3-clause language:python stars:>={stars}",
            f"license:isc language:python stars:>={stars}",
            f"license:lgpl-2.1 language:python stars:>={stars}",
            f"license:gpl-2.0 language:python stars:>={stars}",
            f"license:gpl-3.0 language:python stars:>={stars}",
        ]

        self._start_pool()
        try:
            # Repositories that were interrupted last time continue first, from the last completely processed commit.
            for entry in unfinished:
                print(f"Continuing {entry.name} after {entry.marker or 'its start'}")
                self._dispatch(entry.name)

            for query_base in license_filters:
                self._search_size_range(query_base, size_start, size_end, headers, processed_repos)
        except StopSearch as stop:
            print(f"Search stopped: {stop}")
        except KeyboardInterrupt:
            print("Interrupted: stopping the workers, unfinished repositories continue next time.")
            self._stop_workers()
            raise
        finally:
            self._finish_pool()

        found = 0
        try:
            with self.output_csv.open(encoding="utf-8") as f:
                found = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

        print(f"Done. Processed {self._run_count} repositories, found {found} test cases.")

    def _start_pool(self):
        if self.workers > 1:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="repo-worker")
            self._free_slots = queue.Queue()
            for slot in range(self.workers):
                self._free_slots.put(slot)

    def _finish_pool(self):
        """Wait for the repositories that are still being processed."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for future in self._futures:
            future.result()  # a worker never raises (see _work), this only surfaces a bug
        self._futures = []

    def _stop_workers(self):
        """Kill what the workers started (the miners and the tests under them), e.g. after Ctrl-C."""
        self._interrupted = True  # the workers must not treat the kills below as failed attempts
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        for process in list(self._active):
            _kill_process_group(process)
            process.terminate()

    def _dispatch(self, full_name):
        """Hand a repository to a free worker, waiting while all of them are busy. Without workers it runs here."""
        if self._abort:
            raise StopSearch("a worker found the environment broken")
        if self.max_repositories is not None and self._dispatched >= self.max_repositories:
            raise StopSearch(f"limit of {self.max_repositories} repositories reached")
        self._dispatched += 1
        if self._pool is None:
            print(f"Processing {full_name}")
            self._work(full_name, None)
            return
        slot = self._free_slots.get()  # waits while every worker is busy, so the message below means "started"
        print(f"Processing {full_name} (worker {slot})")
        self._futures.append(self._pool.submit(self._work, full_name, slot))

    def _work(self, full_name, slot):
        try:
            if slot is None:
                self._process_and_record(full_name)
            else:
                # one clone directory per worker: the miner names the clone after the last part of the repository
                # name, so two repositories with the same name under different owners would collide otherwise
                path = Path(self.repository_path) / f"slot{slot}"
                path.mkdir(parents=True, exist_ok=True)
                self._process_and_record(full_name, str(path))
            with self._lock:
                self._run_count = getattr(self, "_run_count", 0) + 1
            with self._pytest_lock:
                self.run_pytest_check(full_name)
        except SystemExit:  # run_pytest_check gives up when the shared environment is broken
            self._abort = True
        except Exception as e:
            print(f"{full_name}: worker error: {e}")
        finally:
            if slot is not None:
                self._free_slots.put(slot)

    def _process_and_record(self, full_name, repository_path=None):
        """
        Run one repository and settle its line in the processed file. The miner marks a repository done itself when
        it finishes. If it was killed (time limit) or crashed while searching, its line still says "in progress" with
        the last completely processed commit and the next run continues from there, but only if this attempt got
        further than the one before; otherwise it is given up so it cannot be retried forever.
        """
        before = processed_log.resume_state(self.processed_file, full_name)
        start_marker = before.marker if before is not None else ""

        finished = (self.process_repository_with_timeout(full_name) if repository_path is None
                    else self.process_repository_with_timeout(full_name, repository_path))
        if self._interrupted:
            # stopped by the user, not a failed attempt: the line stays as the miner left it (in progress, or absent
            # if it had not got that far) and the repository is picked up again next time
            print(f"{full_name}: interrupted, it continues next time")
            return
        if not finished:
            print(f"Failed or timed out: {full_name}")

        after = processed_log.read_entries(self.processed_file).get(full_name)
        if after is None:  # the miner died before it wrote anything
            processed_log.mark_done(self.processed_file, full_name, self.get_latest_commit(full_name) or "")
        elif after.in_progress:
            if after.marker != start_marker:
                print(f"{full_name}: stopped after commit {after.marker[:10]}, the next run resumes from there")
            else:
                print(f"{full_name}: no progress in this attempt, giving up")
                processed_log.mark_done(self.processed_file, full_name, after.head)

    def process_repository_with_timeout(self, full_name, repository_path=None):
        venv_dir = tempfile.mkdtemp(prefix=f"repo_venv_{full_name.replace('/', '_')}_")
        create_virtualenv(venv_dir)

        p = multiprocessing.Process(
            target=_worker_entry,
            args=(full_name, repository_path or self.repository_path, str(self.output_csv), venv_dir,
                  str(self.processed_file))
        )

        start = time.time()
        p.start()
        with self._lock:
            self._active.add(p)
        p.join(PROCESS_TIMEOUT)

        timed_out = False

        if p.is_alive():
            # terminate() only reaches this wrapper; the miner and the tests below it would run on as orphans
            _kill_process_group(p)
            p.terminate()
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join()
            timed_out = True
        _kill_process_group(p)  # whatever the miner left behind
        with self._lock:
            self._active.discard(p)

        try:
            shutil.rmtree(venv_dir)
        except Exception:
            pass

        if timed_out:
            return False

        return p.exitcode == 0


def create_virtualenv(venv_path):
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(venv_path)

class StopSearch(Exception):
    """Raised to leave the search loops early (repository limit reached, environment broken)."""


def _kill_process_group(process):
    """SIGKILL the process group a worker leads (see _worker_entry); a no-op where that is not possible."""
    pid = getattr(process, "pid", None)
    if pid is None or not hasattr(os, "killpg"):
        return
    try:
        os.killpg(pid, getattr(signal, "SIGKILL", 9))
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _worker_entry(*args):
    """Entry point of the process that runs one repository: it leads its own process group, so that a time limit or
    Ctrl-C can kill the miner and the test runs below it together."""
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    run_processor_in_venv(*args)


def run_processor_in_venv(full_name, repository_path, output_csv, venv_path, processed_file=None):

    python_bin = os.path.join(
        venv_path,
        "Scripts" if os.name == "nt" else "bin",
        "python"
    )

    env = os.environ.copy()
    root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(root)

    miner = root / "repository_search" / "main_repository_miner.py"

    subprocess.run(
        [python_bin, "-m", "pip", "install", "-q", "--disable-pip-version-check", "pytest", "coverage"],
        timeout=600,
        check=False,
        capture_output=True,
    )

    try:
        subprocess.check_call([
            python_bin, "-u", str(miner),
            full_name, repository_path, output_csv, *([processed_file] if processed_file else [])
        ], timeout=PROCESS_TIMEOUT + 60, env=env)
    except subprocess.TimeoutExpired:
        raise
    except subprocess.CalledProcessError:
        raise


if __name__ == "__main__":
    cwd = Path.cwd()

    print(f"Pipeline fix revision: {PIPELINE_FIX_REVISION}")
    print()

    # --- Discover existing versioned runs ---
    existing_csvs = sorted(cwd.glob("annotated_cases_*.csv"))
    legacy_csv = cwd / "annotated_cases.csv"

    if existing_csvs:
        print("Existing versioned runs found in this directory:")
        for f in existing_csvs:
            version_label = f.stem.replace("annotated_cases_", "")
            try:
                with f.open(encoding="utf-8") as fh:
                    row_count = max(0, sum(1 for _ in fh) - 1)
            except Exception:
                row_count = "?"
            blacklist_path = cwd / f"processed_repositories_{version_label}.txt"
            if blacklist_path.exists():
                try:
                    processed_count = sum(1 for _ in blacklist_path.open(encoding="utf-8"))
                except Exception:
                    processed_count = "?"
            else:
                processed_count = 0
            # The filename always ends in the revision that produced it
            # (see PIPELINE_FIX_REVISION), so a stale checkout is visible
            # right here instead of something you have to remember to check.
            is_current = version_label.endswith(f"_{PIPELINE_FIX_REVISION}")
            tag = "current revision" if is_current else "OLDER revision - a fresh run will not resume this one"
            print(f"  [{version_label}]  {row_count} test cases, {processed_count} repos processed  ({tag})")
        print()

    if legacy_csv.exists():
        print("Note: a legacy 'annotated_cases.csv' (no version suffix) exists here.")
        print("      It will NOT be included in or merged with any versioned run.")
        print()

    # --- GitHub token ---
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()  # set it to skip the prompt
    if not github_token:
        while True:
            github_token = input("GitHub token (leave blank for unauthenticated): ").strip()
            if not github_token:
                print("Warning: running without a token — rate limits will be very restrictive (10 req/min).")
                confirm = input("Continue without token? [y/N]: ").strip().lower()
                if confirm == "y":
                    break
                continue
            placeholders = ("your_token", "token_here", "xxx", "<token>", "paste")
            if len(github_token) < 10 or any(p in github_token.lower() for p in placeholders):
                print(f"Warning: '{github_token}' looks like a placeholder or is very short.")
                confirm = input("Use it anyway? [y/N]: ").strip().lower()
                if confirm != "y":
                    continue
            break

    # --- Version name ---
    version = os.environ.get("PYTARGET_VERSION", "").strip()  # set it to skip the prompt
    if not version:
        while True:
            version = input("Version name (e.g. v1, experiment2, java-run): ").strip()
            if not version:
                print("Version name cannot be empty.")
                continue
            if not version.replace("-", "").replace("_", "").isalnum():
                print(f"Warning: '{version}' contains special characters. Only letters, digits, hyphens, and underscores are recommended.")
                confirm = input("Use it anyway? [y/N]: ").strip().lower()
                if confirm != "y":
                    continue
            break

    # --- Show paths ---
    # Constructed once here (instead of re-deriving the filename formula) so
    # the paths shown below always match, byte for byte, what the run
    # actually uses - the revision suffix is set in exactly one place.
    print()
    # Optional, from the environment: PYTARGET_WORKERS (repositories processed at the same time),
    # PYTARGET_MAX_REPOS (stop after this many repositories), PYTARGET_STARS, PYTARGET_SIZE_START and
    # PYTARGET_SIZE_END (the repository size window of the search, in KB).
    searcher = GitHubSearch(
        github_token=github_token,
        version=version,
        cwd=str(cwd),
        workers=int(os.environ.get("PYTARGET_WORKERS", "1")),
        max_repositories=int(os.environ["PYTARGET_MAX_REPOS"]) if os.environ.get("PYTARGET_MAX_REPOS") else None,
    )
    output_csv_path = searcher.output_csv
    blacklist_path = searcher.processed_file
    clone_dir = Path(searcher.repository_path)

    if output_csv_path.exists():
        try:
            with output_csv_path.open(encoding="utf-8") as fh:
                existing_cases = max(0, sum(1 for _ in fh) - 1)
        except Exception:
            existing_cases = "?"
        print(f"Continuing existing run: {existing_cases} test cases already saved.")
    if blacklist_path.exists():
        try:
            processed_count = sum(1 for _ in blacklist_path.open(encoding="utf-8"))
        except Exception:
            processed_count = "?"
        print(f"Resuming from blacklist: {processed_count} repositories already processed.")
    print()

    clone_dir.mkdir(parents=True, exist_ok=True)

    searcher.find_and_process_repositories(
        stars=int(os.environ.get("PYTARGET_STARS", "10")),
        size_start=int(os.environ.get("PYTARGET_SIZE_START", "0")),
        size_end=int(os.environ.get("PYTARGET_SIZE_END", "1000000")),
    )
