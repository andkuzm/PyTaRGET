import subprocess
import sys
from pathlib import Path

import requests
import time
import main_repository_miner


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

    def run_pytest_check(self, last_repo):
        """Runs pytest --trace-config to check for module issues after processing a repository."""
        print("Running pytest --trace-config...")
        result = subprocess.run(["pytest", "--trace-config"], capture_output=True, text=True, cwd="dummy_folder")
        print("code: ", result.returncode)
        if result.returncode not in (0, 5):
            print(f"\nPytest encountered an error after processing {last_repo}.")
            print("Stopping execution due to pytest --trace-config failure.")
            print("Pytest Output:\n", result.stdout)
            print("Pytest Errors:\n", result.stderr)
            sys.exit(1)
        else:
            print("pytest module passed.")

    def find_and_process_repositories(self):
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

        # Define the search queries. GitHub supports "license:none" to find repositories with no license.
        # For public non-commercial licenses, you might search for a known license identifier (e.g., "cc-by-nc").
        queries = [
            "license:mit language:python stars:>50 size:>=1000 size:<10000", #size:>=1000 size:<10000
            "license:apache-2.0 language:python stars:>50 size:>=1000 size:<10000", # size:<1000
            "license:Unlicense language:python stars:>50 size:>=1000 size:<10000"
        ]  # TODO: look some more about non-licensed repos
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
                    processor = main_repository_miner.Main(full_name, self.repository_path, self.out_path)
                    processor.process_repository()

                    # Retrieve the latest commit hash for reproduction purposes.
                    latest_commit = self.get_latest_commit(full_name)

                    with self.processed_file.open("a", encoding="utf-8") as f:
                        # Write in the format: "full_name|commit_hash"
                        f.write(f"{full_name}|{latest_commit}\n")

                    # Also update the in-memory set.
                    processed_repos.add(full_name)

                    self.run_pytest_check(full_name)
                # Check for pagination; GitHub API provides link headers.
                if 'next' not in response.links:
                    print("no 'next' link found")
                    break
                page += 1
                time.sleep(2)  # be respectful of rate limits

        print("Finished processing repositories.")


searcher = GitHubSearch(
    github_token="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",
    repository_path="C:\\Users\\kandr\\PycharmProjects\\repos",
    out_path="C:\\Users\\kandr\\Desktop"
)
searcher.find_and_process_repositories()