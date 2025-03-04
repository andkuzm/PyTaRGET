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

    def find_and_process_repositories(self):
        """
        Searches GitHub for repositories that are either unlicensed or have a public non-commercial license,
        extracts their full names (in "username/repository" format), and processes each repository.
        """

        processed_repos = set()
        if self.processed_file.exists():
            try:
                with self.processed_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        processed_repos.add(line.strip())
            except Exception as e:
                print(f"Error reading {self.processed_file}: {e}")

        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'

        # Define the search queries. GitHub supports "license:none" to find repositories with no license.
        # For public non-commercial licenses, you might search for a known license identifier (e.g., "cc-by-nc").
        queries = ["mit language:python", "apache-2.0 language:python", "gpl-3.0 language:python"] #TODO look some more about non-licensed repos
        base_url = "https://api.github.com/search/repositories"

        for query in queries:
            page = 1
            while True:
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
                    processed_repos.add(full_name)
                    print(f"Processing repository: {full_name}")
                    processor = main_repository_miner.Main(full_name, self.repository_path, self.out_path)
                    processor.process_repository()
                    with self.processed_file.open("a", encoding="utf-8") as f:
                        f.write(full_name + "\n")
                    # Also update the in-memory set.
                    processed_repos.add(full_name)
                # Check for pagination; GitHub API provides link headers.
                if 'next' not in response.links:
                    break
                page += 1
                time.sleep(1)  # be respectful of rate limits

        print("Finished processing repositories.")
searcher = GitHubSearch(github_token="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", repository_path="C:\\Users\\kandr\\PycharmProjects\\repos", out_path="C:\\Users\\kandr\\Desktop")
searcher.find_and_process_repositories()