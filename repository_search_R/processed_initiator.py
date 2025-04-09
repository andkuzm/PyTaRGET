import os
from pathlib import Path
from wsgiref import headers

from repository_search_R.CSVHandler import CSVHandler
from repository_search_R.main_repository_miner_R import Main_r


class ProcessedInitiator:

    def __init__(self, github_token, repository_path, out_path):
        self.repository_path = repository_path
        self.github_token = github_token
        self.out_path = out_path
        self.processed_repos_path = Path(str.join(os.path.sep, os.getcwd().split(os.path.sep)[0:-1])+os.path.sep
                                         +"repository_search"+os.path.sep+"processed_repositories.txt")
        csv = CSVHandler(self.out_path)
        csv._initialize_csv()

    def find_and_process_repositories(self):
        """
        Searches GitHub for repositories that are either unlicensed or have a public non-commercial license,
        extracts their full names (in "username/repository" format), and processes each repository.
        After processing, the repository's full name and its latest commit hash are appended to the processed file.
        """
        if self.processed_repos_path.exists():
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            try:
                with self.processed_repos_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        repo_full_name = line.split("|")[0].strip()
                        repo_commit = line.split("|")[1].strip()
                        if repo_commit == "":
                            print("skipping repository:", repo_full_name)
                            continue
                        main = Main_r(repo_full_name, self.repository_path, repo_commit, self.out_path)
                        main.process_repository()

            except Exception as e:
                print(f"Error reading {self.processed_repos_path}: {e}")
            print("Finished processing repositories.")

searcher = ProcessedInitiator(
    github_token="",
    repository_path="",
    out_path=""
)
searcher.find_and_process_repositories()