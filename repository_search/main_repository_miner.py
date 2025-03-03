import csv
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import repository_actions

class Main:
    def __init__(self, repository_name, repository_path, out_path):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.out_path = out_path

    def get_repository_name(self):
        return self.repository_name

    def get_repository_path(self):
        return self.repository_path

    def process_repository(self):
        print("processing repository: {}".format(self.repository_name))
        # Step 1: Clone the repository (and set up appropriate branch/worktree)
        repository = repository_actions.RepositoryActions(self.repository_name, self.repository_path)
        repository.clone_repository_last()

        # Step 2: Check if the repository contains tests
        # repository_path = Path(self.repository_path) / self.repository_name.split("/")[-1]
        if repository.has_tests():
            # Step 3: Check if there are broken-to-repaired test cases in the repo
            if repository.has_broken_to_repaired_test():
                # Extract commit hashes or change pairs for tests that were broken and then fixed
                broken_to_repaired_list = repository.extract_broken_to_repaired_list()

                # Process each broken-to-repaired instance
                for repaired_test in broken_to_repaired_list:
                    # Execute the test on the buggy version to verify failure and then extract the failing test and related source code
                    annotated_code = repository.extract_and_annotate_code(repaired_test)
                    # Save the processed data into the database, along with metadata like commit hashes and timestamps
                    self.save_case(self.repository_name, annotated_code, repaired_test.broken, repaired_test.repaired)
            else:
                print("No broken-to-repaired test cases found.")
        else:
            print("Repository does not contain tests.")

        def handle_remove_readonly(func, path, exc_info):
            # Change the permissions and then call the removal function again
            os.chmod(path, stat.S_IWRITE)
            func(path)

        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        shutil.rmtree(dest_dir, onerror=handle_remove_readonly)
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", self.repository_name.split("/")[-1]]
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)

    def save_case(self, repository_name, annotated_code, broken_hash, repaired_hash):
        """
        Saves the annotated code along with the repository name, timestamp,
        broken commit hash, and repaired commit hash into a CSV file.
        The CSV uses a pipe ('|') as the delimiter to avoid issues with commas in the annotated code.

        Parameters:
            repository_name (str): Name of the repository from which the code is taken.
            annotated_code (str): The fully annotated code string.
            broken_hash (str): The commit hash where the test was broken.
            repaired_hash (str): The commit hash where the test was fixed.
        """
        # Define the output file path; adjust the path as needed.
        output_file = Path(self.out_path) / "annotated_cases.csv" #TODO check if out path correct

        # Determine if the file already exists
        file_exists = output_file.exists()

        # Open the file in append mode with UTF-8 encoding.
        with output_file.open("a", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            # If the file does not exist, write a header row.
            if not file_exists:
                writer.writerow(["repository_name", "annotated_code", "broken_hash", "repaired_hash"])
            writer.writerow([repository_name, annotated_code, broken_hash, repaired_hash])

        print(f"Saved annotated case for repository '{repository_name}' to {output_file}")


