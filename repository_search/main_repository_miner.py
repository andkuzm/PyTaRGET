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

    def process_repository(self):
        print("Processing repository: {}".format(self.repository_name))
        repository = repository_actions.RepositoryActions(self.repository_name, self.repository_path)
        repository.clone_repository_last()

        if repository.has_tests():
            repaired_cases = repository.find_repaired_test_cases()
            if repaired_cases:
                for repaired_test in repaired_cases:
                    annotated_code = repository.extract_and_annotate_code(repaired_test)
                    if annotated_code=="":
                        continue
                    self.save_case(self.repository_name, annotated_code, repaired_test.rel_path, repaired_test.broken, repaired_test.repaired, f"[<TESTLOG>]\n{repaired_test.log}\n[</TESTLOG>]\n")
            else:
                print("No repaired test cases found.")
        else:
            print("Repository does not contain tests.")

        # Cleanup: remove the cloned repository folder and uninstall the package.
        def handle_remove_readonly(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        shutil.rmtree(dest_dir, onerror=handle_remove_readonly)
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", self.repository_name.split("/")[-1]],
                           capture_output=True, text=True, check=True, env=os.environ)
        except subprocess.CalledProcessError as e:
            print(f"Failed to uninstall {self.repository_name.split('/')[-1]}: {e}")

    def save_case(self, repository_name, annotated_code, relative_path, broken_hash, repaired_hash, log):
        output_file = Path(self.out_path) / "annotated_cases.csv"
        file_exists = output_file.exists()

        with output_file.open("a", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            if not file_exists:
                writer.writerow(["repository_name", "annotated_code", "relative_path", "broken_hash", "repaired_hash", "outdated_test_log"])
            writer.writerow([repository_name, annotated_code, relative_path, broken_hash, repaired_hash, log])

        print(f"Saved annotated case for repository '{repository_name}' to {output_file}")