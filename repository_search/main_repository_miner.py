import csv
import fcntl
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import repository_actions

class Main:
    def __init__(self, repository_name, repository_path, output_csv):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.output_csv = Path(output_csv)

    def process_repository(self):
        try:
            repository = repository_actions.RepositoryActions(self.repository_name, self.repository_path)
            repository.clone_repository_last()

            if repository.has_tests():
                repaired_cases = repository.find_repaired_test_cases()
                if repaired_cases:
                    for repaired_test in repaired_cases:
                        annotated_code = repository.extract_and_annotate_code(repaired_test)
                        if not annotated_code or annotated_code == "Error":
                            continue
                        self.save_case(self.repository_name, annotated_code, repaired_test.rel_path, repaired_test.broken, repaired_test.repaired, f"[<TESTLOG>]\n{repaired_test.log}\n[</TESTLOG>]\n")

            try:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", self.repository_name.split("/")[-1]],
                               capture_output=True, text=True, check=True, env=os.environ)
            except subprocess.CalledProcessError:
                pass
        except Exception as e:
            print(f"Repository failed: {self.repository_name}: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        shutil.rmtree(dest_dir, ignore_errors=True)

    def save_case(self, repository_name, annotated_code, relative_path, broken_hash, repaired_hash, log):
        output_file = self.output_csv

        with output_file.open("a", newline='', encoding="utf-8") as csvfile:
            fcntl.flock(csvfile, fcntl.LOCK_EX)
            try:
                write_header = output_file.stat().st_size == 0
                writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_ALL)
                if write_header:
                    writer.writerow(["repository_name", "annotated_code", "relative_path", "broken_hash", "repaired_hash", "outdated_test_log"])
                writer.writerow([repository_name, annotated_code, relative_path, broken_hash, repaired_hash, log])
            finally:
                fcntl.flock(csvfile, fcntl.LOCK_UN)

if __name__ == "__main__":
    repository_name = sys.argv[1]
    repository_path = sys.argv[2]
    output_csv = sys.argv[3]

    m = Main(repository_name, repository_path, output_csv)
    m.process_repository()
