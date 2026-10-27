import csv
import fcntl
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import repository_actions

# One row per tenth of a repository's history plus "repo" rows with the stage a repository reached ("started" is
# written first, so a repository that was killed by the time limit shows up as started without a later row).
FUNNEL_COLUMNS = ["repository_name", "bucket", "stage", "commits", "relevant_commits", "visited_pairs",
                  "relevant_pairs", "pairs_with_changed_test", "changed_tests", "parent_success", "child_success",
                  "cases", "details"]

# test_name ("name" or "Class.name"), broken_date and repaired_date (committer dates) come last, so readers that pick
# columns by name keep working. The Encoder splits chronologically on broken_date when it is present.
CSV_HEADER = ["repository_name", "annotated_code", "relative_path", "broken_hash", "repaired_hash",
              "outdated_test_log", "test_name", "broken_date", "repaired_date"]


class Main:
    def __init__(self, repository_name, repository_path, output_csv):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.output_csv = Path(output_csv)
        self.funnel_csv = self.output_csv.with_name(self.output_csv.stem + "_funnel.csv")

    def save_funnel(self, rows):
        if not rows:
            return
        with self.funnel_csv.open("a", newline='', encoding="utf-8") as csvfile:
            fcntl.flock(csvfile, fcntl.LOCK_EX)
            try:
                writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_ALL)
                if self.funnel_csv.stat().st_size == 0:
                    writer.writerow(FUNNEL_COLUMNS)
                for row in rows:
                    writer.writerow([row.get(column, "") for column in FUNNEL_COLUMNS])
            finally:
                fcntl.flock(csvfile, fcntl.LOCK_UN)

    def save_stage(self, stage, details=""):
        self.save_funnel([{"repository_name": self.repository_name, "bucket": "repo", "stage": stage,
                           "details": details}])

    def process_repository(self):
        stage, details = "no_tests", ""
        self.save_stage("started")
        try:
            repository = repository_actions.RepositoryActions(self.repository_name, self.repository_path)
            repository.clone_repository_last()

            if repository.has_tests():
                repaired_cases = repository.find_repaired_test_cases()
                self.save_funnel(getattr(repository, "funnel_rows", lambda: [])())
                stage = "searched"
                if repaired_cases:
                    for repaired_test in repaired_cases:
                        annotated_code = repository.extract_and_annotate_code(repaired_test)
                        if not annotated_code or annotated_code == "Error":
                            continue
                        self.save_case(self.repository_name, annotated_code, repaired_test.rel_path, repaired_test.broken, repaired_test.repaired, f"[<TESTLOG>]\n{repaired_test.log}\n[</TESTLOG>]\n",
                                       test_name=repaired_test.test_name,
                                       broken_date=repository.get_commit_date(repaired_test.broken),
                                       repaired_date=repository.get_commit_date(repaired_test.repaired))

            try:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", self.repository_name.split("/")[-1]],
                               capture_output=True, text=True, check=True, env=os.environ)
            except subprocess.CalledProcessError:
                pass
        except Exception as e:
            print(f"Repository failed: {self.repository_name}: {e}")
            stage, details = "failed", str(e)[:300]
        finally:
            self.save_stage(stage, details)
            self.cleanup()

    def cleanup(self):
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        shutil.rmtree(dest_dir, ignore_errors=True)

    def save_case(self, repository_name, annotated_code, relative_path, broken_hash, repaired_hash, log,
                  test_name="", broken_date="", repaired_date=""):
        output_file = self.output_csv

        with output_file.open("a", newline='', encoding="utf-8") as csvfile:
            fcntl.flock(csvfile, fcntl.LOCK_EX)
            try:
                write_header = output_file.stat().st_size == 0
                writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_ALL)
                if write_header:
                    writer.writerow(CSV_HEADER)
                writer.writerow([repository_name, annotated_code, relative_path, broken_hash, repaired_hash, log,
                                 test_name, broken_date, repaired_date])
            finally:
                fcntl.flock(csvfile, fcntl.LOCK_UN)

if __name__ == "__main__":
    repository_name = sys.argv[1]
    repository_path = sys.argv[2]
    output_csv = sys.argv[3]

    m = Main(repository_name, repository_path, output_csv)
    m.process_repository()
