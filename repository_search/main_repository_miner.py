import csv
import fcntl
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import processed_log
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
    def __init__(self, repository_name, repository_path, output_csv, processed_file=None):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.output_csv = Path(output_csv)
        # The processed-repositories file; when given, the repository's line there records where an interrupted
        # search stopped (see processed_log.py) and a new run continues from it.
        self.processed_file = Path(processed_file) if processed_file else None
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

    @staticmethod
    def _key(broken, repaired, relative_path, test_name):
        return (broken, repaired, relative_path, test_name)

    def existing_case_keys(self):
        """Keys of this repository's cases that are already in the dataset, so a resumed run never saves one twice."""
        keys = set()
        if not self.output_csv.exists():
            return keys
        csv.field_size_limit(2 ** 31 - 1)
        with self.output_csv.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter="|")
            header = next(reader, None)
            if not header or "repository_name" not in header:
                return keys
            col = {name: i for i, name in enumerate(header)}
            for row in reader:
                if len(row) <= col["repository_name"] or row[col["repository_name"]] != self.repository_name:
                    continue
                name = row[col["test_name"]] if "test_name" in col and len(row) > col["test_name"] else ""
                keys.add(self._key(row[col["broken_hash"]], row[col["repaired_hash"]], row[col["relative_path"]], name))
        return keys

    def handle_case(self, repository, case, seen):
        """Annotate and save one verified case immediately; a failure costs this case only, not the repository."""
        key = self._key(case.broken, case.repaired, case.rel_path, case.test_name)
        if key in seen:
            return
        try:
            annotated_code = repository.extract_and_annotate_code(case)
            if not annotated_code or annotated_code == "Error":
                return
            self.save_case(self.repository_name, annotated_code, case.rel_path, case.broken, case.repaired,
                           f"[<TESTLOG>]\n{case.log}\n[</TESTLOG>]\n",
                           test_name=case.test_name,
                           broken_date=repository.get_commit_date(case.broken),
                           repaired_date=repository.get_commit_date(case.repaired))
            seen.add(key)
        except Exception as e:
            print(f"Case {case.test_name} of {self.repository_name} could not be saved: {e}")

    def resume_point(self, repository, fresh_head):
        """(head to search from, last completely processed commit or None) for this repository."""
        previous = processed_log.resume_state(self.processed_file, self.repository_name)
        if previous is None:
            processed_log.mark_started(self.processed_file, self.repository_name, fresh_head)
            return fresh_head, None
        head = previous.head or fresh_head
        marker = previous.marker or None
        if not repository.is_ancestor(head, head):  # the pinned commit is gone from the remote (force push)
            print(f"{self.repository_name}: recorded start {head} no longer exists, starting over")
            head, marker = fresh_head, None
            processed_log.mark_progress(self.processed_file, self.repository_name, head, "")
        elif marker and not repository.is_ancestor(marker, head):
            print(f"{self.repository_name}: recorded progress {marker} is not in the history, starting over")
            marker = None
            processed_log.mark_progress(self.processed_file, self.repository_name, head, "")
        if marker:
            print(f"{self.repository_name}: resuming after commit {marker}")
        return head, marker

    def process_repository(self):
        stage, details = "no_tests", ""
        self.save_stage("started")
        head, marker, last_marker, finished = None, None, None, False
        try:
            repository = repository_actions.RepositoryActions(self.repository_name, self.repository_path)
            repository.clone_repository_last()
            head = repository.current_hash
            if self.processed_file is not None:
                head, marker = self.resume_point(repository, head)
                last_marker = marker

            if repository.has_tests():
                # Start from the commit recorded in the first run, so the walk repeats exactly.
                repository.current_hash = head
                repository.visited_commits.add(head)
                seen = self.existing_case_keys() if marker else set()

                def on_progress(commit):
                    nonlocal last_marker
                    last_marker = commit
                    if self.processed_file is not None:
                        processed_log.mark_progress(self.processed_file, self.repository_name, head, commit)

                repository.find_repaired_test_cases(
                    on_case=lambda case: self.handle_case(repository, case, seen),
                    on_progress=on_progress,
                    resume_after=marker,
                )
                self.save_funnel(getattr(repository, "funnel_rows", lambda: [])())
                stage = "searched"

            finished = True
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
            self.finish_progress(head, marker, last_marker, finished)
            self.cleanup()

    def finish_progress(self, head, marker_at_start, last_marker, finished):
        """Completed repositories are marked done. After an error the repository stays resumable only if this run
        got further than the one before it, otherwise it would be retried forever."""
        if self.processed_file is None:
            return
        progressed = last_marker is not None and last_marker != marker_at_start
        if finished or not progressed:
            processed_log.mark_done(self.processed_file, self.repository_name, head or "")

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
    processed_file = sys.argv[4] if len(sys.argv) > 4 else None

    m = Main(repository_name, repository_path, output_csv, processed_file)
    m.process_repository()
