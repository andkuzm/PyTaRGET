import csv
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import repository_actions_R

class Main_r:
    def __init__(self, repository_name, repository_path, repo_commit, out_path):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.repo_commit = repo_commit
        self.out_path = out_path

    def process_repository(self):
        print("Processing repository: {}".format(self.repository_name))
        repository = repository_actions_R.RepositoryActions(self.repository_name, self.repository_path,self.repo_commit, self.out_path)

        if repository.checkout_commit(self.repo_commit ,self.repository_path/Path(self.repository_name)):
            if repository.has_tests():
                repaired_cases = repository.find_repaired_test_cases()
                if repaired_cases:
                    print("a")
                else:
                    print("no repaired test cases")
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