import os
import shutil
import stat
import subprocess
from pathlib import Path


class RepositoryActions:
    def __init__(self, repository_name, repository_path, current_hash=None):
        self.repository_name = repository_name
        self.repository_path = repository_path
        self.current_hash = current_hash

    def get_repository_name(self):
        return self.repository_name
    def get_repository_path(self):
        return self.repository_path
    def get_current_hash(self):
        return self.current_hash
    def set_current_hash(self, new_hash):
        self.current_hash = new_hash

    def clone_repository_last(self):
        """
        Clones the repository from GitHub at its latest commit.
        Sets self.current_hash to the latest commit hash.
        Assumes repository_name is in the form "owner/repository".
        """
        # Construct the GitHub repository URL
        repo_url = f"https://github.com/{self.repository_name}.git"
        # Destination directory based on repository name (without owner)
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])

        if os.path.exists(dest_dir):
            print(f"Repository already exists at {dest_dir}. Removing it...")
            # If the destination already exists, remove it for a fresh clone
            def handle_remove_readonly(func, path, exc_info):
                # Change the permissions and then call the removal function again
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(dest_dir, onerror=handle_remove_readonly)

        print(f"Cloning repository from {repo_url} to {dest_dir}...")
        cmd = ["git", "clone", repo_url, dest_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Error cloning repository:", result.stderr)
            raise Exception("Repository clone failed")
        else:
            print("Repository cloned successfully.")

        # Update current_hash with the latest commit hash from the cloned repository
        hash_cmd = ["git", "rev-parse", "HEAD"]
        hash_result = subprocess.run(hash_cmd, cwd=dest_dir, capture_output=True, text=True)
        if hash_result.returncode == 0:
            latest_hash = hash_result.stdout.strip()
            self.set_current_hash(latest_hash)
            print(f"Current commit hash set to: {self.current_hash}")
        else:
            print("Error obtaining latest commit hash:", hash_result.stderr)
            raise Exception("Failed to obtain commit hash")

        return dest_dir

    def has_tests(self):
        """
        Checks if the repository contains Python test files.
        This function recursively walks through the repository directory,
        looking for .py files that contain test-related keywords such as
        "import pytest", "from unittest", or function definitions that start with "def test_".
        """
        repo_dir = Path(self.repository_path) / self.repository_name.split("/")[-1]

        # Recursively iterate through all .py files in the repository
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        # Check for common Python testing keywords
                        if ("import pytest" in content or
                                "from unittest" in content or
                                "def test_" in content):
                            return True
                    except Exception as e:
                        # Skip files that cannot be read (e.g., due to encoding issues)
                        continue
        return False

    def has_broken_to_repaired_test(self):
        pass

    def extract_broken_to_repaired_list(self):
        pass

    def move_to_earlier_commit(self):
        """
        Moves the repository checkout to the parent commit of the current HEAD.
        Updates the current_hash accordingly.
        """
        dest_dir = os.path.join(self.repository_path, self.repository_name.split("/")[-1])
        print(f"Current repository directory: {dest_dir}")

        # Get the parent commit hash (HEAD^)
        cmd_parent = ["git", "rev-parse", "HEAD^"]
        proc_parent = subprocess.run(cmd_parent, cwd=dest_dir, capture_output=True, text=True)
        if proc_parent.returncode != 0:
            print("Error obtaining parent commit:", proc_parent.stderr)
            raise Exception("Failed to get parent commit hash")

        parent_hash = proc_parent.stdout.strip()
        print(f"Parent commit hash: {parent_hash}")

        # Checkout the parent commit
        cmd_checkout = ["git", "checkout", parent_hash]
        proc_checkout = subprocess.run(cmd_checkout, cwd=dest_dir, capture_output=True, text=True)
        if proc_checkout.returncode != 0:
            print("Error checking out parent commit:", proc_checkout.stderr)
            raise Exception("Failed to checkout parent commit")

        # Update the current hash to the parent commit
        self.current_hash = parent_hash
        print(f"Repository is now at commit: {self.current_hash}")
        return parent_hash

test = RepositoryActions("andkuzm/fourth_task", "C:\\Users\\kandr\\PycharmProjects\\PyTaRGET\\test")
# test.clone_repository_last()
# print(test.current_hash)
# test.move_to_earlier_commit()
# print(test.current_hash)
print(test.has_tests())