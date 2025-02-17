import os
import repository_actions

class Main:
    def __init__(self, repository_name, path):
        self.repository_name = repository_name
        self.path = path

    def get_repository_name(self):
        return self.repository_name

    def get_path(self):
        return self.path

    def process_repository(self):
        print("processing repository: {}".format(self.repository_name))
        # Step 1: Clone the repository (and set up appropriate branch/worktree)
        repository = repository_actions.RepositoryActions(self.repository_name, self.path)
        repository.clone_repository_last(self.repository_name, self.path)

        # Step 2: Check if the repository contains tests
        repo_path = os.path.join(self.path, self.repository_name)
        if repository.has_tests(repo_path):
            # Step 3: Check if there are broken-to-repaired test cases in the repo
            if repository.has_broken_to_repaired_test(repo_path):
                # Extract commit hashes or change pairs for tests that were broken and then fixed
                broken_to_repaired_list = repository.extract_broken_to_repaired_list(repo_path)

                # Process each broken-to-repaired instance
                for repaired_test in broken_to_repaired_list:
                    # Validate that the error is of a type we care about
                    if valid_error(repo_path, repaired_test):
                        # Execute the test on the buggy version to verify failure and then extract the failing test and related source code
                        annotated_related_source_code, annotated_test = extract_and_annotate_code(repo_path,
                                                                                                  repaired_test)
                        # Save the processed data into the database, along with metadata like commit hashes and timestamps
                        save_case(self.repository_name, annotated_related_source_code, annotated_test)
                    else:
                        continue
            else:
                print("No broken-to-repaired test cases found.")
        else:
            print("Repository does not contain tests.")

        # Optional: Clean up temporary directories or worktrees
        cleanup_temporary_resources(self.repository_name, self.path)