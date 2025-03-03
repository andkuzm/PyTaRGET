Data collection process includes four files, in this readme whole process of data collection will be described presenting and explining entire workflow that exists at a time of writing it, but since descriptions will be relatively superficial and workflow should stay more or less the same description should fit regardless

Main Components
GitHubSearch

Purpose:
Searches GitHub for repositories using a provided token, a target clone directory, and an output path.

Key Functionality:

    Queries GitHub: Uses search queries to find repositories.
    Iterates Over Results:
        For each new repository, creates an instance of the Main class (from main_repository_miner.py) to process it.

main_repository_miner (Main)

Purpose:
Manages the high-level workflow for processing a single repository.

Workflow:

    Instantiation:
        Initializes with the repository name and the path where it will be cloned.
        Creates an instance of RepositoryActions (from repository_actions.py) for that repository.
    Process Repository:
        Cloning:
            Calls clone_repository_last() to clone the repository, set file permissions, install dependencies, adjust sys.path, and record the latest commit hash.
        Test Detection:
            Uses has_tests() to check for the presence of Python test files.
        Broken-to-Repaired Detection:
            If tests exist, identifies broken-to-repaired test cases via has_broken_to_repaired_test() and extract_broken_to_repaired_list().
        Code Extraction & Annotation:
            For each broken-to-repaired test case, extracts the test and source code by calling extract_and_annotate_code(), which in turn uses annotate_code() to generate an annotated string.
        Saving Data:
            Saves each annotated case to a CSV file using save_case().

repository_actions.py

Purpose:
Implements repository-specific operations including cloning, setting file permissions, installing dependencies, test detection and execution, commit navigation, AST extraction, dynamic coverage measurement, and code annotation.

Key Methods:

    clone_repository_last()
        Clones the repository from GitHub.
        Sets full permissions on the cloned repository.
        Installs dependencies via an editable install.
        Adjusts sys.path to include the repository’s directories (outer and nested).
        Creates an __init__.py in the top-level folder if missing.
        Retrieves and stores the latest commit hash.

    has_tests() & list_test_files()
        Recursively scan the repository for Python files containing test-related patterns.

    find_test_methods(test_rel_path)
        Uses the AST module to parse a test file and extract test function names.

    check_if_test_changed(test_rel_path, test_method)
        Compares the AST of a test method and the dynamic coverage data between commits.
        Determines whether the test code has changed significantly while the underlying source code remains nearly identical (≥95% similarity).

    extract_broken_to_repaired_list()
        Iterates over commits and test files.
        Builds a set of broken-to-repaired test cases (instances of Broken_to_repaired) that capture:
            The relative file path.
            The test method name.
            The commit hash when the test was broken.
            The commit hash when the test was repaired.

    extract_and_annotate_code(broken_to_repaired_instance) & annotate_code()
        Check out both the broken and repaired commits.
        Extract the test method’s code (using extract_method_code() and extract_method_ast()) and dynamic coverage data (via extract_covered_source_coverage()).
        Generate a unified diff between the broken and repaired test code.
        Annotate the diff and code using special markers:
            TESTCONTEXT: Contains the broken test (wrapped in [<BREAKAGE>]... [</BREAKAGE>]), the repaired test (wrapped in [<REPAIREDTEST>]... [</REPAIREDTEST>]), and the diff (annotated with [<HUNK>], [<DEL>], [<ADD>]).
            REPAIRCONTEXT: Contains the related source code (wrapped in [<HUNK>]... [</HUNK>]).
        Returns the annotated string.

    install_dependencies() & set_full_permissions()
        Ensure that repository dependencies are installed.
        Recursively set full permissions (read, write, execute) on all files and directories.

    Commit Navigation:
        move_to_earlier_commit() and move_to_later_commit() allow traversal of the commit history to compare test behavior across revisions.


py_parser.py

Purpose:
Implements subprocess related operations.

Key Methods:

    compile_and_run_test_python()
        Runs individual tests using pytest.
        Returns a TestVerdict object (e.g., SUCCESS, FAILURE, SYNTAX_ERR, TIMEOUT, or UNCONVENTIONAL) based on the test output.
