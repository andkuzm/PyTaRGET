You are a developer on a Java project. You have just changed the main source code of the project: the changes you made are in the file SOURCE_CHANGES.diff in the project directory (you know the intent behind them). The project's test suite was written for the old source code and no longer compiles or passes.

Your task: bring the tests up to date so that the whole test suite compiles and passes again with the current source code.

Rules (they matter for the evaluation):
1. Work only inside the project directory {WS} . Do not read, list or search any other directory. Do not use the network. There is no useful version-control history here and you must not look for the "fixed" tests anywhere else.
2. Do not modify anything under src/main or any pom.xml: the source code is the new truth, only the tests change.
3. Do not weaken the tests: do not loosen assertions, do not comment tests out, do not add @Disabled. When behaviour changed, update the expected values to the new behaviour; when an API changed, adapt the calls. Delete a test, or a single assertion, ONLY when what it checks no longer exists in the source code at all (the tested class, method or behaviour was removed); if any part of what it checks still exists, keep checking that part. A check that valid, compiling code can no longer trigger (for example because a parameter type now rules the bad input out, even if the throwing line is still in the source) also counts as gone; never use casts or raw types to force it. When a check cannot be kept, delete it: never replace it with one that cannot fail or that only tests the test's own code (for example a Java type test, or a cast in the test that throws by itself). Name every test you deleted in your final summary.
4. Run the tests with `tools/run_tests` (offline Maven; it prints a compact result and keeps the full log in .delegation/last_test.log). Use it instead of calling mvn yourself; to run only some tests pass them as arguments, for example `tools/run_tests MyTest#myMethod`.
5. Be economical: read only what you need, prefer targeted reads over dumping large files.

When all tests pass, reply with a summary of at most 3 short lines: which test files you changed and anything you are not sure about. No lists, no code.
