You are helping to prepare the input for a smaller code model that will repair a broken Python test. You work ONLY with the bundle file you are given; do not read any other file, do not use the network, do not look at the repository history.

The bundle (a JSON file) contains: the failing test function, the failure output of running the old test after the project's source code changed, and ALL changed source hunks of the commit (ids 0..N).

Your job: decide what the smaller model needs and write it down. Output a JSON file with exactly these keys:
- "broken_lines": the lines of the test you believe must change (copy them from the test code, exactly).
- "source_changes": the part of the changed source hunks the smaller model needs to repair those lines, copied from the hunks (you may trim a hunk to the relevant function or lines; keep it short, at most about 1500 characters in total).
- "hint": at most one sentence on what is wrong (optional, may be empty).
Do NOT write the repaired test yourself and do not output the repair; only choose and curate context.

Write one file per bundle to the output path you are given, then reply with only the list of files you wrote and the name of the model you are.
