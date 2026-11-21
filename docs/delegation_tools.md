# Delegation experiment: material, tools, prompts and how to run a cell (prepared 2026-11-21; nothing of the experiment itself has been run yet except smoke tests)

Plan and design: `docs/delegation_experiment_plan.md`. State of the experiment: `~/delegation_exp/state.json` (read it first in every session; caps for the usage allowance are in `thresholds`).

## Tasks (`~/delegation_exp/tasks/<id>/`)
A task is a Java project at the AFTER state of its main source with the OLD tests (the tests no longer compile or pass), plus `SOURCE_CHANGES.diff` (the main-source changes: what the "developer" just changed). Reference tests, old tests and source hashes are kept OUTSIDE the workspace (`ref/`, `old_tests/`, `source_hashes.json`).
| id | what | state with the old tests |
|---|---|---|
| `shop` | synthetic Maven project (`~/delegation_exp/synth/shop`, written for this experiment, so not in any training data): 8 kinds of source change (renames, signature, return type, tiered behaviour, exception, package move, constructor, enum rename); 9 test classes / 32 tests | 44 compile errors in 7 files; reference state: 32/32 pass |
| `gson_a80` | real: google/gson commit a80bc24e (module `gson`), exception messages changed; 2 KB source diff | 66 failing tests (reference: 1261 run, all pass) |
| `gson_cbad` | real: google/gson commit cbad1aa7, 14 KB source diff, includes JsonReader position messages | 14 failing tests (reference: 1259 run, all pass) |
Build: `scripts/delegation/prepare_task.py synth <id>` / `real <id> <repo> <aCommit> <bCommit> [module]`; check: `verify_task.py <id>` (reference state passes online and fills the shared Maven cache `~/delegation_exp/m2`; the task state must fail offline). More real tasks: the TaRBench commit groups with 3-10 broken tests in one module (see the search in the session notes: gson, underscore-java, GraphWalker, xwiki-commons ...).

## Tools inside a workspace (`ws/tools/`)
- `tools/run_tests [Class#method ...]`: offline Maven; prints a COMPACT result (compile errors deduplicated, at most ~15 lines; or tests run/passed/failed with the failing tests and one line each); the full log is in `.delegation/last_test.log`.
- `tools/delegate [--units all-failing|files|Class#method] [--context auto|strong|both] [--model alias] [--rounds N] [--dry-run]` (delegated arm only): finds the failing test units (a whole test file when it has at most 400 lines, otherwise per method and a header unit; runtime failures per method), builds the prompt (test code, errors, the part of the failure output that concerns it, source hunks of SOURCE_CHANGES.diff ranked by TF-IDF; `strong`: the strong agent's `.delegation/context.json` replaces the ranked hunks, `both`: it is added), calls the delegate, applies the answers with safeguards (rejects answers that remove most code or assertions; adds imports), re-runs the tests and prints ONE compact result with the delegate's tokens and cost. Rounds feed new errors back. Every call is logged in `.delegation/delegate_calls.jsonl`; backups and raw answers in `.delegation/backup/`.
  Delegates (config aliases): `particle-free` (default; OpenRouter free tier), `particle` (paid 0.07/0.50 USD per 1M), `qwen3-coder` (OpenRouter), `local` (an OpenAI-compatible vLLM endpoint on localhost:8000: the agentic local delegate D2 is NOT implemented yet, only the plain local model call).
  The key is read from `~/.config/openrouter/key` by the tool (not by the agent). A separate key with a small spending limit is recommended before real runs.
- Scoring (`score_task.py`, not visible to the agent): module compiles, every test of the OLD suite exists and passes, no test with fewer assertions than in the reference tests, no change under `src/main` or any pom, plus lists of missing/failing/weakened/new tests.

## Prompts (`docs/delegation_prompts/`, fixed before the first run; never tune them afterwards)
`run_common.md` (the developer framing and the rules: work only in the project directory, no network, no source or pom edits, do not weaken tests, use `tools/run_tests`, be economical), `arm_solo.md`; delegated arm = `arm_delegate_intro.md` + `strategy_S1|S2|S3.md` + `context_auto|strong|both.md` + `verify_exec|diff.md`. Step 0 prompt: `step0_context_strong.md`.

## Running a cell (task x arm x repetition)
1. `python scripts/delegation/launcher.py new-run <task> solo|delegate <rep> [--strategy S1|S2|S3] [--context auto|strong|both] [--verify exec|diff] [--delegate-model particle-free]` creates `~/delegation_exp/runs/<run id>/ws` (a fresh copy of the task workspace; the solo arm has no `tools/delegate`) and `prompt.md`.
2. Give `prompt.md` to a FRESH Claude Code sub-agent (Agent tool, `model: sonnet`, nothing else in its prompt), note its reported tokens, tool uses and duration, read `claude -p "/usage"` before and after.
3. `python scripts/delegation/launcher.py finish-run <run id> --tokens N --tool-uses N --duration-s S --percent-before a --percent-after b` scores the workspace and records everything in the state file. `launcher.py status` lists the cells.
4. Audit the sub-agent's transcript for reads outside the workspace and for network use (the tool calls of the transcript, as done for the calibration run).
Token economy for the strong arm: compact tool outputs, no reasoning from the delegate, one result line per delegate call.

## Smoke test result (2026-11-21, free particle-1, no Claude sub-agent)
`shop`: two `tools/delegate` calls (round 1: 7 units, 6 repaired; round 2: the last file) repaired everything: 32/32 pass, nothing missing or weakened, source untouched; 8 delegate calls, 19k tokens in, 10k out, cost 0 (free tier).
