# Next potential step: is it effective to delegate test repair to a cheaper model? (WP4 pilot; idea of the user, saved 2026-11-21; NOT started)

## Question
When a strong agent has to bring a project's tests up to date with the current source, does it save money (and time) to let a cheap model, here particle-1, write the repairs while the strong agent only
checks them and fixes what is still wrong, without losing quality compared with the strong agent doing everything alone?

## Design: two fresh sub-agents, the same task, the same repositories
Both sub-agents start with NO context (no memory of this project, no results) and get the same task text: "the tests of this repository no longer match the current source code; update the tests so that they pass again
without weakening what they check". Only the working method differs.

| Arm | Who works | Method |
|---|---|---|
| **A, solo** | one strong agent (for example Claude, fixed model and settings) | finds the failing tests, reads the source changes, edits the tests, runs them; no delegation |
| **B, delegated** | the same strong agent, told to use a delegate | for each failing test it calls the delegate (particle-1 through a tool, see below), applies/inspects the returned repair, runs the test, and edits by itself only what the delegate did not get right |

- **Same inputs:** the same set of tasks (below), the same tool permissions (shell, file edit, run tests), the same limits on turns and tokens, the same repetitions.
- **Delegate tool** (this is the WP4 tool in its smallest form): `delegate_test_repair(test id)` collects the broken test, runs it to get the failure output, takes the relevant source diff, builds the prompt of our best prompt-only variant
  (test context + failure output; the cheap-model results show B gives +6..+31 points), calls particle-1, applies the candidate to the file and runs the test, and returns the patch and the test result. The strong agent never sees the cheap model's reasoning, only the result.
  Particle-1 is API-only (closed weights, post-trained on DeepSeek V4.1 Flash); for a reproducible variant use the paid endpoint (0.07 / 0.50 USD per 1M tokens) or a fully local model (Qwen3-Coder-30B, or the open DeepSeek V4.1 Flash base).

## Tasks
- Commit-level tasks, not single tests: a repository at the AFTER commit (source change in place) with ALL test files that the repair changed put back to their BEFORE version, one fresh single-commit repository without history or remote
  (the sibling-safe workspaces of `data_processing/agent_workspace.py` extended from one test to all broken tests of the commit; Java needs the whole module to compile, so there the unit is the module).
- Pool: the cases of the Python and Java contrast subsets grouped by repository and commit; choose about 20-30 tasks with 1-5 broken tests each and a working test environment (the preparation scripts `scripts/agent/prepare_cases*.py` already build and check such workspaces).
- Contamination: the strong model may know these public repositories; report it as a limitation, do not claim otherwise.

## Measurements (decided before running)
- **Success per test:** the repaired test passes AND the checked rule holds (it needs the repaired part or still asserts), scored with the plausibility runners (`plausibility_test.py`, `java_plausibility.py`) on the final files; also the share of tasks with all tests repaired.
- **Cost:** tokens and dollars of the STRONG model in each arm (from the API usage), tokens and dollars of the delegate, wall-clock time, number of strong-model calls. Headline numbers: cost per plausible repair and strong-model tokens per plausible repair.
- **Behaviour of arm B:** how often the delegate's repair was accepted as is, how often the strong agent had to repair it, how often it wrongly accepted a weakened or wrong repair (checked against the scoring), time spent verifying.
- **Quality problems in both arms:** weakened or deleted tests, edits outside the tests (source changes), use of history/network (audit like `audit_trajectory`).
- **Statistics:** paired by task, 2-3 repetitions per task and arm to see the run-to-run variance (temperature > 0), paired differences with a bootstrap interval; with 20-30 tasks only large effects are detectable, say so.

## Decision rule (to fix before running)
Delegation is worthwhile if the strong-model cost per plausible repair drops by at least a stated share (for example 40%) and the success rate stays within a stated margin (for example 5 points) of arm A; if the strong agent accepts many bad repairs, report that as the key risk.

## What exists already and what is missing
- **Exists:** leak-safe workspaces, offline/sandboxed agent runner, scoring (plausibility, checked rule), failure-output context builders for Python and Java, cost accounting from request logs, results showing the cheap prompt-only models with failure output reach the local agent's level (Qwen3-Coder: Python 48.4 vs agent 48.3; particle-1: Python 58.2, Java 73.0-73.5), `docs/results/paid_models_summary.md`.
- **Missing:** the commit-level workspace builder; the delegate tool (an MCP server or a CLI plus a Claude Code skill/sub-agent definition); a way to run the two arms with fresh contexts and to read their token usage (Claude Code sub-agents report usage; a scripted agent loop through the API gives exact control but costs credit); the task selection and the pre-registered success definition.
- **Budget:** a strong main agent is the expensive part. At the measured agent usage (about 570k prompt tokens per task for a 30B local agent) a Claude Sonnet-class model at 2 / 10 USD per 1M tokens would cost about 1 USD per task, so 25 tasks x 2 arms x 3 repetitions is about 150 USD without prompt caching; a cheaper strong model (for example Claude Haiku 5.5 at 0.10 / 0.50 USD per 1M tokens) is about 10 times less. Decide the main model and the credit before starting.

## Risks
- The delegate returns plausible but test-weakening edits and the checker accepts them; the cheap model's strength may partly be thinking depth (not tested yet); the free tier of particle-1 can change; prompt differences between the arms; small samples; token accounting in an interactive tool.
