# Next potential step: is it effective to delegate test repair to a cheaper model? (WP4 pilot; idea of the user, saved 2026-11-21; NOT started)

## Question
When a strong agent has to bring a project's tests up to date with the current source, does it save money (and time) to let a cheap model, here particle-1, write the repairs while the strong agent only
checks them and fixes what is still wrong, without losing quality compared with the strong agent doing everything alone?

## Design: two fresh sub-agents, the same task, the same repositories
Both sub-agents start with NO context (no memory of this project, no results) and get the same task text: "the tests of this repository no longer match the current source code; update the tests so that they pass again
without weakening what they check". Only the working method differs.

| Arm | Who works | Method |
|---|---|---|
| **A, solo** | a fresh **Claude Code sub-agent** (the "strong arm": spawned with the Agent tool, one per task, fixed model and instructions) | finds the failing tests, reads the source changes, edits the tests, runs them; no delegation |
| **B, delegated** | a fresh Claude Code sub-agent of the same model and with the same text plus one paragraph | for each failing test it calls the delegate tool (particle-1, see below), inspects the returned repair, runs the test, and edits by itself only what the delegate did not get right |

- **Strong arm = our own sub-agents.** Each task and arm is one new sub-agent started with the Agent tool and nothing but the written task prompt (prompts are fixed in files BEFORE the first run and never tuned afterwards); the model is fixed by the tool's `model`
  parameter and is the same in both arms. Several sub-agents can run in parallel in the background, each on its own copy of the workspace. The delegate is called from the sub-agent's shell (a command on its PATH).
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
- **Missing:** the commit-level workspace builder; the delegate tool (an MCP server or a CLI plus a Claude Code skill/sub-agent definition); the two fixed prompts and a small launcher that prepares one workspace copy per task, arm and repetition; reading the sub-agents' usage and transcripts for the audit; the task selection and the pre-registered success definition.
- **Budget:** with Claude Code sub-agents the strong arm is not billed per token on the OpenRouter credit; the measured quantity is its token usage, tool calls and duration as reported by the harness for each sub-agent (to be verified in the pilot: the Agent tool result should
  carry them), converted to a list-price-equivalent cost (for example Claude Sonnet 5.5 at 2 / 10 USD per 1M tokens) for the cost comparison. The only real spend is the delegate (particle-1: free tier, or the paid endpoint at 0.07 / 0.50 USD per 1M tokens, cents in total).
  What limits the experiment is the usage allowance of the session and wall-clock time, not credit: keep the number of parallel sub-agents modest.
- **Pilot first:** 3 tasks x 2 arms x 1 repetition, to check that usage is reported, that the delegate tool works from a sub-agent shell, that the isolation holds, and how long a task takes; only then the full 20-30 tasks.

## Isolation and honesty of the comparison (specific to using our own sub-agents)
- A sub-agent shares our file system and network: it could read the datasets in the home directory (they contain the reference repairs) or fetch upstream history. Mitigations: workspaces are single-commit repositories without history or remote; the task prompt forbids looking outside the workspace;
  run the sub-agents in a session whose home holds only the workspaces and the delegate command (datasets stay in the verified backup `~/backups/2026-11-19/`), or start them in a mount namespace that hides the home; then AUDIT every sub-agent transcript for reads outside the workspace and for network access, and exclude or flag tasks that violate this.
- The delegate needs the OpenRouter key: use a separate key with a small spending limit, kept in a file read only by the delegate command (never in a prompt), so that a transcript can never contain the main key.
- The orchestrator (this assistant) must not influence the arms after the run starts: fixed prompts, automatic scoring, no manual repairs; the model that acts as strong arm may also know the public repositories (contamination).

## Risks
- The delegate returns plausible but test-weakening edits and the checker accepts them; the cheap model's strength may partly be thinking depth (not tested yet); the free tier of particle-1 can change; prompt differences between the arms; small samples; token accounting in an interactive tool.
