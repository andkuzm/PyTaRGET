# Next potential step: is it effective to delegate test repair to a cheaper model? (WP4 pilot; idea of the user, saved 2026-11-21; NOT started)

## Question
When a strong agent has to bring a project's tests up to date with the current source, does it save money (and time) to let a cheap model, here particle-1, write the repairs while the strong agent only
checks them and fixes what is still wrong, without losing quality compared with the strong agent doing everything alone?

## Design: two fresh sub-agents, the same task, the same repositories
Both sub-agents start with NO context (no memory of this project, no results) and get the same task text: "the tests of this repository no longer match the current source code; update the tests so that they pass again
without weakening what they check". Only the working method differs.

| Arm | Who works | Method |
|---|---|---|
| **A, solo** | a fresh **Claude Code sub-agent** (the "strong arm": spawned with the Agent tool, one per task, model fixed to **Claude Sonnet 5.5**, fixed instructions) | finds the failing tests, reads the source changes, edits the tests, runs them; no delegation |
| **B, delegated** | a fresh Claude Code sub-agent of the same model and with the same text plus one paragraph | for each failing test it calls the delegate tool (particle-1, see below), inspects the returned repair, runs the test, and edits by itself only what the delegate did not get right |

- **Strong arm = our own sub-agents.** Each task and arm is one new sub-agent started with the Agent tool and nothing but the written task prompt (prompts are fixed in files BEFORE the first run and never tuned afterwards); the model is fixed by the tool's `model`
  parameter and is the same in both arms. Several sub-agents can run in parallel in the background, each on its own copy of the workspace. The delegate is called from the sub-agent's shell (a command on its PATH).
- **Same inputs:** the same set of tasks (below), the same tool permissions (shell, file edit, run tests), the same limits on turns and tokens, the same repetitions.
- **Delegate tool** (this is the WP4 tool in its smallest form): `delegate_test_repair(test id)` collects the broken test, runs it to get the failure output, takes the relevant source diff, builds the prompt of our best prompt-only variant
  (test context + failure output; the cheap-model results show B gives +6..+31 points), calls particle-1, applies the candidate to the file and runs the test, and returns the patch and the test result. The strong agent never sees the cheap model's reasoning, only the result.
  Particle-1 is API-only (closed weights, post-trained on DeepSeek V4.1 Flash); for a reproducible variant use the paid endpoint (0.07 / 0.50 USD per 1M tokens) or a fully local model (Qwen3-Coder-30B, or the open DeepSeek V4.1 Flash base).

## Decisions of the user (2026-11-21)
- **Strong model: Claude Sonnet 5.5** for every sub-agent of both arms (set explicitly in each Agent call, never left to a default).
- **Pacing:** use at most about HALF of the usage limit at a time and spread the work over SEVERAL SESSIONS; the experiment is not meant to be run in one go.
- **Pilots decide the delegation strategy:** the first (small) pilots compare ways of delegating; the full comparison against the solo arm uses the best one.

## Pacing and continuity across sessions
- **Usage is readable:** `claude -p "/usage"` run from a shell prints the current session percentage (and its reset time) and the weekly percentage for all models (approximate, "sessions on this machine only"; each call costs one small request).
  Rule: read it at the start of a session and between batches; stop starting new sub-agents when the SESSION reaches about 80% (the user's setting for 2026-11-21) or the WEEK reaches about 90%, whichever comes first, and never start a batch that cannot finish below these thresholds.
  The weekly limit is the real constraint when it is already high (2026-11-21 03:20: session 1%, week 79%, week resets 18:59 the same day): heavy runs wait for the reset. Record every reading in the state file together with the tokens the sub-agents reported, so that the tokens-to-percent ratio can be estimated.
- **Small batches:** at most a few sub-agents in parallel, a fixed maximum number of tasks per session, stop when the cap is reached; nothing is started that cannot finish within the cap.
- **State file:** `~/delegation_exp/state.json` (plus one directory per task, arm and repetition with workspace, prompt, transcript path, usage, result): the status of every task x arm x strategy x repetition, so that a new session reads it and continues where the last one stopped;
  the launcher is resumable and never reruns a finished cell. Results are scored automatically afterwards (plausibility runners), also in a later session.
- **Order:** step 0 (offline context makers, above), then pilots first (3 tasks), strategy comparison next (about 10 tasks x the candidate strategies), then the final comparison (20-30 tasks x solo vs the best strategy x 2-3 repetitions), each stage only after the previous stage was read.

## Delegation strategies to compare in the pilots (arm B variants)
- **S1, delegate everything, then verify:** the delegate repairs every failing test first; the strong agent runs the tests, reads the diff and fixes the leftovers.
- **S2, triage:** the strong agent looks at each failing test, delegates the ones it judges routine (renames, signature changes) and repairs the hard ones itself.
- **S3, delegate with feedback:** S1 plus one second delegate call that passes the failure output of the first attempt back before the strong agent steps in.
- **S4, verification depth:** S1 with execution-only verification (run the tests, do not read the diff) against reading the diff as well (is the test still meaningful?).
Measure for each: strong-model tokens, how often the delegate's repair was accepted as is, how often it had to be repaired, wrongly accepted repairs (against the scoring), and success per test.

## Second factor (user idea, 2026-11-21): how the delegate gets its context, and what the delegate is
The delegate (a prompt-only model such as particle-1) is only as good as the context it is given. Three ways to make that context, and an extra kind of delegate:

| Context maker | How the delegate's context is made |
|---|---|
| **C-auto** | automatically, the way the searcher/prompt pipeline does it: the failing tests, their failure output (running the tests), the error lines found in that output, and the source hunks of the change ranked by the TF-IDF `HunkPrioritizer` and trimmed to the window |
| **C-strong** | the strong agent (Sonnet 5.5) reads the changes for ALL failing tests at once and itself picks, per test, which hunks, helper code and failure lines the delegate gets (one reading pass shared by all tests of the task) |
| **C-both** | the automatic context, which the strong agent then curates or extends (adds what it thinks is missing, removes noise) |

| Delegate | What it is |
|---|---|
| **D1** | particle-1 (or another cheap API model), prompt-only, one call per test with the context above |
| **D2** | an **agentic local model**: Qwen3-Coder-30B with mini-SWE-agent (our sandboxed agent, 80 calls) that explores the repository itself; the context of the strong agent, if any, is only a short brief |

- **Realism gap to measure:** in our datasets the prompts contain the BREAKAGE lines (the lines the reference repair changes), which a real run does not have; C-auto must derive the location from the error lines of the failure output (the TaRGet paper: error lines and breakage lines coincide in only about half of the cases).
  So the pipeline comparison has to include "oracle lines" vs "derived lines".
- **Strong-agent cost of context:** C-strong and C-both spend strong-model tokens on reading the changes; the savings of delegating must be larger than that. Measure the tokens of the context-making step separately.
- **D2 needs a GPU behind the sub-agent:** a sub-agent cannot start a GPU job interactively without queueing delays; either keep a vLLM server on a reserved GPU node for the duration of a session (then D2 is a call like D1) or run D2 as a batch step (the strong agent hands over, waits for the result). Decide in the pilots.

### Step 0 (offline, cheap, no sub-agents): test the context makers on the data we already have
Use the 59 Java and 61 Python cases (and their workspaces) with prompt-only delegates (particle-1, Qwen3-Coder with failure output: about 0.0005 USD per case) and compare C-auto (without oracle lines), C-auto with oracle lines (what we measured so far), C-strong and C-both:
plausible@1 per context maker, and the strong-model tokens spent on making the context. Only a few Sonnet 5.5 sub-agent calls are needed (one per task for C-strong), so it also serves as the first calibration of the usage cap. The result decides which context maker goes into the pilots.

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
- **Pilot first:** 3 tasks x (solo + S1) x 1 repetition, to check that usage is reported, that the delegate tool works from a sub-agent shell, that the isolation holds, and how long a task takes; only then the full 20-30 tasks.

## Isolation and honesty of the comparison (specific to using our own sub-agents)
- A sub-agent shares our file system and network: it could read the datasets in the home directory (they contain the reference repairs) or fetch upstream history. Mitigations: workspaces are single-commit repositories without history or remote; the task prompt forbids looking outside the workspace;
  run the sub-agents in a session whose home holds only the workspaces and the delegate command (datasets stay in the verified backup `~/backups/2026-11-19/`), or start them in a mount namespace that hides the home; then AUDIT every sub-agent transcript for reads outside the workspace and for network access, and exclude or flag tasks that violate this.
- The delegate needs the OpenRouter key: use a separate key with a small spending limit, kept in a file read only by the delegate command (never in a prompt), so that a transcript can never contain the main key.
- The orchestrator (this assistant) must not influence the arms after the run starts: fixed prompts, automatic scoring, no manual repairs; the model that acts as strong arm may also know the public repositories (contamination).

## Risks
- The delegate returns plausible but test-weakening edits and the checker accepts them; the cheap model's strength may partly be thinking depth (not tested yet); the free tier of particle-1 can change; prompt differences between the arms; small samples; token accounting in an interactive tool.
