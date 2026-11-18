# WP5 level 3: an open-source coding agent with full repository access (preparation, written 2026-11-18; NOTHING HAS BEEN RUN)

## Question
The ablation (docs/ablation_plan.md) adds fixed context to a single prompt. Does letting a model explore the repository and run tests itself help
much more, and at what cost in tokens and time? This is the upper bound of "better context" for a lesser model.

## Choice of agent
mini-SWE-agent (github.com/SWE-agent/mini-swe-agent): a very small agent whose only tool is a bash shell; talks to any OpenAI-compatible endpoint (LiteLLM),
including a local vLLM server; no Docker needed. Chosen because it is simple to sandbox, its trajectories are easy to audit and cost-account, and it is
the minimal baseline against which heavier agents (SWE-agent, OpenHands, Aider) could later be compared. The project is now at version 2 (a migration guide
exists); config paths and CLI flags differ between versions. **Before the first run: read the current docs (mini-swe-agent.com) and `mini --help`,
and pin the version in the environment.**

## Pieces that exist (tested, no network, no agent started)
- `data_processing/agent_workspace.py` (+ `tests/test_agent_workspace.py`, 8 tests):
  - `prepare_workspace(row, dest)`: project at the REPAIRED commit (the source change is in place), the test's repaired span put back to the broken lines,
    `.git` removed and re-initialised as ONE snapshot commit without remote: no history, branches or tags can reveal the repaired test.
  - `task_prompt(row)`: repair only the named test function, do not weaken the test, no git history / network / installs, stop when it passes.
  - `collect_result(row, dest)`: final test function text, files touched, whether anything outside the test function changed.
  - `audit_trajectory(commands)`: flags `git log/show/fetch/...`, `curl`, `wget`, `pip install`, github/pypi URLs, `.git/` access.
- The vLLM + Slurm conventions of slurm/run_ablation_qwen.sbatch (one job: server and loop on the same node).

## Still to build (in this order, after the ablation runs finish)
1. Environment: a separate venv (`~/venv_agent`) with the pinned mini-swe-agent; no other packages needed on the compute node.
2. vLLM server for Qwen3-Coder-30B-A3B in the job with the OpenAI-compatible API (tool-call parser flags only if the agent version uses function calling;
   the bash-only v1 agent parses a code block from the reply). Config of the agent: model name with the `hosted_vllm/` prefix, `api_base`
   `http://localhost:8000/v1`, step limit (e.g. 30), per-command timeout, cost/step logging. A LiteLLM model registry entry may be needed for a local model.
3. Runner script `scripts/agent/run_cases.py` (to write): for each case of the nested sample (same cases as the ablation: first 100 of the seed-7 shuffle):
   workspace in the job's scratch dir, project dependencies as the plausibility runner builds them (so the agent can run the test), run the agent with the
   task prompt, save trajectory + `collect_result` + `audit_trajectory` + token/step counts, delete the workspace. Resumable per case.
4. Scoring: build a derived split where the reference is the WHOLE repaired test function and each agent's final function is the candidate
   (`plausibility_test.py` then runs it unchanged, including the control). "Needs the span" comes from the normal runs' span control of the same case; the
   assertion check compares the number of assertions in the agent's function with the repaired one (a function-level candidate cannot use the snippet rule).
   Cases where the agent changed other files or code outside the function are reported separately and not counted as plausible repairs.
5. Report: plausible@1 per group and weighted, tokens and steps per case, cost per solved case, vs A/B/C/BC of the ablation, and the share of
   trajectories flagged by the audit (flagged cases are excluded or reported separately).

## Risks and open points
- Leakage through the network: compute nodes can reach github.com; the instruction forbids it and `audit_trajectory` flags it afterwards, but nothing blocks it.
  A stricter option is a proxy variable pointing to a dead port or a network namespace if available (to test).
- Third-party code runs in the sandbox: scratch space only, no secrets in the environment (no tokens in env vars), time limits, kill the process group.
- A local 30B model in an agent loop may be weak at long tool use; a negative result is still a result (report steps, failure modes).
- The environment problem of the Python projects (missing modules) limits the agent too: it cannot run tests that cannot run in the plausibility environment.
- Cost accounting: local GPU time per solved case, not dollars; for comparison with remote models record tokens as well.
