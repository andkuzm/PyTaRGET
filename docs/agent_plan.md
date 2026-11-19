# WP5 level 3: an open-source coding agent with full repository access (written 2026-11-18; first run 2026-11-19, see Results)

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
0. Prepare the workspaces and project environments in a step WITH internet (login or prep job), then run the agent job offline (see the risks section).
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
- Leakage through the network: the agent job does not need the internet, so it runs OFFLINE. Tested on 2026-11-18 on a compute node (artemis18): inside
  `unshare -rn` (unprivileged private network namespace; `user.max_user_namespaces` is large) github.com is unreachable and loopback works after
  `ip link set lo up`. Plan: start BOTH the vLLM server and the agent inside one `unshare -rn bash -c '...'` in the Slurm job (model weights from the local cache with
  HF_HUB_OFFLINE=1; workspaces and project environments prepared beforehand, which are the only steps that need the network). Then `git fetch`, `curl` and
  `pip install` fail by themselves and `audit_trajectory` is only a check. To verify at the first run: vLLM starts inside the namespace (no NCCL/ZMQ
  surprises for one GPU), the job's Slurm environment still works, and a probe command like `curl https://github.com` fails inside the agent's shell.
- Third-party code runs in the sandbox: scratch space only, no secrets in the environment (no tokens in env vars), time limits, kill the process group.
- A local 30B model in an agent loop may be weak at long tool use; a negative result is still a result (report steps, failure modes).
- The environment problem of the Python projects (missing modules) limits the agent too: it cannot run tests that cannot run in the plausibility environment.
- Cost accounting: local GPU time per solved case, not dollars; for comparison with remote models record tokens as well.


## Results of the first run (2026-11-19; job 3737497 + 3741162, 80 cases, ~17 min of one H200 in total)
- The pieces worked as designed: vLLM started inside the private network namespace in ~80 s, the in-job network probe printed 000, the sandbox showed only the case workspace, and the agent used tool calls (median 27 calls, 46 s per case; 57 submitted, 21 hit the 40-call limit, 2 exceeded the context window).
- Rule violations: 13 cases left no test function, 10 edited code outside the test function, 1 changed another tracked file (excluded). 12 cases were flagged by the audit, all `pip install pytest-asyncio` (fails offline) or `git checkout <file>` inside the snapshot repository; none tried to fetch the answer.
- Checked plausible@1 on the 71 cases common to all columns: agent 48/71 (weighted 65.2%) against Qwen3-Coder with the prompt-only context B 32/71 (37.3%), BC 34/71 (41.0%), baseline A 20/71 (21.9%) and particle-1 BC 47/71 (60.4%).
- CONFOUND found when splitting by sibling tests: with other cases of the same repository+commit in the subset the agent solves 29/36 (Qwen B 15/36), without a sibling 19/35 (Qwen B 17/35, particle BC 25/35). The agent workspace is the repaired commit, in which the other tests fixed by the same commit are already repaired, so the agent can copy the fix pattern; the prompt-only runs never see them. The agent's advantage is therefore NOT established.
- Control prepared: `prepare_workspace(..., sibling_safe=True)` reverts ALL test files that the repair changed to their broken-commit version (new test files removed), keeping the repaired source; `scripts/agent/prepare_cases.py --sibling-safe --root ~/agent_work2 --venvs-from ~/agent_work` builds it. Needs a rerun of the agent job on that root (`ROOT=$HOME/agent_work2 OUT=$HOME/eval_agent/qwen3coder_sibling_safe sbatch slurm/run_agent_qwen.sbatch`, ~15 min) and scoring with score_layout.py / plausibility_test.py, then agent_vs_ablation.py with the new result directory.

## Results of the sibling-safe control (2026-11-19; job 3741384, 62 prepared cases, 13:45 min on one H200; docs/results/agent_sibling_safe_results.txt)
- Workspaces with ALL test files changed by the repair reverted to their broken version (the repaired source stays). 35 submitted, 22 hit the 40-call limit, 5 exceeded the context window; median 33 calls, 52 s per case. Rule failures: 8 without a test function, 4 changed other tracked files (excluded), 14 edited outside the function; 21 trajectories flagged by the audit.
- Checked plausible@1 on the 54 cases common to all columns (weighted): agent 50.5% (plain 57.4%); Qwen3-Coder A 20.9%, B 36.8%, BC 39.0%; particle-1 B 57.1%, BC 56.0%. So copying repaired sibling tests explained about 15 of the 65% of the first run; the agent still adds ~14 points over Qwen3-Coder with the failure output (paired +17/-12 against B) but does not beat the larger free model with B/BC context.
- Group pattern: the agent is weaker than Qwen B on the solved group (46.4 vs 67.9) and much stronger on the unsolved group (53.8 vs 11.5): it finds repairs no earlier model found, but misses some easy ones (it often changes more than needed or hits the call limit).
- The sibling split (more passes for cases with siblings: 17/26 vs 10/28) remains although siblings are hidden, so it reflects easier cases and not copying alone.
- Conclusion for the thesis: failure output in a prompt is the cheapest big gain; an agent adds ~14 points over the same small model at ~30 model calls per case (about 50 s on one GPU) and unlocks different cases, but a bigger prompt-only model with the failure output is as good. Evidence is small (54 cases).
