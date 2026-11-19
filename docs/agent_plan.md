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

## Java agent run (2026-11-19; job 3741614, 59 cases, 11 min on one H200; docs/results/java_agent_results.txt)
- Set-up: Qwen3-Coder + mini-SWE-agent in the offline namespace; Maven offline through MAVEN_ARGS and a shared repository filled by `prepare_cases_java.py` (broken test online) and `warm_cases_java.py` (repaired method online, because only a test that actually runs downloads the surefire provider); JDKs/Maven from the cluster software directory made visible in the sandbox. First trial (3 cases) found the surefire gap; the first workspace design (revert all changed test files) broke compilation of the whole module, so the Java workspace breaks only the target method and records how many OTHER test lines the repair changed (only 4 of 59 cases changed none).
- Results (checked plausible@1, weighted, 59 cases): AGENT strict 47.4%, AGENT relaxed 57.3%; prompt-only: Qwen3-Coder 41.7%, gpt-oss-20b 36.7%, ernie-5.5-flash 50.9%, nemotron-3-ultra 55.1%, particle-1 66.9%. Strict scoring loses 25 of 59 cases to rule violations (17 edits outside the method, 8 changes to other tracked files); of the 34 scored, 29 pass (85%). Relaxed scoring (whole test file as candidate) gives 35/59.
- Reading: the agent beats the same model prompt-only by ~16 points (relaxed), is on par with the stronger free models, and stays below particle-1; as on Python, its weakness is discipline (editing more than the method), not repair ability. The comparison is confounded the other way as well: the prompt-only Java models had no failure output, the agent sees compile errors by running the test.

## Step limit 40 -> 80 (2026-11-19; jobs 3748197 Java, 3748198/3748199 Python; docs/results/agent_step_limit_results.txt)
- Same cases, prompt, model and temperature; 64k context instead of 32k; workspaces reset first. Call-limit exits fell from 7/59 to 0 (Java) and 22/62 to 6/73 (Python); context-window errors from 2 and 5 to 0.
- JAVA (59 cases): strict 47.4% -> 63.4% (paired +14/-5), relaxed 57.3% -> 70.3% (+11/-4); relaxed 80-call is on par with particle-1 (66.9%; paired +11/-10) and above nemotron-3-ultra (55.1%), qwen3coder prompt-only (41.7%). Rule failures 25 -> 12; no agent changed another tracked file.
- PYTHON (sibling-safe, 61 cases judged in both runs): 46.1% -> 48.0% (paired +10/-8), i.e. no measurable gain; still below particle-1 with the failure output (57-58%). Rule failures unchanged (12 -> 12).
- Reading: more calls matter where the task needs build/test iterations (Java compile-test loops); on Python the 40-call agent was already finishing most cases, and the remaining failures are about discipline (editing more than the function, other files), not about running out of calls. The next lever is the prompt/harness (forbid edits outside the target, reject them, or score the diff), or a stronger model.
