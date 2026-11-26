# Overnight run: repair pipeline (S6) with Inkling-2-Mini as delegate, Claude Sonnet 5.5 as strong model - started 2026-11-26

## Rules agreed with the user
At most 2 own Slurm jobs at once; agent delegate first (clear winner, docs/results/inkling_agent_full.md), single prompt second if limits allow; agent only inside an isolated no-network sandbox;
on a regression: if there is a clear theory and clean fix, attempt it and rerun that repo, then go on; if not, go on; document each. Only clear bug fixes. 5-hour window cap 70 %, weekly caps per scripts/delegation/usage.py.
No paid OpenRouter calls. Regression = strict score fails where earlier runs passed, tests weakened, or more strong-arm output tokens than the solo baseline (calls, total/fresh input and work are reported too).

## Agent delegate (new, 2026-11-26 evening)
slurm/agent_server.sbatch runs vLLM and a worker (scripts/agent/delegate_worker.py) inside `unshare -rn` (no network). tools/delegate (alias inkling-agent) passes requests through ~/delegation_exp/agent_queue.
The agent works on a clean copy of the run workspace (no tools/, no .delegation, no history) inside the mount sandbox (no home, no reference repair), with a private copy of the Maven cache; only changed or new
files under the test roots are copied back; deletions and changes outside are discarded and reported. Step limit 150, 30 min per request. tools/start and the literal sync still run first; per-unit routing does not apply to the agent.

## Runs
(table filled after each repo)

## Regressions and fix attempts
(none yet)
