# Plan: delegation strategies on real repositories (prepared 2026-11-21, to run after the weekly reset at 19:00)

## Why
The generated repo set (docs/results/delegation_repo_set.md) was all small, uniform fixes: solo scripts the repair and wins. This phase uses REAL commit pairs from TaRBench with many changed test methods in several files and heterogeneous repairs, so that the strong arm cannot repair everything with one script.

## Cases
Candidate scan: `scripts/delegation/select/scan_tarbench.py` -> ~/delegation_exp/tarbench_pairs.csv (8097 commit pairs; 542 with >= 15 tests in >= 3 files). Handpicked, single-module pairs from different projects (spread of size, build and domain): LogicNG (two pairs, 38 and 122 tests), ta4j-origins (42 tests), cactoos (30), tablesaw core (27), cucumber-jvm core (27), mockserver-core (31), sonar-cxx checks (37). Built and verified by `scripts/delegation/select/prep_real_batch.py` (status in ~/delegation_exp/real_prep.json: the repaired tests must pass offline and the old tests must fail); only verified tasks are used. Check the status first: `cat ~/delegation_exp/real_prep.json`, and `tail ~/delegation_exp/scratch/prep_real.log`.

## Runs
Per verified task, 3 arms x 2 repetitions: solo, S4+review (strong agent does plain replaces, delegates logic), S1+review (delegate does everything, strong agent runs/reviews). Same prompts as the repo-set phase (agent reads its prompt file; no developer handoff for real tasks, as for gson). Order: `python scripts/delegation/select/next_real.py` prints the next (task, rep); create the three runs with `python scripts/delegation/make_batch.py <task> <rep>`; launch the three agents (Agent tool, model sonnet, prompt = the pointer text "Your complete instructions are in the file <run dir>/prompt.md ... "); register ids with `python scripts/delegation/batch_report.py <run id> <agent id> ...` (it scores a run only after the agent's SubagentHandback call); aggregate with `python scripts/delegation/aggregate.py` (per repo and arm).
- ONE task (3 agents) at a time: the free OpenRouter tier returns HTTP 429 under parallel load; a run where the delegate was unavailable is excluded (flag in batch_report) and rerun.
- Delegate paid fallback cap 0.30 USD (about 0.23 spent); ask the user before raising; the free tier is tried first.
- Strong-arm sizes are larger on real tasks: expect 1-3 usage points per cell; read `python scripts/delegation/usage.py` before every batch; caps this time: window 70%, week 95% (user, 2026-11-21).
- After each batch: look at `tools/review` flags and the scorer's `weakened`/`shrunk`; real tasks have a reference (the real repaired tests): the scorer compares assertion counts per class.

## Metrics for the decision
Success (compile, all pass, nothing shrunk/weakened); strong-arm output tokens, model calls, added context and work (batch_report); delegate cost and applied/rejected units; wall time. Pooled and per repo; at least 2 valid runs per cell.

## Continuity
State lives in files (real_prep.json, batch_results.json, agent_map.json), not in the conversation. A wake-up cron at 19:02 resumes the queue (see the cron prompt); stop when the week reaches 95% or the window 70%.
