# Real-repo runs (TaRBench commit pairs), first results 2026-11-21 evening

Tasks verified (repaired tests pass, old tests fail): cactoos (1319 tests), LogicNG pair b (570), ta4j-origins (312), tablesaw core (776), sonar-cxx checks (820). Preparation failed for cucumber-jvm and LogicNG pair a (scratch directory error in the clone step: to be retried) and mockserver-core (build problem). Runs so far: cactoos r1, ta4j r1-r2, tablesaw r1-r2 (solo, routed S4 + review, delegate-all S1 + review); LogicNG b r1 was started but its agents had not handed back when this was written. Runs where the delegate was unavailable (HTTP 402 / 429) are excluded from the delegated arms.

| repo | arm | valid runs | success | output tokens (mean) | model calls (mean) | added context (mean) | work (mean) | delegate $ (mean) |
| real_cactoos | S1 | 0 of 1 | - | - | - | - | - | - |
| real_cactoos | S4 | 0 of 1 | - | - | - | - | - | - |
| real_cactoos | solo | 1 (of 1) | 1/1 | 4856 | 12.0 | 22540 | 164192 | 0.0000 |
| real_ta4j | S1 | 1 (of 2) | 1/1 | 3075 | 18.0 | 17836 | 174386 | 0.0129 |
| real_ta4j | S4 | 2 (of 2) | 2/2 | 1462 | 6.5 | 12549 | 55706 | 0.0000 |
| real_ta4j | solo | 2 (of 2) | 2/2 | 1272 | 6.0 | 11662 | 43364 | 0.0000 |
| real_tablesaw | S1 | 1 (of 2) | 1/1 | 2174 | 11.0 | 9808 | 64737 | 0.0000 |
| real_tablesaw | S4 | 2 (of 2) | 2/2 | 1088 | 6.0 | 12217 | 50255 | 0.0000 |
| real_tablesaw | solo | 2 (of 2) | 2/2 | 1065 | 6.0 | 12142 | 47222 | 0.0000 |


## Reading (preliminary, n small)
- The changes in 3 of the 4 finished real tasks are one mechanical API change each: ta4j (`new TradingRecord(...)` -> `new BaseTradingRecord(...)`, `TradingRecord` became an interface, 15 files), tablesaw (`minimizeColumnSizes(true)` -> `minimizeColumnSizes()`, 28 calls in 5 files). Solo and routed do these with one `sed` (5-6 calls, about 1.1-1.3k output tokens); delegate-all costs more calls (11-18) and about 2x output tokens for the same result. Same picture as the generated repo set: a uniform rename is the strong arm's cheapest job.
- cactoos is the only heterogeneous case so far: a library upgrade removed the one-argument `MatcherOf(lambda)` constructor. Solo wrote a helper matcher in about 12 calls / 4.9k output tokens; the routed and delegate-all agents could not use the delegate (HTTP 402) and made 19 calls / 5.3k-8.3k output tokens; one of them worked around it with a perl one-off that adds the two missing arguments (a valid repair with no helper), one wrote a helper like solo. These runs are NOT valid delegation runs.
- Selection finding: the TaRBench commit pairs with many failing tests are overwhelmingly mechanical API refactors, which is what the strong arm scripts; pairs with heterogeneous logic changes are rarer and need a different filter (distinct repair shapes per test file, not just test counts).

## Status of the delegate (important for the next runs)
- The OpenRouter account is essentially out of credit (account usage 9.948 of 10 USD at 18:55; the paid fallback returned HTTP 402 to agents, one trivial call to the paid model still worked), my own fallback cap 0.30 USD is exceeded (0.316 spent), and the free particle-1 tier is rate-limited upstream (429) for long stretches. So delegated arms are currently unreliable; do not run more of them until the user decides about credit (a top-up, a different free model, or a local model via Slurm, which the user allowed: at most one job, with a wake-up).
- Weekly Claude usage reached about 99% at 18:58 (the weekly limit resets at 19:00).

## State at the graceful stop (2026-11-21 19:05; user: "basically stop gracefully")
- Weekly usage had just reset (0% at 19:02); the user's cap for this phase was 20% of the week. No agents are running. No crons are scheduled.
- LogicNG pair b, r1 (3 agents, started 18:36): CORRECTION of an earlier note - they did finish (about 30-35 minutes each; their transcripts are only written when the agent ends, so "no progress" was a wrong inference) and are scored. See the LogicNG section below. Their run directories exist; the pair counts as done for repetition 1.
- The six runs prepared at 19:03 (sonar-cxx r1 and cactoos r2, three arms each) were never started and were removed; `next_real.py` will offer them again.
- Free-tier delegate answered at 19:02 (a trivial call took 5.5 s); paid credit is nearly exhausted (see above); the fallback cap in the run configs is 0.30 USD and already exceeded.
- To resume: `python scripts/delegation/select/next_real.py` -> `make_batch.py <task> <rep>` -> launch the three agents with the pointer prompt -> `batch_report.py <run> <agent> ...` -> `aggregate.py`. Verified tasks: cactoos, logicng_b, ta4j, tablesaw, sonarcxx. To fix first: preparation of cucumber-jvm and LogicNG pair a (scratch directory error in the clone step), mockserver-core (build problem); decide the delegate (credit / other free model / local model via one Slurm job with a wake-up).


## LogicNG pair b (the first heterogeneous real case), repetition 1
API change in LogicNG: `Literal` -> `Variable` typed signatures, `literal(...)` -> `variable(...)`, `contains` -> `containsVariable`/`containsNode` (new node-based semantics), `SubformulaFunction` -> `SubNodeFunction`, `getSubstitution` removed. About 30-34 test files change, plus real semantic edits (flipped expectations in SubformulaTest, reworked SubstitutionTest, tests that can no longer be expressed).

| Run | Tests pass | Model calls | Output tokens | Work (extra input tokens) | Delegate | Strict success |
|---|---|---|---|---|---|---|
| solo | 569/569 | 58 | 19.8k | 3.9M | none | no (see below) |
| S4 + review | 569/569 | 54 | 22.0k | 2.9M | HTTP 402, unusable | no |
| S1 + review | 570/570 | 51 | 20.3k | 2.7M | 1 unit applied, then HTTP 402 | no |

- All three finish with a passing suite but fail the strict criterion: 3-4 tests were deleted (`PBConstraintTest#testIllegalCC/AMO/EXO`, `SubstitutionTest#testIllegalAdd`) because the new Variable-typed API makes negative literals / mixed factories impossible to express; `tools/review` flags exactly these as REMOVED and LiteralTest has fewer assertions than the reference. The reference repair (the real commit) did the same kind of thing for some of them, so the strict rule is too tight for this task: it needs a per-case judgement (the real reference tells which removals are legitimate).
- This is where delegation could matter (about 20k output tokens and 50+ calls per run, mostly hand-written adaptation), but the delegate was out of credit during all three runs, so none of them tests delegation. First priority after the delegate is available again: rerun this task with S4/S1.

## Local delegate (Qwen3-Coder-30B-A3B via vLLM, Slurm job 3847312, H200): reruns on the two heterogeneous real tasks (2026-11-22)
Delegate = `local` alias (the server answers in about 0.1-1 s per call; a full ledger repair of 15 units in 30 s; zero cost, no rate limit). Repetition 3 of the routed (S4) and delegate-all (S1) arms, same prompts as before; solo = repetition 1. One run per cell.

| Task | Arm | Tests | Model calls | Output tokens | Work (extra input tokens) | Delegate units applied |
|---|---|---|---|---|---|---|
| cactoos (library upgrade removed `MatcherOf(lambda)`, 20 test files) | solo | 1282/1286 ok | 12 | 4.9k | 164k | - |
| cactoos | S4 | 1282/1286 ok | 19 | 7.0k | 338k | 30/31 |
| cactoos | S1 | 1282/1286 ok | 18 | 7.2k | 299k | 66/71 |
| LogicNG b (Literal -> Variable API, 24-34 files, real semantic edits) | solo | 569 pass, strict criterion NOT met | 58 | 19.8k | 3.9M | - |
| LogicNG b | S4 | 573 pass, not met | 48 | 16.4k | 2.3M | 31/34 |
| LogicNG b | S1 | 569 pass, not met | 52 | 21.0k | 2.3M | 31/33 |

- cactoos: the delegate HURT. Both delegated agents reported useless or wrong edits from the local model (hallucinated import `org.cactoos.matchers.HasToString`, casts, stray ```java fence lines in the files), reverted them with `git checkout -- src/test` and wrote the same helper matcher solo wrote; so delegation cost 40% more output tokens and 6-7 more calls for the same result. The "applied" counts are units written, not units that were useful. A patch that strips stray fence lines from answers was applied afterwards (not re-measured).
- LogicNG: routed (S4) used 17% fewer output tokens than solo (16.4k vs 19.8k), 10 fewer calls and 40% less work, and was the only run that kept all tests (it kept the four expected-exception tests as positive checks instead of deleting them; the other three deleted 3-4 tests). Delegate-all (S1) was no cheaper than solo in output tokens (21.0k). Both agents report the delegate made wrong edits (negative literals turned positive, weakened assertions in `PBConstraintTest`) that they repaired or restored by hand; the S4 agent did not re-check every `.variable()` edit that `tools/review` counts as a rename.
- Strict success is not met by any LogicNG run (deleted tests / fewer assertions in `LiteralTest`): the criterion needs per-case judgement against the real reference for this task.
- Reading (n=1 per cell, treat as a signal): with a delegate that is free, instant and reliable to call, a weak local model (30B-A3B) still produces edits that the strong agent has to check and sometimes revert; the saving shows up only where the delegate's mistakes are cheap to catch (LogicNG routed) and disappears where one wrong approach spoils whole files (cactoos). The review step is what makes the strong agent notice; the delegate model's quality, not its speed or cost, is now the limit.
