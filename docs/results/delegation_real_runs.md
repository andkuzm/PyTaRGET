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
