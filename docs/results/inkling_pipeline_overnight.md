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
Solo mean in brackets (n/a: the solo transcripts of older runs are gone). Strong-arm tokens except the last column.

| run | strict | output tok (solo) | calls (solo) | total input (solo) | fresh input (solo) | work (solo) | tools/repair ended | delegate |
|---|---|---|---|---|---|---|---|---|
| ledger_dev r20 | yes (55/55) | 1388 (4194) | 5 (5.5) | 240971 (n/a) | 50798 (n/a) | 25406 (39764) | ESCALATE | 1 agent run(s): 9 calls, 47 s, 100k in / 5.8k out |
| rules_exc r21 | yes (56/56) | 545 (900) | 3 (6) | 138193 (n/a) | 27445 (n/a) | 8833 (26186) | ACCEPT | 1 agent run(s): 7 calls, 10 s, 31k in / 0.8k out |
| gson_a80 r22 | yes (1198/1261) | 558 (804) | 3 (4) | 137485 (n/a) | 26845 (n/a) | 8125 (22979) | ACCEPT | not called (no-model fixes only) |
| gson_cbad r23 | NO DefaultTypeAdaptersTest,JsonReaderTest,NamingPolicyTest,ReflectionAccessTest (1194/1257) | 914 (1990) | 4 (6) | 185545 (n/a) | 27719 (n/a) | 13061 (84716) | ACCEPT | 1 agent run(s): 11 calls, 39 s, 121k in / 1.9k out; audit flags 2 |
| shop r24 | NO MoneyTest (34/34) | 517 (3224) | 3 (4) | 138128 (n/a) | 27460 (n/a) | 8717 (36803) | ACCEPT | 1 agent run(s): 7 calls, 37 s, 69k in / 4.5k out |
| shop r25 | yes (37/37) | 1095 (3224) | 4 (4) | 188874 (n/a) | 30114 (n/a) | 16406 (36803) | ESCALATE | 1 agent run(s): 7 calls, 43 s, 78k in / 5.1k out | (after fix R1)
| real_ta4j r26 | yes (312/312) | 593 (1272) | 3 (6) | 138440 (n/a) | 27796 (n/a) | 9077 (43364) | ACCEPT | 1 agent run(s): 14 calls, 25 s, 203k in / 2.0k out |
| real_cactoos r27 | yes (1282/1286) | 1506 (4856) | 8 (12) | 377784 (n/a) | 28870 (n/a) | 32816 (164192) | ACCEPT | 1 agent run(s): 29 calls, 688 s, 531k in / 7.4k out; audit flags 2 |
| real_logicng_b r28 | NO LiteralTest (573/573) | 11048 (19838) | 24 (58) | 2031027 (n/a) | 88800 (n/a) | 996099 (3913678) | ESCALATE, round rolled back | 1 agent run(s): 91 calls, 250 s, 5764k in / 25.5k out |
| rules_logic r29 | yes (56/56) | 449 (1126) | 3 (5) | 138478 (n/a) | 48212 (n/a) | 9118 (26798) | ACCEPT | 1 agent run(s): 8 calls, 16 s, 62k in / 1.6k out |
| rules_rename r30 | yes (56/56) | 438 (842) | 3 (5) | 138032 (n/a) | 27273 (n/a) | 8672 (19272) | ACCEPT | not called (no-model fixes only) |
| rules_big r31 | yes (56/56) | 444 (1014) | 3 (5) | 138408 (n/a) | 27556 (n/a) | 9051 (30594) | ACCEPT | 1 agent run(s): 8 calls, 24 s, 64k in / 2.8k out |
| real_tablesaw r32 | yes (750/755) | 463 (1065) | 3 (6) | 138130 (n/a) | 27482 (n/a) | 8764 (47222) | ACCEPT | 1 agent run(s): 14 calls, 36 s, 200k in / 2.0k out; audit flags 1 |
<!--rows-->

## Regressions and fix attempts
### R1. shop (run r24, Inkling agent): a test was deleted and accepted
- **What regressed:** strict score NO (`MoneyTest` weakened) although 34/34 tests pass; every earlier shop run (solo, S1, S4, S5, S6 particle-1/oracle) passed strict. Strong-arm cost was low (517 output tokens, 3 calls).
- **How:** the agent DELETED `MoneyTest#convertsToDouble` (its target `toDouble()` no longer exists), against its instruction not to delete tests. The real repair keeps the test and rewrites it to `assertEquals(250L, new Money(250).cents())`.
- **Why it was not caught (theory, confirmed):** shop workspaces were prepared before `tools/review` existed, so `tools/repair` printed ACCEPT with "assertion check not done". Running the review on the r24 workspace flags `MoneyTest.java#convertsToDouble [REMOVED]`.
- **Fix attempted:** launcher installs `tools/review` into EVERY run (it was only in workspaces prepared later). This also covers the gson workspaces. It does not change the delegate. Note: earlier shop and gson runs (all delegates) ran without this check.
- **Rerun (r25): strict yes, 37/37, 1095 output tokens (solo 3224), 4 calls.** HONEST READING: the pass is not caused by the fix: this time the agent did not delete the test (it rewrote it as `exposesCents`, asserting `cents()`); the run-to-run behaviour of the delegate differs. What the fix demonstrably does: the review ran, flagged `itemsAreReturnedInOrder [REMOVED]` (a rename to `linesAreReturnedInOrder`, a false positive), and the strong agent checked it (+578 output tokens against r24). On the r24 workspace the same review flags the real deletion. Whether the strong agent repairs a real deletion when flagged has not been observed.
- **Residual risk:** the delegate can still delete tests; detection now relies on the review's rename matching.

### R2. real_logicng_b (r28): agent delegate ran out of context; weak tests by the strong agent
- **Not a cost regression** against solo (11048 vs 19838 output tokens, 24 vs 58 calls, work 996k vs 3914k; particle-1 S6: 12567 / 19). Strict NO = LiteralTest, as in every run of this repo including solo; 573/573 pass.
- **Delegate failure:** the agent made 91 calls in 250 s (5.76M input tokens) and ended with ContextWindowExceededError (server limit 131072). It had changed 35 files; the round counted 84 compile errors before and 90 after, so `tools/repair` rolled it back and the strong agent repaired everything. The delegate contributed nothing; the saving over solo is NOT from delegation.
- **Test quality (strong agent, same pattern as particle-1 run r11):** testIllegalAdd and testIllegalCC/AMO/EXO now force a negative literal by a cast and expect ClassCastException (they test a cast in the test, not the library); three SubformulaTest assertions flipped to false. tools/review did not flag this. The reference has no methods of these names.
- **Caveat on the rollback rule:** javac stops after 100 errors and the count is not monotonic, so 84 -> 90 does not prove the round made no progress.
- **Fix attempt:** (see below) server context 131072 -> 262144, rerun after the 5-hour window reset.

