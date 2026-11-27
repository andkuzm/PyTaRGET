# Overnight run: repair pipeline (S6) with Inkling-2-Mini as delegate, Claude Sonnet 5.5 as strong model - started 2026-11-26

## FINAL SUMMARY (2026-11-27)

### Regressions and fix attempts (details at the end of this file)
| id | what regressed | cause | attempted | outcome |
|---|---|---|---|---|
| R1 shop (r24) | strict fail: the delegate agent DELETED `MoneyTest#convertsToDouble`; accepted | shop workspaces had no tools/review | review installed in every run | reruns r25, r46 passed because the agent did not delete; in r59 it deleted again, the review flagged it and the strong agent restored it (the restored test calls no production method). Delegate deleted in 2 of 4 agent shop runs: DETECTION fixed, delegate behaviour not |
| R2 logicng_b (r28) | delegate contributed nothing: context overflow after 91 calls, round rolled back | server context 131072 | 262144 | fixed: r33, r52, r65 finished (103-135 calls) with all tests passing |
| R3 logicng_b (r33) | hollow tests (`assertFalse(x instanceof Variable)`) accepted | two review bugs (assertFalse counted as an expected exception; a change of exception type was invisible) | new flag EXPECTED EXCEPTION LOST | detection works (r34, r52, r65 ESCALATE); the RESULT is not reliably better: in r34 the strong agent asserted real behaviour, in r52 and r65 it wrote `expected = ClassCastException` raised by the test's own cast (review is not rerun after hand edits). All 3 SOLO runs and the single-prompt run ended with that same form. No faithful repair exists (the new types remove the behaviour; the reference drops the tests). NOT SOLVED |
| D1 cactoos (r63), delay only | 30-minute delegate limit hit, 14 calls vs solo 8 | the agent ran `find /` twice (600 s each) | none (would change the delegate's instructions) | open decision; strict pass, output below solo |
Not regressions but fixed or found on the way: new solo runs would have received the automation tools (fixed before any solo repeat); two scoring processes at once gave a false 541/573 (score one at a time); gson_cbad (4 classes) and logicng_b (LiteralTest) fail the strict score in EVERY arm including solo.

### Results (per-repo means, summed over repos; strong-arm Claude Sonnet 5.5 tokens only)
```
pipe: over 12 repos (sum of per-repo means; solo = transcript/recorded runs of the same repos)
  out       solo      41345  pipe      15214  saving of sums  63.2% | per-repo saving: median  57.7%, min  43.1%, max  77.8%
  calls     solo        113  pipe         66  saving of sums  41.7% | per-repo saving: median  46.1%, min -25.9%, max  52.9%
  total_in  solo    7242432  pipe    3282526  saving of sums  54.7% | per-repo saving: median  46.8%, min -35.5%, max  67.4%
  fresh_in  solo     576236  pipe     429018  saving of sums  25.5% | per-repo saving: median  26.4%, min -31.9%, max  49.4%
  work      solo    3428440  pipe     450937  saving of sums  86.8% | per-repo saving: median  76.2%, min  50.0%, max  89.7%
  without logicng_b: out 63.5%, calls 37.1%, total_in 41.6%, fresh_in 19.6%, work 74.5%
single: over 10 repos (sum of per-repo means; solo = transcript/recorded runs of the same repos)
  out       solo      39269  single      25950  saving of sums  33.9% | per-repo saving: median  48.3%, min   9.6%, max  75.2%
  calls     solo        102  single         69  saving of sums  32.5% | per-repo saving: median  40.0%, min  11.1%, max  50.0%
  total_in  solo    6728810  single    4533848  saving of sums  32.6% | per-repo saving: median  42.8%, min  -2.8%, max  56.8%
  fresh_in  solo     489287  single     404800  saving of sums  17.3% | per-repo saving: median  12.8%, min   1.9%, max  46.7%
  work      solo    3375666  single    1558589  saving of sums  53.8% | per-repo saving: median  69.8%, min  37.7%, max  84.5%
  without logicng_b: out 52.4%, calls 35.5%, total_in 39.0%, fresh_in 17.3%, work 67.9%
single vs agent pipeline on the 10 repos with a single-prompt run:
  out       agent      14185 single      25950
  calls     agent         60 single         69
  total_in  agent    3007008 single    4533848
  fresh_in  agent     354323 single     404800
  work      agent     434139 single    1558589
runs counted: solo 32 agent 36 single 10
solo strict passes 27 / 32 | passes excluding gson_cbad and logicng_b: 27 / 27
pipe strict passes 30 / 36 | passes excluding gson_cbad and logicng_b: 30 / 30
single strict passes 8 / 10 | passes excluding gson_cbad and logicng_b: 8 / 8
```
Reading: the agent pipeline used 63% fewer strong output tokens than solo (median repo 58%; every repo saved output, 43-78%), 87% less extra input processed, 55% less total input, but only 26% less FRESH (uncached, the expensive) input; calls were fewer on most repos, MORE on at least one (waiting for a slow delegate). The single prompt (1 run per repo, 10 repos) saved 34% of output and on the same repos used 83% more strong output than the agent pipeline; its round was rolled back on cactoos and logicng_b and the strong agent finished by hand on shop and ledger_dev.

### Limits of these numbers
- 12 repos that were ALL used while the tool was developed (several fixes were made on them): expect worse on unseen repos.
- About 3 runs per arm per repo; solo varies a lot (ledger_dev solo runs from about 1.7k-4k in the pilots to 7.9k; shop one solo run 6.9k with a stuck command).
- Solo input-token columns rest on 1-3 runs with transcripts. Means exclude runs superseded by a fix (shop r24 failed strict; logicng_b r28, r33).
- Pipeline fixes R1/R3 changed the review during the night, so early and late pipeline runs are not byte-identical tool versions.
- Quality is checked by the strict scorer and tools/review; after ACCEPT the strong agent does not read the changes. logicng_b shows both can miss weak tests.
- gson_a80 and rules_rename never call a delegate: their saving is the no-model automation alone. No automation-without-delegate control was run.
- Delegate GPU time is not in the table. No paid API spend (ledger unchanged).

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
| real_logicng_b r33 | NO LiteralTest (573/573) | 1782 (19838) | 12 (58) | 572149 (n/a) | 50301 (n/a) | 54685 (3913678) | ACCEPT | 1 agent run(s): 115 calls, 694 s, 11701k in / 43.3k out; discarded changes: 21; audit flags 2 | (after fix R2)
| real_logicng_b r34 | NO LiteralTest (573/573) | 6350 (19838) | 20 (58) | 1125982 (n/a) | 71415 (n/a) | 263542 (3913678) | ESCALATE | 1 agent run(s): 129 calls, 616 s, 11634k in / 43.1k out; discarded changes: 21; audit flags 3 | (after fix R3)
<!--rows-->

## Repeats: solo vs pipeline (Inkling agent), strong-arm means (generated by scripts/delegation/repeat_summary.py)
Solo runs of tonight get only tools/run_tests (as the original solo runs). 'earlier pilots' = old solo numbers without transcripts. Runs superseded by a fix are excluded from the means.

| repo | arm | strict pass | output tok | calls | total input | fresh input | work |
|---|---|---|---|---|---|---|---|
| gson_a80 | solo | 2/2 | 1190 (1074-1307, n=2) | 5.5 (5.0-6.0, n=2) | 266.5k (242.4k-290.6k, n=2) | 52.8k (52.6k-53.0k, n=2) | 29.4k (26.9k-31.9k, n=2) |
| gson_a80 | pipeline (Inkling agent) | 3/3 | 551 (545-558, n=3) | 3.0 (3.0-3.0, n=3) | 137.5k (137.5k-137.5k, n=3) | 40.6k (26.8k-47.4k, n=3) | 8126 (8125-8126, n=3) |
| gson_cbad | solo | 0/2 | 2298 (2173-2423, n=2) | 7.5 (7.0-8.0, n=2) | 408.0k (382.5k-433.4k, n=2) | 52.0k (40.0k-64.1k, n=2) | 84.7k (80.8k-88.5k, n=2) |
| gson_cbad | pipeline (Inkling agent) | 0/3 | 853 (756-914, n=3) | 4.0 (4.0-4.0, n=3) | 185.5k (185.5k-185.5k, n=3) | 27.7k (27.7k-27.7k, n=3) | 13.1k (13.1k-13.1k, n=3) |
| gson_cbad | pipeline (Inkling single prompt) | 0/1 | 1085 (n=1) | 4.0 (n=1) | 185.6k (n=1) | 27.7k (n=1) | 13.1k (n=1) |
| ledger_dev | solo | 2/2 | 5726 (3595-7856, n=2) | 8.5 (5.0-12.0, n=2) | 472.9k (258.8k-686.9k, n=2) | 53.0k (46.3k-59.8k, n=2) | 106.4k (43.3k-169.6k, n=2) |
| ledger_dev | pipeline (Inkling agent) | 3/3 | 1358 (1298-1389, n=3) | 4.3 (4.0-5.0, n=3) | 208.0k (190.2k-241.0k, n=3) | 37.4k (29.6k-50.8k, n=3) | 21.2k (17.7k-25.4k, n=3) |
| ledger_dev | pipeline (Inkling single prompt) | 1/1 | 1418 (n=1) | 5.0 (n=1) | 244.6k (n=1) | 52.0k (n=1) | 29.0k (n=1) |
| real_cactoos | solo | 3/3 | 3768 (3074-4856, n=3) | 9.0 (7.0-12.0, n=3) | 399.6k (363.3k-436.0k, n=2) | 42.4k (41.1k-43.6k, n=2) | 105.6k (61.5k-164.2k, n=3) |
| real_cactoos | pipeline (Inkling agent) | 3/3 | 2061 (1506-2499, n=3) | 11.3 (8.0-14.0, n=3) | 541.5k (377.8k-673.0k, n=3) | 29.7k (28.9k-30.3k, n=3) | 52.8k (32.8k-69.3k, n=3) |
| real_cactoos | pipeline (Inkling single prompt) | 1/1 | 3197 (n=1) | 8.0 (n=1) | 410.7k (n=1) | 39.4k (n=1) | 65.8k (n=1) |
| real_logicng_b | solo | 0/3 | 17.0k (13.2k-19.8k, n=3) | 41.7 (27.0-58.0, n=3) | 3659.2k (2700.2k-4618.1k, n=2) | 114.8k (105.2k-124.4k, n=2) | 2781.2k (1536.2k-3913.7k, n=3) |
| real_logicng_b | pipeline (Inkling agent) | 0/3 | 6324 (5967-6654, n=3) | 21.0 (20.0-22.0, n=3) | 1191.5k (1126.0k-1266.8k, n=3) | 58.1k (51.1k-71.4k, n=3) | 285.9k (263.5k-318.1k, n=3) |
| real_logicng_b | pipeline (Inkling single prompt) | 0/1 | 15.3k (n=1) | 30.0 (n=1) | 2661.2k (n=1) | 95.2k (n=1) | 1367.6k (n=1) |
| real_ta4j | solo | 3/3 | 1374 (1233-1577, n=3) | 6.0 (6.0-6.0, n=3) | 320.2k (n=1) | 43.1k (n=1) | 49.4k (42.4k-61.5k, n=3) |
| real_ta4j | pipeline (Inkling agent) | 3/3 | 584 (575-593, n=3) | 3.0 (3.0-3.0, n=3) | 138.4k (138.4k-138.4k, n=3) | 27.8k (27.8k-27.8k, n=3) | 9066 (9053-9077, n=3) |
| real_ta4j | pipeline (Inkling single prompt) | 1/1 | 771 (n=1) | 3.0 (n=1) | 138.3k (n=1) | 27.7k (n=1) | 8972 (n=1) |
| real_tablesaw | solo | 3/3 | 1064 (1056-1074, n=3) | 5.7 (5.0-6.0, n=3) | 252.1k (n=1) | 34.9k (n=1) | 43.7k (36.5k-47.8k, n=3) |
| real_tablesaw | pipeline (Inkling agent) | 3/3 | 448 (428-463, n=3) | 3.0 (3.0-3.0, n=3) | 138.1k (138.1k-138.1k, n=3) | 34.3k (27.5k-48.1k, n=3) | 8756 (8737-8768, n=3) |
| real_tablesaw | pipeline (Inkling single prompt) | 1/1 | 701 (n=1) | 4.0 (n=1) | 185.5k (n=1) | 27.7k (n=1) | 13.0k (n=1) |
| rules_big | solo | 3/3 | 1096 (980-1259, n=3) | 5.0 (5.0-5.0, n=3) | 244.8k (n=1) | 32.2k (n=1) | 30.1k (29.3k-31.4k, n=3) |
| rules_big | pipeline (Inkling agent) | 3/3 | 469 (444-502, n=3) | 3.0 (3.0-3.0, n=3) | 138.4k (138.4k-138.4k, n=3) | 34.4k (27.6k-48.2k, n=3) | 9058 (9051-9071, n=3) |
| rules_big | pipeline (Inkling single prompt) | 1/1 | 502 (n=1) | 3.0 (n=1) | 138.5k (n=1) | 27.7k (n=1) | 9186 (n=1) |
| rules_exc | solo | 3/3 | 963 (829-1089, n=3) | 5.3 (5.0-6.0, n=3) | 241.2k (n=1) | 31.0k (n=1) | 26.0k (24.3k-28.0k, n=3) |
| rules_exc | pipeline (Inkling agent) | 3/3 | 548 (541-559, n=3) | 3.0 (3.0-3.0, n=3) | 138.2k (138.2k-138.2k, n=3) | 34.3k (27.4k-48.0k, n=3) | 8833 (8833-8834, n=3) |
| rules_exc | pipeline (Inkling single prompt) | 1/1 | 696 (n=1) | 3.0 (n=1) | 138.3k (n=1) | 27.5k (n=1) | 8901 (n=1) |
| rules_logic | solo | 3/3 | 1249 (1118-1496, n=3) | 5.0 (5.0-5.0, n=3) | 242.8k (n=1) | 31.4k (n=1) | 27.0k (26.8k-27.3k, n=3) |
| rules_logic | pipeline (Inkling agent) | 3/3 | 484 (449-528, n=3) | 3.0 (3.0-3.0, n=3) | 138.5k (138.5k-138.5k, n=3) | 41.4k (27.6k-48.2k, n=3) | 9119 (9118-9119, n=3) |
| rules_logic | pipeline (Inkling single prompt) | 1/1 | 528 (n=1) | 3.0 (n=1) | 138.5k (n=1) | 27.7k (n=1) | 9138 (n=1) |
| rules_rename | solo | 3/3 | 886 (783-974, n=3) | 5.0 (5.0-5.0, n=3) | 247.1k (n=1) | 34.1k (n=1) | 23.4k (19.0k-31.6k, n=3) |
| rules_rename | pipeline (Inkling agent) | 3/3 | 478 (438-507, n=3) | 3.0 (3.0-3.0, n=3) | 138.0k (138.0k-138.0k, n=3) | 34.1k (27.3k-47.8k, n=3) | 8673 (8672-8673, n=3) |
| shop | solo | 2/2 | 4754 (2615-6893, n=2) | 8.5 (5.0-12.0, n=2) | 488.0k (256.4k-719.7k, n=2) | 54.6k (50.8k-58.5k, n=2) | 121.6k (40.9k-202.4k, n=2) |
| shop | pipeline (Inkling agent) | 3/3 | 1056 (556-1516, n=3) | 4.0 (3.0-5.0, n=3) | 188.9k (138.1k-239.6k, n=3) | 29.3k (27.5k-30.2k, n=3) | 16.4k (8757-24.0k, n=3) |
| shop | pipeline (Inkling single prompt) | 1/1 | 1709 (n=1) | 6.0 (n=1) | 292.6k (n=1) | 52.2k (n=1) | 33.9k (n=1) |

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
- **Fix attempt:** server context 131072 -> 262144. **Rerun r33: worked.** The agent finished (115 calls, 694 s, 11.7M input tokens, no overflow), all 573 tests passed after its round, tools/repair printed ACCEPT; strong arm 1782 output tokens (r28: 11048; solo: 19838), 12 calls (24; 58). One run; the extra calls are mostly `tools/repair --wait` while the agent ran. The 21 "discarded changes" are files the test suite itself writes under tests/ (logs, .dot files).

### R3. real_logicng_b (r33): the DELEGATE wrote hollow tests and the review missed them
- **What regressed (test quality):** testIllegalAdd and testIllegalCC/AMO/EXO expected an IllegalArgumentException from the library; the agent replaced that by `assertFalse(<negative literal> instanceof Variable)`, a statement about Java types that tests nothing the library does. (Reference: no methods of these names. Earlier runs r11/r28 had the strong agent write the same kind of hollow test with `expected = ClassCastException`.) The strict scorer does not detect it.
- **Why it passed (two confirmed bugs in tools/review):** (1) any failure-style check (here assertFalse) counted as keeping an expected exception; (2) a change of only the expected exception TYPE was treated as "unchanged" (the comparison looked at assertions and literals only).
- **Fix R3:** new flag EXPECTED EXCEPTION LOST (old test expects an exception, new one none; or newly expects ClassCastException, which a cast in the test can raise); the expected exception type is part of the comparison. Replayed on stored workspaces: flags exactly the 4 hollow tests in r28 and r33, no new flags in 9 other runs.
- **Limit of the fix:** it only DETECTS. The new library types make the old situation impossible to express, so there may be no faithful repair; the reference drops these tests, which our rules forbid.
- **Rerun r34 (with R3): worked as a detection fix.** review flagged all 4 tests (FAILURE CHECK LOST; EXPECTED EXCEPTION LOST), tools/repair ended ESCALATE, and the strong agent rewrote them to assert real library behaviour (operands, coefficients, comparator, rhs, stored mappings) with positive inputs, instead of a statement about Java types. They no longer check that bad input is REJECTED (not expressible with the new types; the reference drops these tests). 573/573; strict NO only on LiteralTest (as every run incl. solo). Cost: 6350 output tokens (r33 without the flag: 1782; solo 19838), 20 calls (12; 58).
- **Repeat pair r52 (pipeline) vs solo r52:** pipeline 6654 output tokens / 21 calls, solo 17882 / 40; both 573/573, strict NO only on LiteralTest. The delegate again wrote the four weak tests and review flagged them (ESCALATE); the strong agent "restored" them as `expected = ClassCastException` (raised by a cast or raw list in the test itself) and, as the S6 prompt only asks for a test run after hand repairs, did not run the review again, so the weak form stayed. **The SOLO run wrote the same four ClassCastException tests on its own** (review flags 4 of them there too): this weakness is not specific to the pipeline. In r34 the strong agent instead rewrote them to assert real library behaviour: the outcome varies between runs. Open: rerun review after hand repairs (prompt change, not made: it would change the compared pipeline).
- **Scoring note:** the solo r52 workspace first scored 541/573 while two scoring processes ran at the same time (0 failures, tests missing from the reports); scored alone it is 573/573. Score one run at a time.
- **R1 follow-up (run shop r59):** the delegate DELETED `MoneyTest#convertsToDouble` again; with fix R1 the review flagged it (ESCALATE) and the strong agent RESTORED it (checking `cents() / 100.0` against 2.5; since `toDouble()` no longer exists the test no longer exercises a production conversion). This is the first observation that a flagged deletion gets repaired. 33/33, strict yes, 1516 output tokens. So the delegate's deleting behaviour occurred in 2 of 4 shop runs.
- **Delay observation (cactoos r63, not a cost regression, NOT fixed):** the agent delegate hit its 30-minute request limit (50 calls) because it twice ran `find /` to locate a library jar; each ran into the 600 s per-command limit (the sandbox shows the cluster software tree, which is huge). Strict yes, 2499 output tokens (solo r64: 3224) but 14 strong-arm calls (solo 8), most of them `tools/repair --wait`; the copied-back changes were already complete. A fix (forbid searching outside the project in the agent task text, or a shorter command limit) would change the delegate's instructions, so it was left for the user to decide.
- **Third logicng_b pair (r65):** pipeline 5967 output tokens / 22 calls vs solo 13212 / 27, both 573/573 (strict NO only LiteralTest). The delegate wrote the four weak tests a third time, review flagged them, the strong agent turned them into ClassCastException tests; **all three solo runs wrote that same form too.**
- **Single-prompt delegate, last four runs:** ledger_dev r76 strict yes (1418 out), shop r77 yes (1709); both ESCALATE with failing tests, finished by hand. cactoos r78 yes (3197), round rolled back. logicng_b r79 (15343 out, 30 calls), round rolled back (73 -> 89 compile errors), same ClassCastException tests, strict NO only LiteralTest.


## Changes after the overnight run (2026-11-27). All results above were produced WITHOUT them.
1. **Deletion rule (user ruling):** a test or assertion may be deleted only when what it checks no longer exists in the source; a check that cannot be kept is deleted, never replaced by one that cannot fail. Applied to the strong agent (run_common.md: solo AND pipeline; verify_repair.md) and the agent delegate's task text. The single-prompt delegate is unchanged (it must return exactly the one method, so it cannot delete it).
   Under this rule R1 (shop `convertsToDouble`, tests only the removed `toDouble()`) is an allowed deletion; the strict scorer is UNCHANGED and still counts it as a failure where the reference kept a rewritten test, so such cases have to be reported separately. By the same rule the logicng_b rejection tests (R3) could be deleted, as the reference does.
   Tooling: agent deletions of `.java` test files are now copied back (before: discarded); tools/review now reports the tests of deleted test files (before: NOT reported, a gap that also applied to the strong agent) and adds a hint naming declarations used by a flagged old test that are no longer declared in src/main (removed OR renamed; a hint only). Replay on 6 stored runs: same summaries and flag counts; a deleted file in a scratch copy is reported; the copy-back refuses src/main, pom.xml, non-.java and paths outside the workspace.
2. **D1 fix (`find /`):** the agent sandbox now exposes only the install folders of the JDK, Maven and the agent's Python plus what their libraries and links need (scripts/agent/runtime_paths.py), not the two cluster software trees. Same command as in the stuck runs, same node: old sandbox still running when cut at 300 s, new one finished in 9 s and found the jar. Offline Maven, Java and Maven versions checked for JDK 8, 11 and 17 (no library or resolution errors; for JDK 11 also Python imports, mini, git). The earlier dataset agent runners keep the old mounts. Command time limit unchanged (600 s). NOT yet run with the real model.

## Automation-only control (S7, 2026-11-27, one run, ledger_dev)

Arm S7: the same `tools/repair`, with no delegate model. It does the renames, runs the tests and the review; Claude Sonnet repairs the rest itself. Run r80, made WITH the 2026-11-27 changes (deletion rule, review); the older arms below were run without them.

| arm | strict pass | output tok | calls | total input | fresh input | work |
|---|---|---|---|---|---|---|
| S7 automation only, as measured | 1/1 (53/53) | 6342 | 14 | 794.9k | 123.0k | 190.0k |
| S7 without the stalled calls (estimate) | | ~3.8k-5.2k | 8 | ~449k | ~57k | ~103k |
| solo (2 runs) | 2/2 | 3595-7856 | 5-12 | 258.8k-686.9k | 46.3k-59.8k | 43.3k-169.6k |
| pipeline, Inkling agent (3 runs) | 3/3 | 1298-1389 | 4-5 | 190.2k-241.0k | 29.6k-50.8k | 17.7k-25.4k |
| pipeline, Inkling single prompt (1 run) | 1/1 | 1418 | 5 | 244.6k | 52.0k | 29.0k |

- Automation did 12 renames (63 occurrences). 11 compile errors in 3 files remained, from behaviour and signature changes, and Claude fixed them. Literal sync did not apply: it needs tests that run and fail.
- The stall was Claude's own command (`cat > /dev/null && python3 ...` waited on input and was moved to the background). It cost 6 extra calls, one of them with the whole context uncached (58k), plus a script written twice. The estimate row only subtracts these calls; it is not a second run.
- No test deleted. The review flags 3 methods; in each, the dropped assertion is `assertTrue` around a call that now returns `void`, and the other assertions remain.
- Reading: on this repo, automation without a delegate showed no clear saving over solo, with or without the stall. The agent pipeline's saving here came from the delegate. This does not carry over to repos where the renames fix everything (rules_rename: the delegate was never called).
- Limits: one run, one repo; solo has only 2 runs here and varies widely.

## Ruling on R3 (user, 2026-11-27)

A check that valid, compiling code can no longer trigger counts as gone, so deleting such a test is correct (logicng: testIllegalAdd, testIllegalCC, testIllegalAMO, testIllegalEXO; the reference drops them too). Casts or raw types must not be used to force it. For CC/AMO/EXO the runtime check was removed; for Add the throwing line is still in `addMapping` but `Variable` is final and always positive. The sentence was added to the solo/pipeline rules, the agent task and the final-check text. Not yet tested whether models now delete these tests (needs a logicng rerun). Stored results are unchanged.

## Narrowed sandbox with the real agent (cactoos r81, 2026-11-27, no Claude)

Job 4097258 (17 min, completed). `tools/repair` run directly with the Inkling agent delegate. Agent: submitted after 92 calls, 467 s, 5.16M input / 34.8k output tokens, no ignored changes. Tests 1317 passed, 0 failed, 2 skipped of 1319; scorer success (1283 of 1286; nothing weakened, source unmodified).

- **D1:** the agent ran `find /` again (one command, two searches, for the cactoos-matchers jar). It returned in 4 s with the jar. No command timeout seen. Earlier r63 ran into two 600 s limits and the 30-minute request limit. One run.
- **New problem, sandbox-forced test rewrites (not caused by the narrowing as far as checked):** two tests cannot pass in the agent sandbox. `readsSecureUrlContent` needs www.yegor256.com (no network); `createFileInCustomPath` writes to user.home = `/root` (not writable). The agent task demands a passing suite, so it rewrote both: the HTTPS test now starts a local HTTPS server with an embedded self-signed keystore it generated; the temp-file test uses java.io.tmpdir (r63 made this second change too; none of the earlier cactoos workspaces has the first). The reference keeps the external URL. These are not repairs required by the source change; the first changes what the test exercises. The scorer does not detect it and the review did not flag it for this reason.
- **Review/scorer name bug:** the only flag (`InputOfTest#MethodBodyCommentsCheck [REMOVED]`, also in the scorer's missing_by_name) is false: the name comes from the comment `// @checkstyle MethodBodyCommentsCheck (50 lines)` between `@Test` and the method. The agent removed that comment; `readsSecureUrlContent` still exists. Without this accidental flag tools/repair would have printed ACCEPT.
- **Sandbox /tmp:** the sandbox hides /gpfs but not the node's /tmp. The same search listed a private Maven cache left by the earlier job 4085715 and a sandbox staging folder. On the job node there are leftover `/tmp/agent_m2_*` and many `/tmp/tmp.*` folders owned by us (not deleted). World-readable files of other users in /tmp would also be reachable. Not fixed.
- None of the three is fixed yet.
