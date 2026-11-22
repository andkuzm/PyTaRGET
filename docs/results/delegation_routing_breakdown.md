# Where the strong arm's tokens go in delegation runs, and what automation removes

Source: sub-agent transcripts of 56 registered runs (`scripts/delegation/routing_breakdown.py`). Per-run means; the fixed ~44k start-up context is excluded. "work" = extra input tokens processed (context above the first call, summed over calls). Phase = the first tool call of each model call.


### solo (14 runs with transcripts); per-run means; fixed start-up context excluded
| phase | calls | output tok | % out | input added by result | % in | work share (extra ctx re-read) | % work |
|---|---|---|---|---|---|---|---|
| sed / scripted edit | 2.4 | 1148 | 43% | 2844 | 17% | 100474 | 32% |
| read code | 3.3 | 841 | 32% | 5494 | 33% | 149330 | 47% |
| final report | 1.0 | 220 | 8% | 0 | 0% | 16498 | 5% |
| orient (ls/diff/tests) | 1.3 | 189 | 7% | 4471 | 27% | 5625 | 2% |
| wait/poll for delegate | 0.4 | 94 | 4% | 119 | 1% | 35580 | 11% |
| read prompt | 1.0 | 90 | 3% | 3127 | 19% | 0 | 0% |
| run tests | 0.1 | 30 | 1% | 263 | 2% | 9686 | 3% |
| review (+tests) | 0.1 | 25 | 1% | 162 | 1% | 1257 | 0% |
| other | 0.1 | 16 | 1% | 18 | 0% | 459 | 0% |
| **total** | 9.6 | 2652 | | 16498 | | 318910 | |

### S4 (21 runs with transcripts); per-run means; fixed start-up context excluded
| phase | calls | output tok | % out | input added by result | % in | work share (extra ctx re-read) | % work |
|---|---|---|---|---|---|---|---|
| sed / scripted edit | 2.6 | 1466 | 39% | 3660 | 19% | 102375 | 31% |
| read code | 2.2 | 695 | 18% | 3129 | 16% | 78217 | 24% |
| wait/poll for delegate | 2.0 | 420 | 11% | 1510 | 8% | 64234 | 20% |
| orient (ls/diff/tests) | 1.7 | 343 | 9% | 6000 | 30% | 14506 | 4% |
| final report | 1.0 | 288 | 8% | 34 | 0% | 19700 | 6% |
| review (+tests) | 1.1 | 218 | 6% | 678 | 3% | 22474 | 7% |
| delegate call | 0.8 | 192 | 5% | 460 | 2% | 10288 | 3% |
| read prompt | 1.0 | 99 | 3% | 4058 | 21% | 0 | 0% |
| other | 0.2 | 31 | 1% | 166 | 1% | 7508 | 2% |
| run tests | 0.1 | 22 | 1% | 38 | 0% | 5219 | 2% |
| text / thinking only | 0.0 | 1 | 0% | 0 | 0% | 853 | 0% |
| **total** | 12.8 | 3776 | | 19734 | | 325375 | |

### S1 (21 runs with transcripts); per-run means; fixed start-up context excluded
| phase | calls | output tok | % out | input added by result | % in | work share (extra ctx re-read) | % work |
|---|---|---|---|---|---|---|---|
| sed / scripted edit | 2.2 | 1300 | 31% | 3441 | 17% | 105059 | 32% |
| wait/poll for delegate | 3.9 | 747 | 18% | 2476 | 13% | 70605 | 21% |
| read code | 2.0 | 654 | 16% | 3564 | 18% | 66354 | 20% |
| orient (ls/diff/tests) | 1.6 | 396 | 9% | 3730 | 19% | 26052 | 8% |
| final report | 1.0 | 329 | 8% | 33 | 0% | 19682 | 6% |
| delegate call | 1.6 | 249 | 6% | 1049 | 5% | 8809 | 3% |
| review (+tests) | 1.1 | 225 | 5% | 813 | 4% | 22344 | 7% |
| run tests | 0.9 | 117 | 3% | 609 | 3% | 3789 | 1% |
| read prompt | 1.0 | 99 | 2% | 3817 | 19% | 0 | 0% |
| text / thinking only | 0.2 | 63 | 1% | 89 | 0% | 3893 | 1% |
| other | 0.1 | 10 | 0% | 93 | 0% | 4677 | 1% |
| **total** | 15.7 | 4190 | | 19715 | | 331263 | |

## Findings
- Routed (S4) run: 12.8 calls, 3.8k output tokens. Delegate-all (S1): 15.7 calls, 4.2k. Solo: 9.6 calls, 2.7k (all on repos where solo is already cheap).
- Largest output share is hand-written sed/scripted edits (31-43%), then reading code (16-32%).
- Pure overhead of the delegation pipeline: **waiting/polling for the delegate** is 2.0 calls / 11% of output / 20% of work in S4 and 3.9 calls / 18% / 21% in S1. Cause: the harness moves any command running longer than 180 s to the background; agents then poll with Read/Monitor/sleep loops.
- Orientation (ls, diff, first test run) is 1.6-1.7 calls and 6k tokens of added input.

## Automation built (no model involved)
1. `tools/start` (dtools/start.py + autorename.py): one call that applies pure identifier renames found in SOURCE_CHANGES.diff (methods, constants, classes) to the tests and prints the test result. On ledger it cuts compile errors 23 -> 11 and the rest are behaviour changes. Conservative: only identifiers that vanish from the new source; behaviour changes are left alone.
2. `tools/delegate` guard (dtools/delegate_guard.py): runs the delegate detached and returns within 160 s; if unfinished it prints a fixed line and the agent calls `tools/delegate --wait` (no improvised polling).
3. Prompt S5 (docs/delegation_prompts/strategy_S5.md): start, delegate, verify.

## Measured (oracle delegate, so only the strong arm's overhead is seen; 1 run each)
| Task | solo | S1 oracle | S4 oracle | **S5 oracle** |
|---|---|---|---|---|
| ledger out tok | 4194 | 1427 | 2186 | **1047** |
| ledger calls / work | 5.5 / 39.8k | 7.5 / 45.2k | 9 / 85.8k | **7 / 39.3k** |
| shop out tok | 3224 | 1453 | 2039 | **795** |
| shop calls / work | 4 / 36.8k | 8 / 40.1k | 9 / 90.9k | **6 / 25.5k** |

Both runs pass (53/53, 32/32). S5 is 27-45% below the previous best (S1) in output tokens and removes the work penalty versus solo. Caveats: single runs; oracle delegate; the shop workspace has no tools/review; a real delegate takes longer, so the `--wait` call will occur (one cheap call). Not yet measured with a real delegate (credit exhausted).

## S6: `tools/repair` (one call: start, delegate rounds, tests, review, ACCEPT/ESCALATE)
`dtools/repair.py`; review flags that are only a dropped `assertTrue/False(call())` on a method that is now `void` in src/main are auto-accepted. Prompt: strategy_S6.md + verify_repair.md. Oracle delegate, 1 run each, both pass with nothing weakened (ledger 53/53, shop 32/32; shop has no tools/review).
| Task | S5 out / calls / work | **S6 out / calls / work** |
|---|---|---|
| ledger | 1047 / 7 / 39.3k | **446 / 3 / 10.6k** |
| shop | 795 / 6 / 25.5k | **448 / 3 / 8.2k** |

Compared with solo (ledger 4194 out / 39.8k work; shop 3224 / 36.8k): -89% and -86% output tokens, -73% and -78% work. Caveats: oracle delegate (a real one is slower and wrong sometimes, so ESCALATE and `--wait` calls will appear); one run per task; the 3 calls are read prompt, `tools/repair`, final report, which is the floor of this design.

## Automatic routing inside `tools/delegate` (dtools/route.py; `--no-route` disables it)
Pattern found in the 651 particle-1 units: 179 units (all gson) had a reference method identical to the failing one, because the fix lived in a helper/constant (`master` -> `main` in a URL). Signal: the failure says `expected: X but was: Y` and X is not in the method text (X in method: 94% fixed; not in method: 9%). All 99 "empty answers" are in this group (85 used the whole 8192-token budget; earlier I wrongly called them infrastructure failures).
1. **Literal sync (no model):** one substitution X->Y shared by >= 3 failing tests is replaced in the test files of the failing classes (also for failing tests whose message merely contains the old text).
2. **Routing:** method units whose expected text is not in the method are not sent to the delegate; they remain failing for the strong agent (listed in the delegate output as "ROUTED TO YOU"). Only when the delegate picks the units itself (not with `--units`).

Replay on the stored units (`scripts/delegation/replay_routing.py`):
```
method units 562; literal sync would resolve 260 of them without a model; substitutions: 'gson/blob/master/Troubleshoo'->'gson/blob/main/Troubleshoo'

| decision | units | fixed by particle-1 anyway | not fixed |
|---|---|---|---|
| literal sync | 260 | 109 | 151 |
| routed to strong | 10 | 0 | 10 |
| to delegate | 292 | 271 | 21 |

sent to the delegate: 292 units, fixed 271 (93%) vs 380/562 (68%) without routing
remaining misses among delegated: [(('rules_exc', 'rejected by the apply step'), 10), (('gson_cbad', 'applied but rewritten by the'), 4), (('gson_a80', 'empty answer'), 2), (('gson_a80', 'rejected by the apply step'), 2), (('gson_a80', 'applied but rewritten by the'), 2), (('gson_a80', 'rejected: the answer must co'), 1)]
routed units that particle-1 did fix: []
```
End-to-end on a copy of the gson_a80 task: 66 failing tests -> 0 with literal sync alone, no model call (strict score: success). No change on rules_exc, rules_logic, gson_cbad (10 of 14 units routed to the strong agent) and real_tablesaw.
Caveats: the evidence is mostly from gson (one task family); literal sync accepts the new behaviour blindly (as the delegate would), so a real regression that changes a message would be "fixed" too.
