# Repo-set comparison: solo vs routed (S4) vs delegate-all (S1), all with tools/review (2026-11-21)

Four generated Java projects of 8 rule classes / 56 tests each (scripts: ~/delegation_exp/synth/gen_rules.py): **rules_rename** (class and method renamed, behaviour unchanged: one search-and-replace does everything), **rules_logic** (every formula changes: all expected values change), **rules_big** (same change as logic but all 56 tests in ONE class), **rules_exc** (a negative amount returned -1, now throws: 16 tests change structure). Strong arm = fresh Sonnet 5.5 sub-agent per run; each reads its prompt file first (one extra call versus the earlier ledger/gson runs, same for every arm); handoff text = what the developer changed (generated, not from a developer agent). 2 valid runs per cell (some cells rerun because the free-tier delegate returned HTTP 429 and the agent fell back to doing it itself - those runs are excluded; one run per arm were scored while unfinished by a bug of mine and were redone). Numbers: mean per run; 'work' = extra input tokens processed beyond the 44k fixed start-up context.

| repo | arm | valid runs | success | output tokens (mean) | model calls (mean) | added context (mean) | work (mean) | delegate $ (mean) |
|---|---|---|---|---|---|---|---|---|
| rules_big | S1 | 2 (of 2) | 2/2 | 1460 | 9.0 | 8576 | 50732 | 0.0096 |
| rules_big | S4 | 2 (of 3) | 2/2 | 1145 | 7.0 | 10640 | 50665 | 0.0105 |
| rules_big | solo | 2 (of 2) | 2/2 | 1014 | 5.0 | 10642 | 30594 | 0.0000 |
| rules_exc | S1 | 2 (of 2) | 2/2 | 3534 | 19.5 | 13868 | 167828 | 0.0011 |
| rules_exc | S4 | 2 (of 3) | 2/2 | 1377 | 7.5 | 9472 | 49182 | 0.0008 |
| rules_exc | solo | 2 (of 2) | 2/2 | 900 | 5.5 | 7394 | 26186 | 0.0000 |
| rules_logic | S1 | 2 (of 3) | 2/2 | 1168 | 7.5 | 7184 | 35086 | 0.0065 |
| rules_logic | S4 | 2 (of 3) | 2/2 | 1442 | 8.0 | 11377 | 61473 | 0.0065 |
| rules_logic | solo | 2 (of 2) | 2/2 | 1126 | 5.0 | 8695 | 26798 | 0.0000 |
| rules_rename | S1 | 2 (of 2) | 2/2 | 1902 | 12.0 | 9681 | 87201 | 0.0008 |
| rules_rename | S4 | 2 (of 2) | 2/2 | 832 | 5.0 | 6673 | 23123 | 0.0000 |
| rules_rename | solo | 2 (of 2) | 2/2 | 842 | 5.0 | 5876 | 19272 | 0.0000 |

All repos pooled (valid runs):
  S1: n=8 out mean 2016, calls 12.0, added ctx 9827, work 85212
  S4: n=8 out mean 1199, calls 6.9, added ctx 9541, work 46111
  solo: n=8 out mean 970, calls 5.1, added ctx 8151, work 25712


## Reading (honest)
- All 24 valid runs succeed (56/56 tests, nothing weakened, `tools/review` raised no flag that needed a repair); on these projects quality does not separate the arms.
- **Solo is the cheapest arm on every repo here**: lowest output tokens (about 1.0k vs 1.2k routed and 2.0k delegate-all), fewest calls (5.1 vs 6.9 vs 12.0) and the lowest work. This contradicts the earlier `ledger` result (solo 3.9-4.5k output tokens vs 1.7-3.0k for delegated), and the reason is visible: on these generated, uniform tests the solo agent writes one small script (a perl/python/sed replacement or a computation of the new expected values) and never writes test code by hand, whereas on `ledger` it had to hand-edit logic tests of different shapes. Delegation pays when the repair needs a lot of hand-written code from the strong arm; here it does not.
- Delegate-all (S1) is the most expensive in strong-arm calls, mainly because the agent waits for the delegate (polling, 2-3 extra calls; on `rules_exc` the delegate rejected 5 of 16 answers - a bug of mine with import lines that shifted line numbers, fixed since - and the agent repaired them; 19.5 calls on average there). Routed (S4) is in between.
- The delegate itself costs almost nothing ($0.0008-0.011 per run) but is not free of risk: the free tier returned 429 under parallel load (excluded runs) and the paid fallback cap (0.30 USD) is at 0.22+.
- Caveats: n=2 per cell; synthetic and uniform; the developer handoff is generated; one extra call per run for the prompt file. Not a verdict on delegation in general, only on tasks where a strong agent can script the repair.

## What to test next (to find where delegation wins)
- Tests of different shapes and hand-written setups (like `ledger`) at larger scale, and real repositories (TaRBench Java cases) where solo cannot script the repair.
- Cases where the strong arm's output is large: many distinct failing tests each needing its own new assertion.
- Reduce the waiting calls (delegate faster or return a handle) before drawing conclusions about S1.
