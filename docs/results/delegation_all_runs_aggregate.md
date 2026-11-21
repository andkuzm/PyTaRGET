# All scored repo-set and real-repo runs, aggregated (2026-11-22)

50 runs scored with scripts/delegation/batch_report.py, aggregated by scripts/delegation/aggregate.py. A delegated run counts as valid when the delegate answered at least some units (runs where it was unavailable, HTTP 402/429, are excluded). 'success' is the strict criterion (all tests, none shrunk or weakened vs the reference); the strict criterion is not met by any LogicNG run, see delegation_real_runs.md. Stray-fence patch of the delegate applied 2026-11-22 (after the cactoos/LogicNG reruns).

| repo | arm | valid runs | success | output tokens (mean) | model calls (mean) | added context (mean) | work (mean) | delegate $ (mean) |
|---|---|---|---|---|---|---|---|---|
| real_cactoos | S1 | 1 (of 2) | 1/1 | 7200 | 18.0 | 30829 | 298941 | 0.0000 |
| real_cactoos | S4 | 1 (of 2) | 1/1 | 7032 | 19.0 | 28365 | 338032 | 0.0000 |
| real_cactoos | solo | 1 (of 1) | 1/1 | 4856 | 12.0 | 22540 | 164192 | 0.0000 |
| real_logicng_b | S1 | 2 (of 2) | 0/2 | 20642 | 51.5 | 86524 | 2522787 | 0.0000 |
| real_logicng_b | S4 | 1 (of 2) | 0/1 | 16376 | 48.0 | 73992 | 2281055 | 0.0000 |
| real_logicng_b | solo | 1 (of 1) | 0/1 | 19838 | 58.0 | 95615 | 3913678 | 0.0000 |
| real_ta4j | S1 | 1 (of 2) | 1/1 | 3075 | 18.0 | 17836 | 174386 | 0.0129 |
| real_ta4j | S4 | 2 (of 2) | 2/2 | 1462 | 6.5 | 12549 | 55706 | 0.0000 |
| real_ta4j | solo | 2 (of 2) | 2/2 | 1272 | 6.0 | 11662 | 43364 | 0.0000 |
| real_tablesaw | S1 | 1 (of 2) | 1/1 | 2174 | 11.0 | 9808 | 64737 | 0.0000 |
| real_tablesaw | S4 | 2 (of 2) | 2/2 | 1088 | 6.0 | 12217 | 50255 | 0.0000 |
| real_tablesaw | solo | 2 (of 2) | 2/2 | 1065 | 6.0 | 12142 | 47222 | 0.0000 |
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
  S1: n=13 out mean 5374, calls 18.9, added ctx 23857, work 481949
  S4: n=14 out mean 2722, calls 10.5, added ctx 16301, work 228564
  solo: n=14 out mean 2652, calls 9.6, added ctx 16498, work 318910
