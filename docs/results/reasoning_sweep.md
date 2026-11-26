# Reasoning-effort sweep on 150 VALIDATION cases (2026-11-26; one candidate per case, prompt version 1)
Sample: ~/eval_inkling/val150 (seed 1234 from the per-project validation set, 643 cases). Inkling-2-Mini and gpt-oss-20b limit 12288 output tokens; Qwen3-Coder (no reasoning setting) 512. Chosen on this sample: Inkling effort low (matches equal within noise to none/medium, 4 empty answers instead of 35/21, cheapest and fastest).

| run | cases | exact | ignoring whitespace | empty | cut off | mean output tokens | seconds |
|---|---|---|---|---|---|---|---|
| inkling_none | 150 | 17 (11.3%) | 39 (26.0%) | 35 | 35 | 5449 | 661 |
| inkling_low | 150 | 9 (6.0%) | 38 (25.3%) | 4 | 1 | 2061 | 427 |
| inkling_medium | 150 | 20 (13.3%) | 37 (24.7%) | 21 | 18 | 4614 | 603 |
| inkling_high | 150 | 13 (8.7%) | 31 (20.7%) | 64 | 63 | 7436 | 802 |
| qwen3coder | 150 | 13 (8.7%) | 16 (10.7%) | 0 | 1 | 68 | 240 |
| gptoss20b_low | 150 | 6 (4.0%) | 18 (12.0%) | 3 | 0 | 304 | 157 |
| gptoss20b_medium | 150 | 5 (3.3%) | 20 (13.3%) | 2 | 1 | 2008 | 217 |
| gptoss20b_high | 150 | 17 (11.3%) | 29 (19.3%) | 52 | 52 | 7187 | 382 |

Paired against qwen3coder (ignoring whitespace): only-this / only-qwen
  inkling_none: 24 / 1
  inkling_low: 23 / 1
  inkling_medium: 22 / 1
  inkling_high: 16 / 1
  gptoss20b_low: 7 / 5
  gptoss20b_medium: 7 / 3
  gptoss20b_high: 18 / 5

Caveats: textual match only (no plausibility yet); 150 cases: differences of a few cases between effort levels are noise; 'cut off' = the answer hit the output limit, mostly inside the reasoning.
