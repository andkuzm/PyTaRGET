# Delegation pilots (2026-11-21, unattended run) - first results

Strong arm: fresh Claude Sonnet 5.5 sub-agents (prompt only, offline Maven sandbox in a workspace, no reference material). Delegate: free particle-1 (Qwen3-Coder-class endpoint), via `tools/delegate`; one run used 0.032 USD of the paid fallback. Tasks: `shop` (synthetic, 9 test files, 32 tests, API + behaviour changes), `gson_a80` (66 failing tests, URL string change, 1261 tests), `gson_cbad` (537 expected tests, message-suffix change). Success = compiles, all old test classes keep their test count and assertion count (not below the reference repair), all pass, src/main and poms untouched. Every transcript audit was clean (one benign flag: a sub-agent reading its own spilled tool output).

| Cell | Task | Arm / path | Success | Strong tokens | Tool uses | Time |
|---|---|---|---|---|---|---|
| P1-1 | shop | solo r1 | no (deleted a test whose API vanished) | 58.5k | 3 | 38 s |
| P3-1 | shop | solo r2 | yes | 58.4k | 3 | 33 s |
| P1-2 | shop | S1 auto | yes | 48.2k | 4 | 98 s |
| P2-0b | shop | S1 auto r2 | yes | 50.7k | 5 | 167 s |
| P2-1 | shop | S2 selective | yes | 57.4k | 6 | 51 s |
| P2-2 | shop | S3 3 rounds | yes | 50.3k | 6 | 252 s |
| P2-3 | shop | S1 strong context | yes | 57.6k | 8 | 237 s |
| P2-4 | shop | S1 both contexts | yes | 58.2k | 8 | 229 s |
| P2-5 | shop | S1 + diff check | yes | 52.2k | 7 | 80 s |
| P1-3 | gson_a80 | solo | yes | 52.9k | 3 | 94 s |
| P1-4 | gson_a80 | S1 auto (tool bug) | no | 76.9k | 33 | 543 s |
| P2-0 | gson_a80 | S1 auto (bug fixed) | yes, after manual repairs | 76.5k | 40 | 1207 s |
| P1-5 | gson_cbad | solo | no* | 65.4k | 5 | 88 s |
| P1-6 | gson_cbad | S1 auto | no* | 63.8k | 14 | 348 s |
| P3-2 | gson_cbad | S2 selective | no* | 58.1k | 8 | 220 s |

\* all three gson_cbad runs pass every test; they fail only the "no fewer assertions than the reference" rule in the same four classes (the reference adds assertions for the new message suffix). This is a strictness artefact of the criterion, not a delegation effect; solo fails it too. Treat gson_cbad as "all pass", and decide whether the criterion should compare per class against the OLD suite instead.

## What the pilots say
- Small/medium test classes (shop): the delegate repairs most files correctly (7/7 delegated runs succeed, 0 USD), but the strong agent's tokens drop only ~0-15% (48-58k vs 58k): the fixed cost of the sub-agent (prompt, tools, reading failures, verifying) dominates. S3 (multi-round) and plain S1 were the cheapest (50k); context strong/both and S2 gave no gain. Time is worse for delegation (1.5-4 min vs 0.5 min).
- Large files (gson JsonReaderTest, 2000+ lines): the delegate corrupts them (stray braces, missing imports, duplicated suffix, wrong URL form); the strong agent needs 30-40 tool uses and 9-20 minutes to repair, i.e. delegation is clearly WORSE than solo (77k vs 53k tokens, 20 min vs 1.5 min). The apply order bug found in P1-4 (units applied top-down shifted line numbers) was fixed; corruption still happened afterwards, so the cause is the delegate's output on method units of huge files.
- Solo is not perfect either: one of two solo runs on shop deleted a test when an API disappeared (the delegated runs kept it).
- Cost side: the strong arm is far costlier per run (~50-75k tokens of a Sonnet session) than the delegate (<=0.03 USD); delegation only pays if the strong agent's share shrinks, which it did not in these pilots.

## Ranking of paths (for the next sessions)
1. S1/S3 with automatic context on small and medium test files (cheapest, 7/7 on shop).
2. Do NOT delegate large files; hybrid: delegate only files below ~400 lines or mechanical changes, strong agent applies string/URL changes with sed (it did so in all gson runs).
3. Context from the strong agent (strong/both) and diff verification: no measurable gain on shop.

## Suggested next steps
- Add a size guard to `tools/delegate` (skip files above N lines; report them) and rerun gson_a80/gson_cbad.
- Reduce the strong agent's fixed overhead (shorter prompt, one-call workflow) before measuring savings; measure with more repetitions (shop has only 2 solo runs).
- Fix the scorer criterion (per class vs the old suite) and rescore gson_cbad.
- D2 (agentic local delegate) and Step 0 context-maker comparison are still not run.
- Usage: window 44%, week 85% at the end (the week resets 19:00 Nov 21); the pilots used about 25 window points and 3 week points.
