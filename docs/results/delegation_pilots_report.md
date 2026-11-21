# Delegation pilots (2026-11-21, unattended run) - first results

(The 'Strong tokens' column below includes ~43k tokens of fixed start-up context; use docs/results/delegation_strong_arm_work.md for the real work.)

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
- Small/medium test classes (shop): the delegate repairs most files correctly (7/7 delegated runs succeed, 0 USD), but the table's 'strong tokens' column is MISLEADING (it is the context size at the end, of which ~43k is the fixed start-up context of the sub-agent and unrelated to the task); see the corrected section below. Time is worse for delegation (1.5-4 min vs 0.5 min).
- Large files (gson JsonReaderTest, 2000+ lines): the delegate corrupts them (stray braces, missing imports, duplicated suffix, wrong URL form); the strong agent needs 30-40 tool uses and 9-20 minutes to repair, i.e. delegation is clearly WORSE than solo (77k vs 53k tokens, 20 min vs 1.5 min). The apply order bug found in P1-4 (units applied top-down shifted line numbers) was fixed; corruption still happened afterwards, so the cause is the delegate's output on method units of huge files.
- Solo is not perfect either: one of two solo runs on shop deleted a test when an API disappeared (the delegated runs kept it).
- Cost side: the strong arm is far costlier per run (~50-75k tokens of a Sonnet session) than the delegate (<=0.03 USD); whether delegation pays is decided by the strong agent's own work, see the corrected section below.

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

## Addendum 2026-11-21 (validation of delegate answers)
`tools/delegate` now validates every answer before writing it (balanced braces, exactly the expected method, no text after it, class declaration kept in the header, file brace balance unchanged); a bad answer is rejected and the unit stays unchanged. Direct runs of the delegate (no strong agent, 3 rounds) on fresh workspaces: gson_cbad left a compiling file with 1 failing test (the delegate wrote the URL form `blob/main` instead of `master`, a wrong expected value), gson_a80 left 1 compile error in JsonReaderTest (wrong exception type in one test, a semantic mistake). Neither run produced a structurally corrupted file (before: stray braces, lost class declaration, duplicated methods). Not done: programmatic repair of unbalanced answers and re-asking the delegate with the error text (the rejection log showed almost no brace problems in these two runs; the remaining failures are wrong values or types).

## Correction 2026-11-21: strong-arm work without the fixed overhead
Only the strong arm's own work counts (fixing tests, or calling the delegate and validating). The first model call of every sub-agent already holds 43.0-43.5k tokens (system prompt + tool definitions); that is subtracted. Table: docs/results/delegation_strong_arm_work.md (script scripts/delegation/token_report.py). "Work" = extra input tokens processed because of the task (sum over calls of context minus the start-up context), "added context" = growth of the context.

- shop (small files): solo 37k work / 15k added; S1 17k and 26k work (4.6k and 7k added) and S3 30k: delegation cuts the strong arm's work by about 30-55%. S2, strong context and both contexts were NOT better than solo (69-93k work): the strong agent reads the diff and writes the context itself, which costs more than it saves. Best path on small files: S1 or S3 with the automatic context.
- gson_a80 (huge JsonReaderTest): solo 23k work, S1 569k-638k (34-40 calls to repair the delegate's broken file): delegation is about 25x worse there (before the answer validation was added).
- gson_cbad: solo 85k, S1 124k, S2 93k: no gain from delegation on the large files.
