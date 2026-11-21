| Cell | model calls | added context | work (extra input tokens processed) | output tokens |
|---|---|---|---|---|
| L1 ledger solo | 4 | 10,961 | 22,797 | 3,864 |
| L1 ledger S4 routed | 9 | 10,316 | 56,210 | 3,014 |
| L2 ledger solo | 7 | 12,695 | 56,732 | 4,523 |
| L2 ledger S4 routed | 8 | 8,996 | 47,186 | 2,057 |

fixed start-up context (first call): 43,874 - 44,469 tokens

## Ledger mixed task (2026-11-21): plain renames + logic changes, same developer context for both arms
Task: synthetic `ledger` project (53 tests, 10 classes): four classes need only plain renames/moves (EntryTest, LedgerTest, IbanValidatorTest, AccountStatusTest), five need real logic changes (new fee tiers and rounding, overdraft rule, compound interest, suspended-account transfers with two ledger entries, new statement format), plus the mechanical parts inside those. The source change was made by a developer sub-agent from a change request; its handoff note (what it changed, rename vs behaviour change) is given to both arms as "what you remember" (approximation of a fork; a sub-agent cannot be forked). Reference tests (mine) pass 53/53 against the developer's source. Scoring: compiles, all pass, no class shrunk or with fewer assertions than the reference, src/main untouched. Arms: solo (all by itself) vs S4 (plain renames by `sed`, logic classes delegated, then validate/repair).

| Run | Success | Strong-arm work | Added context | Output tokens | Model calls |
|---|---|---|---|---|---|
| solo r1 | yes | 22.8k | 11.0k | 3.9k | 4 |
| solo r2 | yes | 56.7k | 12.7k | 4.5k | 7 |
| S4 r1 | no: one test lost its meaning (the delegate turned "withdraw more than balance fails" into a withdrawal that succeeds with balance -400; assertion count below the reference) | 56.2k | 10.3k | 3.0k | 9 (2 wasted: `--units` did not accept class names, since fixed) |
| S4 r2 | yes (assertions = reference) | 47.2k | 9.0k | 2.1k | 8 |

Reading (n=2 per arm, so a signal at best): S4 produced about 40% fewer output tokens (2.6k vs 4.2k mean) and slightly less added context (9.7k vs 11.9k) than solo, but "work" (re-read context summed over calls) is not lower (51.7k vs 39.8k mean) because it takes more calls, and one of two S4 runs left a semantically weakened test that the strong agent never reviewed. Solo succeeded twice. No clear win for routing on this small project; the benefit, if any, is in output tokens (the expensive ones). The strong agent still does not review the delegate's edits (both S4 agents said so), which is where the quality risk sits.

Tool bugs found with this task and fixed: `find_methods` missed one-line test methods (scorer saw 17/53 tests, delegate unit boundaries could be wrong), `--units` ignored bare class names and `Class#method`, scorer treated a renamed test class as a deletion.
