| Cell | model calls | added context | work (extra input tokens processed) | output tokens |
|---|---|---|---|---|
| L1 ledger solo | 4 | 10,961 | 22,797 | 3,864 |
| L1 ledger S4 routed | 9 | 10,316 | 56,210 | 3,014 |
| L2 ledger solo | 7 | 12,695 | 56,732 | 4,523 |
| L2 ledger S4 routed | 8 | 8,996 | 47,186 | 2,057 |

fixed start-up context (first call): 43,874 - 44,469 tokens

| Cell | model calls | added context | work (extra input tokens processed) | output tokens |
|---|---|---|---|---|
| L1 ledger solo | 4 | 10,961 | 22,797 | 3,864 |
| L1 ledger S4 routed | 9 | 10,316 | 56,210 | 3,014 |
| L2 ledger solo | 7 | 12,695 | 56,732 | 4,523 |
| L2 ledger S4 routed | 8 | 8,996 | 47,186 | 2,057 |
| L3 ledger S4+review r3 | 10 | 12,409 | 62,779 | 2,720 |
| L3 ledger S4+review r4 | 9 | 11,944 | 59,315 | 2,554 |

fixed start-up context (first call): 43,874 - 44,668 tokens

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

### Review step (tools/review) added, 2 more runs (L3)
`tools/review` (committed) shows, per test method, the old and new assertion lines only where the MEANING changed (assertion kinds + literals; pure renames are counted but hidden) and flags ASSERTIONS DROPPED / FAILURE CHECK LOST / REMOVED. S4 + review: r3 success, r4 success (both: assertions not below the reference, 53/53 pass). Both agents read the flags, judged the dropped `assertTrue(withdraw(...))` as harmless (withdraw is void now), and r4 additionally noticed a suspended-transfer test that does not check unchanged balances. Work 62.8k and 59.3k extra input tokens (model calls 10 and 9), output 2.7k and 2.6k: the review step costs 1-2 calls and about 8-12k extra work but removes the unreviewed-delegate-edit risk; it did not have to repair anything here, so its protective value is shown only by the flag it raises on the earlier failing S4 run 1 (FAILURE CHECK LOST on withdrawMoreThanBalanceFails), not yet by a live repair.

Summary of all six ledger runs: solo 2/2 success (work 22.8k, 56.7k; output 3.9k, 4.5k); S4 1/2 (56.2k, 47.2k; output 3.0k, 2.1k); S4+review 2/2 (62.8k, 59.3k; output 2.7k, 2.6k). Output tokens (the costly ones) are about 40% lower with routing, total re-read work is higher with routing, quality is the same as solo once a review step is present. Too few runs for a verdict; the next lever is cutting the strong agent's calls (the first `run_tests`, diff read and review each cost a call).

## Where the strong arm's tokens go (call-by-call, routed runs; script scripts/delegation/call_breakdown.py)
Fixed start-up context 44k tokens (system prompt + tool definitions, every call re-reads it from cache) is not the task. Output tokens per routed run: 2.1k-3.0k (solo 3.9k-4.5k, dominated by writing the edits: 1.0k+1.9k in two calls).
- Final report: 380-610 tokens = 14-22% of the output (the prompt allowed 10 lines; now 3 lines).
- Writing the plain-rename `sed` list and the delegate call: 420-600 tokens (about 20%), one call.
- Waiting for the delegate: 3-5 calls (`sleep`, `until` loops, reading a spilled output file) = 450-1000 output tokens (20-35%) plus 3-5 extra re-reads of the 50k context. Cause: `tools/delegate` takes more than the 2-minute Bash default, so the harness backgrounds it and the agent polls. Prompt note added (call it with a 300000 ms timeout, wait in one call).
- Reading the first test run 2.8-3.1k chars; `tools/review` 6.7k chars (about 1.7k tokens of added context, the largest single addition): review output now lists only flagged methods in full and the rest as one line of names (1.5k chars).
- Wasted calls in S4 run 1: two delegate calls with 0 units (fixed).
Not yet re-measured with these changes; expected saving per routed run: about 0.5-1k output tokens (final report, polling) and 3-5 calls.
Open option: let the delegate do the plain renames too (S1 + review): the strong agent's output would be one delegate call, the review and a 3-line report, but earlier S1 runs showed the delegate sometimes mishandling large files.

## Rerun with the call-cutting changes (L4, 2026-11-21): S4+review r5 and S1+review r5
| Run | Success | Work | Added context | Output tokens | Calls |
|---|---|---|---|---|---|
| S4+review r5 | yes (54 tests: review flagged `withdrawMoreThanBalanceFails` FAILURE CHECK LOST; the agent kept it as a new-behaviour test and added an OverdraftException test; also added balance/ledger checks to the suspended-transfer test) | 64.5k | 10.6k | 2.6k | 10 |
| S1+review r5 (delegate does ALL, strong agent only runs it, reviews, reports) | yes (53/53, no manual edits, review flagged only the harmless assertTrue(transfer())) | 46.2k | 8.3k | 1.7k | 9 |
Reading: S1+review has the lowest strong-arm output so far (1.7k vs solo 3.9-4.5k, routed 2.1-3.0k) and the lowest work among delegated runs; it needed no repair this time, but it is one run, on a small project, and earlier gson runs showed the delegate corrupting large files (now guarded by answer validation). The review step produced a live catch in S4 r5 (the lost failure check was flagged and repaired by the agent), which was the open question. The 3-line report cut the final message to 270-315 tokens (was 380-610). Polling for the delegate is NOT gone: the harness still moves the long `tools/delegate` call to the background (per-unit latency of the free endpoint up to 186 s with 4 workers), so the agents still spend 2-3 calls waiting; workers raised from 4 to 8 to shorten it (not re-measured).

## L5 (2026-11-21): S1+review on ledger r6 and on gson_a80 (large files)
- ledger S1+review r6: success (54/54, no weakened class); review flagged a lost failure check on the overdraft test again and the agent added a test for it; output 2.5k tokens, 12 calls (3-4 of them polling for the delegate). With r5 (1.7k) the delegate-all arm on ledger: 2 of 2 successes, output 1.7k-2.5k (solo 3.9k-4.5k, routed 2.1k-3.0k).
- gson_a80 S1+review r6: all 1261 tests pass (0 failed), but the scorer marks it NOT successful: ReflectionAccessTest still expects ".../blob/master/Troubleshooting.md#reflection-inaccessible" where the reference has "blob/main"; the test is in the skipped part of the suite, so the run cannot see the wrong value and `tools/review` did not flag it (a literal change within an unchanged assertion shape). Output 3.6k tokens, 15 calls; compare solo 0.8k output, 4 calls on the same task (the change is one URL: delegation does not pay when a single replace does it all).
- Delegate on gson_a80: 66 units, only 24 applied. 27 of the 66 answers were EMPTY (the free endpoint returned nothing), the rest were rejected by the structural validation or by the assertion rule; 34 of the 66 calls went to the paid fallback (spent 0.087 USD; ledger total 0.14 of the 0.20 cap). Fix made: an empty answer is requested again, at most twice; the rejection reason is now logged per unit (`note` in delegate_calls.jsonl). Not re-measured.
- Lesson for the repo set: tasks where one search-and-replace does everything are bad for delegation; tasks with many real logic edits are where it can pay. Skipped tests hide wrong values from both the suite and the review.
