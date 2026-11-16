# Reference material for the thesis (kept 2026-11-09)

## TaRGet (the paper and repository the thesis compares against)
- Paper: A. S. Yaraghi, D. Holden, N. Kahani, L. Briand, "Automated Test Case Repair Using Language Models", IEEE Transactions on Software Engineering 51(4), 1104-1133, 2025, DOI 10.1109/TSE.2025.3541166, arXiv 2401.06765 (v4, 14 Feb 2025).
  - Local copy: `TaRGet_paper/Yaraghi2025_TaRGet_TSE_arXiv2401.06765v4.pdf` (arXiv v4 PDF, as fetched on 2026-11-09). `TaRGet_paper/extracted_text_crude.txt` is a crude text dump of it (ligatures, missing spaces; tables are readable but check against the PDF before quoting).
- Repository: https://github.com/Ahmadreza-SY/TaRGET , shallow clone at `TaRGET_repo/`, commit 14b854cc4fcc04fff2e113941f1f8b858f7a8707 (2025-02-15, "Update ReadMe.md"). Third-party code, never run from here. Key files: `fine-tuning/eval.py` (EM/BLEU/CodeBLEU), `fine-tuning/test_run.py` (plausible repair), `common/maven_parser.py` (verdicts).

## How the paper's numbers are defined (checked against the paper text and the repo code)
- EM: a case counts when ANY beam candidate equals the ground truth (`eval.py: compute_scores`); BLEU/CodeBLEU use that matching candidate, otherwise the top beam. Our `Tester.compute_scores` is the same function (same tokenization for BLEU: strip + split on whitespace), so our EM/BLEU/CodeBLEU are directly comparable, as best-of-beam numbers, not top-1.
- PR (plausible repair accuracy): share of test cases with at least one candidate that compiles and passes on the updated version. Candidate is `strip()`ped; equal to the ground truth = success without running; each distinct candidate is run once (cache); per candidate time limit 30 min; verdicts timeout, dependency_error, test_not_executed, unrelated_* are "invalid"; a case with >= 75% invalid candidates is dropped from the PR denominator (`test_run.py: apply_and_run_preds`).
- Beam sizes: 40 for CodeT5+ and CodeGen, 200 for PLBART. Test split: 7,103 repairs of 45,373 in TaRBench (59 Java projects, time-ordered split 80/5/15).
- Paper results (Table 5, best input/output format IO2): CodeT5+ EM 66.1 / PR 80.0 / CodeBLEU 79.3 / BLEU 80.0; PLBART 57.9 / 79.2 / 77.9 / 79.2; CodeGen 57.7 / 77.0 / 74.7 / 75.7. Table 7 baselines: SUTCopy EM 10.8 PR 13.8; CodeT5+ without context EM 28.7 PR 55.9. (IO3 gives PLBART/CodeGen slightly different values; use Table 5 for the full grid.)

## Our corresponding numbers (Python, PyTaRGET, eval_results/, 2026-11-09)
- Test cases 1556 / 1604 / 1886 (CodeT5+ / PLBART / CodeGen); beam 20.
- Best-of-20 EM / BLEU / CodeBLEU: CodeT5+ 11.95 / 44.59 / 50.53, PLBART 4.93 / 46.57 / 51.64 (after the decode fix), CodeGen 8.43 / 48.15 / 52.28.
- Plausibility, top-5 candidates executed, over evaluated cases (1410 / 1456 / 1697): PR@5 19.6 / 7.3 / 16.6; PR@1 11.35 / 1.24 / 6.54. Over ALL test cases (a case whose unmodified test fails counts as 0): 17.7 / 6.7 / 15.0.
  Same cases for all models (1321): PR@5 19.91 / 7.65 / 16.65.
- On the evaluated cases: EM top-1 5.7 / 0.5 / 2.3; EM@5 9.7 / 2.2 / 5.8; EM@20 11.8 / 4.3 / 8.3. PR@5 without an exact-match candidate among the top 5: 9.9 / 5.2 / 10.8.

## Where the comparison is NOT like for like (state this in the thesis)
1. Beam depth for PR: paper = best of 40 (200 for PLBART); ours = best of the top 5 of 20. Run with k=20 (about 2-3x the 3.5 h) to get closer; a beam of 40 would need new predictions.
2. Language and test execution: Java/Maven (compile errors are a breakage type) vs Python/pytest (no compile step; a candidate with a syntax error counts as a failure).
3. Cases whose unmodified test cannot be run: the paper's benchmark was validated when built; ours are filtered by a control run (8-10% left out, ~40% of those collection errors from dependencies we did not install). The paper also drops cases with >= 75% invalid verdicts; we count a timeout as a FAIL and do not drop cases.
4. Candidate normalization: they strip and compare; we compare after indentation normalization (more lenient for the equality shortcut).
5. Data: 59 hand-selected Java projects vs about 500 mined Python repositories (cases cluster per commit); different training sizes (they train on 80% of 45k repairs: ~36k; we train on about 8-10k).
6. Input window and format: ours is 1024 tokens for CodeT5+/PLBART and 2048 for CodeGen, with the changed hunks prioritised by TF-IDF (`HunkPrioritizer`). The paper's best format IO2 is "text-based similarity hunk ordering" (`encoders/simOrderEncoder.py` in their repo); I did NOT check whether the two are equivalent, nor their window sizes. Read both before claiming the inputs match.
7. Models: the same checkpoints as the paper (verified in `data_processing/encode_tune_test.py`): `salesforce/codet5p-770m`, `uclanlp/plbart-base` (140M), `salesforce/codegen-350M-multi`.

## Published artifacts (downloaded 2026-11-09, md5 verified against figshare; https://figshare.com/articles/dataset/_b_TaRBench_A_Comprehensive_Benchmark_for_Automated_Test_Case_Repair_b_/25008893, CC BY 4.0)
- `TaRBench_download/TaRBench/` (TaRBench.zip, 8 GB unpacked: `projects/<owner>/<repo>/dataset.json`, `splits.csv`), `TaRGet_Results_download/TaRGet_Results/` (TaRGet_Results.zip: `Best_on_TaRBench/{test,valid,train}.json, test_predictions.json, test_verdicts.json`, `VS_CEPROT/`). Not downloaded: TaRGet_Best_FineTuned_Model.zip (2.7 GB).
- From their own test_predictions.json (7103 cases, 40 beams) and test_verdicts.json (284,120 verdicts = 7103 x 40; counts: compile_error 199,869, failure 43,460, success 33,226, test_not_executed 3,790, unrelated_compile_error 2,070, unknown 1,485, dependency_error 158, timeout 62), recomputed with the rule of `test_run.py` (cases with >= 75% invalid verdicts dropped, 44 of 7103):
  | k (prefix of the 40 beams) | EM@k | PR@k |
  | 1 | 42.8 | 50.3 |
  | 5 | 55.5 | 63.8 |
  | 10 | 58.5 | 67.4 |
  | 20 | 60.9 | 71.2 |
  | 40 | 66.1 | 80.1 (80.0 without dropping) |
  k=40 reproduces the paper's EM 66.1 and PR 80.0, so the metric definitions above are right. A beam-20 search would give slightly different candidates than the first 20 of a beam-40 search.
- Reading: at equal depth (top 5) the paper's CodeT5+ has EM 55.5 / PR 63.8 on Java against our 9.7 / 19.6 on Python, so beam depth is NOT what explains the gap in plausibility.

### Completeness of the published data and the "EM-implies-plausible" shortcut (checked 2026-11-09)
- Complete for the test set: `splits.csv` lists 45,373 ids (36,639 / 1,631 / 7,103); every id has a `dataset.json` entry (54 project files); `Best_on_TaRBench/test.json` (with `input`, `output` = the correct repair, `hunk`, sources), `test_predictions.json` (40 beams) and `test_verdicts.json` (40 verdicts per case) cover exactly the 7,103 test ids. Only the best model (CodeT5+ with IO2) is published; PLBART and CodeGen predictions/verdicts are not.
- A candidate equal to the correct repair always has verdict success (0 exceptions in 284,120 verdicts), so "correct repair in the top k => plausible" holds. It is a lower bound: EM@k vs the real PR@k (k = 1/5/10/20/40): 42.8/55.5/58.5/60.9/66.1 vs 50.3/63.8/67.5/71.2/80.0, i.e. 7.5-14.0 points of plausible repairs that are not exact matches. Use their verdicts, not the shortcut, for PR.
- The same split for ours (top 5 of 20, evaluated cases, CodeT5+/PLBART/CodeGen): EM@5 9.7/2.2/5.8, PR@5 19.6/7.3/16.6, so 9.9/5.2/10.8 points of plausible-but-not-exact repairs (theirs: 8.4 at k=5). The gap to the paper is almost entirely the exact-match rate (55.5 vs 9.7 at k=5), not the extra plausible repairs.

### Extension run: plausibility for candidates 6-20 (hand-off, 2026-11-09)
- What: candidates 6-20 of the 20 beams for every case that was evaluated in the top-5 run (CodeT5+ 1410, PLBART 1456, CodeGen 1697 cases), results in `~/eval_results/<model>/0.8/plausibility_ranks6_20.json` (the top-5 files `plausibility.json` are untouched). The control run is skipped for these cases and a candidate whose text was judged in the top 5 keeps its verdict. Code: `--first-rank 5 --k 20 --extend plausibility.json` (commit 0079253), report: `PYTHONPATH=. venv3.12/bin/python -m data_processing.plausibility_report` (PR@1/5/10/20 and EM for our models and the paper's best model; cells for k=10/20 count only cases with verdicts up to k, the number is in brackets).
- State when the run was stopped cleanly (2026-11-09 11:04, no process or scratch left): only a few big repositories done (CodeT5+ 108, PLBART 29, CodeGen 242 cases of 1410/1456/1697), so the k=10/20 cells are tiny samples of the biggest repositories and mean nothing yet.
- Continue (finished cases are kept; start from `~/` inside a session with >= 4 CPUs, 16 GB): `bash ~/slurm/run_direct_plausibility.sh <seconds = session time left - 1800> --k 20 --first-rank 5 --extend plausibility.json --out-name plausibility_ranks6_20.json` via Bash run_in_background; expected about 4-8 h in total (first run of 5 candidates took 3.5 h with 1000-1400 cases/h). Recreate the health crons (30 min, 1 h, 2 h, 3 h, then every 3 h; compare case counts with the previous check, STALL on the first line). Stop cleanly with `kill -INT $(pgrep -f "[p]ython -u -m data_processing.plausibility_test --results" | head -1)`. When done: run the report and put the table into `~/references/README.md` and here.
- Not started: running the TaRGet repo with our parameters on TaRBench (Python 3.8 venv; `module` is not usable from the session shell, JDK 8/11/17 and Maven 3.6.3 in user space, A100 for their torch 1.12.1, batch 8 / input 1024, three models, PR on a fixed random sample of ~500 test cases). TaRBench and their published predictions/verdicts are already in `~/references/`.

### Extension run finished (2026-11-09 19:02): plausibility of candidates 6-20, all expected cases (1410 / 1456 / 1697)
Result files `~/eval_results/<model>/0.8/plausibility_ranks6_20.json`, numbers merged by `data_processing/plausibility_report.py` into `~/eval_results/plausibility_report.json`; the check below in `~/eval_results/plausibility_trivial_check.json`. Candidate timeouts (counted as failures): 2 / 0 / 5. PR@k in %, EM@k (indentation-normalized exact match of the reference among the top k) in brackets; our cases = evaluated cases; paper = its best model (CodeT5+, Java, 7,059 cases after dropping cases with >= 75% invalid verdicts, from its published verdicts):

| k | CodeT5+ (ours) | PLBART (ours) | CodeGen (ours) | paper CodeT5+ (Java) |
|---|---|---|---|---|
| 1 | 11.35 (5.74) | 1.24 (0.55) | 6.54 (2.30) | 50.29 (42.76) |
| 5 | 19.57 (9.72) | 7.35 (2.20) | 16.62 (5.77) | 63.78 (55.47) |
| 10 | 23.19 (11.06) | 11.26 (3.50) | 21.57 (7.13) | 67.45 (58.48) |
| 20 | 40.50 (12.20) | 32.83 (4.60) | 35.06 (8.31) | 71.20 (60.86) |

**Caveat for k = 20: the tail of the beam contains trivial "repairs".** PR@10 -> PR@20 jumps by 17 / 21 / 13 points while EM barely moves. Passing candidates per rank rise at the end of the beam (e.g. CodeT5+ ranks 16-20: 96, 138, 238, 141, 89 vs about 60-100 at ranks 6-15): many are only a comment, `pass` or a blank, which removes the failing statement so the test passes without testing anything (454 of 1983 passing CodeT5+ candidates, 914 of 1615 PLBART, 234 of 1697 CodeGen; none of them in the top 10). Weakened assertions (assertNotEqual, assertLessEqual, a bare call) also pass and are not detected by this check. PR@k without candidates that are only comments/pass/blank: PR@20 = 31.13 (CodeT5+), 17.45 (PLBART), 30.52 (CodeGen); PR@1/5/10 are unchanged. The paper's published verdicts show a similar late rise (PR@20 71.2 -> PR@40 80.0 while EM goes 60.9 -> 66.1), so the same question applies to its deep-beam numbers; I did not check their candidates. For the thesis: report PR@1/5/10 as they are, and PR@20 only together with this caveat (or the comment-free number).

### Is the paper's PR inflated by trivial candidates too? (checked 2026-11-09 on their published predictions and verdicts, CodeT5+, 7,059 cases)
Yes, at the end of the beam. Passing candidates per rank are flat at about 600-800 for ranks 6-37, then rise at the last ranks (ranks 38/39/40: 959 / 1198 / 1461), exactly where our own beams show the same effect. PR@k jumps 73.0 (k=30) -> 80.1 (k=40). Candidates that are only a blank or a comment (the dataset's own `// Deleted` marker included; these delete the failing statement so the test passes without checking it; stays valid when the reference is a deletion itself, 250 cases) account for 1,407 of 14,459 passing candidates at ranks >= 21 and for 1,223 of the passing candidates at ranks 38-40. Excluding them: PR@1/5/10/20/30/40 = 50.0 / 63.6 / 67.3 / 71.1 / 72.8 / 77.8 (raw 50.3 / 63.8 / 67.4 / 71.2 / 73.0 / 80.1). So up to k=30 trivial candidates are negligible (<= 0.3 pts), and 2.3 of the 7.1 points between k=30 and k=40 are trivial; the remaining rise at ranks 38-40 is non-trivial-looking but degenerate (a random sample of 12 cases whose first success is at rank >= 21 and which have no exact match anywhere: 3 `// Deleted`, 1 blank, 4 weakened statements such as dropping an argument matcher or replacing a computed value with `ZonedDateTime.now()`, 4 plausible alternative assertions). Consequence: compare our PR@k with theirs for k <= 20 (their 71.2 is robust: PR@20 trivial-free 71.1), and treat their headline 80.0 (k = 40) with the same caveat as our PR@20. Our own k = 20 is the case where the tail effect is much larger (PR@20 40.5 -> 31.1 without comment/pass/blank candidates for CodeT5+).

## Data collection and filtering: TaRGet (TaRBench) vs ours (checked against the paper text and our miner/filter code, 2026-11-09)
| | TaRGet / TaRBench (Java) | PyTaRGET (Python) |
|---|---|---|
| Projects | 4,662 CodeSearchNet Java projects -> Maven and >= 50 stars: 1,257 -> 116 with single-hunk test changes -> 112 with repairs -> 59 kept (projects where > 50% of changed tests did not meet the first two execution conditions dropped; two of the 55 dropped ones added back for their many repairs) | GitHub search API: language:python, nine licence queries (MIT, Apache-2.0, ...), stars >= 10, size bands walked up from 1000 KB, 10,086 repositories searched, 477 with at least one case (476 after filtering); no per-project quality selection |
| Commits | every reachable commit of the default branch | per repository at most 100 commit pairs per run (`PYTARGET_MAX_PAIRS`), only pairs that change a test file and (`REQUIRE_SOURCE_CHANGE=1`) a non-test .py file |
| Changed test | single-hunk test-method changes, matched by qualified name | any test function whose source differs between parent and child; no single-hunk rule (only the later 40-line edit limit) |
| Validation by running | three Maven runs: T_i passes on V_i, T_i fails on V_i+1 and the failure is in the test code, T_i+1 passes on V_i+1; runs with dependency/other external errors discarded (-91,059, -33,977, -1,592) | three pytest runs: parent test passes at the parent, child test passes at the child, the parent's test body run on the child's code fails (failure or syntax error); the failure does not have to be in the test code; unknown verdicts and environment failures are not cases; repository / file circuit breakers |
| Failure type | 82% compile errors (37,245), 18% runtime (8,128) | runtime failures and syntax errors only (no compile step) |
| Source change | SUT changes collected (whole-file additions/removals excluded); repairs where only test files change dropped (-6,302) | pairs without a non-test .py change skipped at mining time |
| Duplicates / repetition | identical repairs removed (-1,039) | cap of 5 per (commit pair, final error line, token-substitution pattern): 16,914 -> 14,414; mirrored cases not removed |
| Other filters | no repair context (-3,748), no breakage location (-1,274), input > 512 / output > 256 tokens (-5,383), trivial rename repairs removed from the evaluation set only (-2,046; 7,196 kept for training) | environment errors by log text (-51), non-repairs: adds skip/xfail/pass, block < 30% of the broken one, edits > 40 lines (-273): 14,363 -> 14,090; no length filter in the dataset (each model's encoder drops what does not fit: 1024 tokens for CodeT5+/PLBART, 2048 CodeGen) |
| Final size | 45,373 in 59 projects (36,639 / 1,631 / 7,103) | 14,090 in 476 repositories; 11,248 train / 2,842 held-out; test 1,556 / 1,604 / 1,886 after the length filter (differs per model) |
| Split | per project, older commits to train, newer to valid and test (80/5/15), so every test project is also in train | global chronological split by commit: oldest 80% of commits train, valid = oldest ~25% of the rest; 25% of held-out cases come from repositories not in train |
| Data age / character | established Maven projects | mostly 2026 commits (65%), 20% from agent/LLM-tooling repositories, cases cluster per commit pair; no manual quality check |

## Additional research (2026-11-16): related work on LLM test repair/update, and on cost-effective delegation
How this was gathered: 8 web searches on 2026-11-16, read from search summaries and abstracts, NOT from the full papers. Treat the numbers
below as unverified until the originals are read; "I found no paper that ..." means "my searches did not surface one", not that none exists.

### 1. Is "LLMs are effective at test repair/update" already shown? Yes for Java, thinly for Python
| Work | What it reports | Language / size |
|---|---|---|
| TaRGet, Yaraghi et al., TSE (https://arxiv.org/abs/2401.06765) | fine-tuned code models repair broken tests; 66.1% exact match; the TaRBench dataset | Java, 45,373 repairs, 59 projects |
| UTFix, OOPSLA 2025 (https://arxiv.org/pdf/2503.14924) | LLM repair of unit tests broken by changes; about 89% on a synthetic benchmark, about 60% of assertion failures on real cases | Python, small benchmarks |
| TestUpdater / UPDATES4J (https://arxiv.org/html/2509.24419v1) | LLM context collection from code changes plus refinement on compile/run errors | Java, 195 samples, 7 projects |
| TEBench (https://arxiv.org/html/2605.06125) | project-level test evolution; separates tests that break from tests that go stale (still pass, no longer check the new behaviour) | Java, 314 tasks, 10 projects |
| TestEvo-Bench, Waterloo and Google, 2026 (https://arxiv.org/abs/2607.02469) | executable "live" benchmark; best agents about 74.6% on test update; success drops on the newest tasks and under tight per-task cost ($3 down to $0.50) | Java, 509 update tasks |
| Source-test co-evolution repair study (https://alphaxiv.org/abs/2611.03197v1) | which context (failure logs, source diffs) and self-repair steps help LLMs repair broken tests | Java, 500 cases, 15 projects |
| CommitUp (ASE 2025) and ReAccept (ISSTA 2025, 60.16% update accuracy) | named in search summaries only; pages not retrieved | Java |
Reading: the general claim is established for Java. Python has UTFix with small benchmarks; our 14k mined Python cases with execution-based
verdicts are far larger. The methodology point is also open: our audit found that "the test passes" overstates success (35-40% of evaluated
cases pass even with the repaired span replaced by `pass`; some passes come from weakened assertions), and we added a span control and an
assertion-aware "checked" score. TEBench's break/stale split is related; I did not see a paper that measures this with a control like ours.

### 2. Has cost-effective delegation of test updates to cheaper/local models been evaluated? I found nothing specific
Nearest work (all adjacent, none on test update/repair):
- Cheap-first cascades and routing: FrugalGPT, RouteLLM (reviewed at https://web.stanford.edu/~shuhuiq/blog/review-routing-cascades.html);
  a practitioner piece on routing for coding agents (https://getunblocked.com/blog/model-routing-coding-agents) warns that cheap-first can cost
  more on hard tasks because failed attempts and retries add up. Theory: cascades succeed when a good quality estimator exists.
- Small model as a subagent: Terminus-4B (https://huggingface-proxy.pages.dev/papers/2605.03195.md), reported to cut the main agent's token use
  by up to about 30% on execution tasks (not tests).
- Architect/implementer splits: a write-up of 606 sessions (https://www.flowhunt.io/blog/the-architect-doesnt-lay-bricks/), blog-level rigour.
- Position and general studies: "Small Language Models are the Future of Agentic AI" (https://arxiv.org/pdf/2506.02153); a cost-aware open-vs-frontier
  benchmark on vulnerability detection (https://www.mdpi.com/2073-431X/15/9/623); AgentFloor on small models and tool use (https://arxiv.org/pdf/2605.00334).
- TestEvo-Bench reduces the budget of the same strong agents; it does not hand the task to a cheaper model.

### 3. Why the delegation angle fits our data (assessment, not a finding)
- Test update has a free verifier: run the test (plus our span control and assertion check). Cascade work says the quality estimator is the hard part;
  here execution provides one.
- We already have a ladder of models on the same cases: fine-tuned small models, local mid-size models (Qwen3-Coder-30B-A3B, gpt-oss-20b),
  free large models via OpenRouter (see docs/openrouter_free_models_contrast_eval.md), and the option to add a paid model.
- First evidence on the contrast subset (rank 1, checked scoring, weighted overall estimate): free Nemotron-3-Ultra 33.2%, particle-1 32.8%,
  ernie-5.5-flash 27.1% vs Qwen3-Coder 17.7%, gpt-oss-20b 17.4%, CodeT5+ 17.1% (contamination of recent models is possible).
- Possible research question: at what quality per cost does a cheap-first cascade (local model, run the test, escalate only on failure) match
  a single strong model on test update?
- Risks: my search was limited, so a closer prior study may exist; the cost of local models needs a defensible accounting (GPU hours, energy);
  a real Claude Code subagent experiment is harder to run at scale than an offline cascade simulation; a paid frontier model needs credit and approval.

### 4. Open to-do for this research note
1. Read the originals (TestEvo-Bench, UTFix, the co-evolution study, Terminus-4B, CommitUp, ReAccept) and confirm what they claim.
2. Search Semantic Scholar / Google Scholar for papers that cite TaRGet and mention cost, small or local models.
3. Fix the cost metric (dollars, tokens, GPU time) and the thesis question; decide on one paid model with a cost estimate first.

### Note on the numbers above in this file (2026-11-16)
The tables in "Our corresponding numbers" and the extension runs were produced BEFORE the indentation fix in `patch_source`, the span control and
the assertion-aware "checked" score. The corrected numbers (per-project split, `plausibility_v3.json`) are in PIPELINE_NOTES.md.
