# Thesis plan: cost-effective test update with small/local language models (written 2026-11-16, update as work proceeds)

Read this first in a new session, together with PIPELINE_NOTES.md (numbers) and docs/thesis_references.md (paper comparison and related work).
Status marks: [x] done, [~] in progress, [ ] not started. Update the marks and the "Decisions" lists as things change.

## 0. Direction
Thesis focus (decided by the user 2026-11-16): **cost-effective delegation of test updates** - can a cheaper or local model handle test updates, with execution
(running the test, the span control, the assertion check) as a cheap verifier, so that a strong main model (e.g. Claude Code) can defer them? Supporting
claim: LLMs are effective at test update (Java: established in the literature; Python: our own large dataset, see docs/thesis_references.md section
"Additional research"). Fit for a University of Tartu computer-science bachelor's thesis: judged plausible (9 ECTS, about 20 pages per snippets of the
institute's guidelines); the guidelines PDF could not be fetched automatically - the user will save it locally to read the rubric and the AI-use rules
(path to be recorded here). Supervisor agreement is still to be confirmed.

## 1. State at 2026-11-16 (details: PIPELINE_NOTES.md, docs/openrouter_free_models_contrast_eval.md, docs/results/)
- [x] Python dataset 14,090 cases, per-project split, three fine-tuned models (CodeT5+, PLBART, CodeGen) at the 100% training set; zero-shot Qwen3-Coder-30B-A3B and gpt-oss-20b (vLLM).
- [x] Corrected plausibility scoring (indentation fix, span control, assertion-aware "checked" score); audited by blind hourly samples.
- [x] Contrast subset (300 Python cases: 150 solved by some earlier model, 150 by none) and three free OpenRouter models: nemotron-3-ultra 33.2%, particle-1 32.8%, ernie-5.5-flash 27.1% (weighted rank-1 estimate) vs 17.1-17.7% for the best earlier models.
- [x] Related-work check (Java benchmarks exist, Python thin, no delegation study for test update found).
- Code: PyTaRGET branch second-approach on GitHub (public repository, never commit secrets); OpenRouter key in ~/.config/openrouter/key (mode 600), $0 of the 10 EUR credit used.

## 2. Standing rules (do not skip)
- HPC: follow ~/CLAUDE.md; ask the user before ANY Slurm submission (list script, resources, time limit); GPU only via Slurm; heavy CPU work inside the session job.
- OpenRouter: only ':free' models until the user approves a paid model WITH a cost estimate; the runner refuses other slugs (--allow-paid). Tell the user what data is sent; test cases come from public repositories.
- Do not push code or docs to the public repository without the user asking; never push keys, tokens, third-party papers or datasets.
- Contamination of recent models with public repositories is a known limitation: state it, do not chase it.
- Report faithfully; blind-audit new scoring changes (own expectation first, then compare with the verdict).

## 3. Work packages and order
Gate after WP1+WP2: "unless something changes drastically" continue to WP3-WP5; if the free/local models fall far behind on Java, revisit the claim before building tools.

### WP1 Fine-tuning data size on Python (the paper's RQ3.1, section 4.7) [x] DONE 2026-11-17 (all trainings, tests and plausibility; see the final status at the end)
- Paper: subsets of 20%, 40%, 60%, 80% of the TRAINING set, made by excluding the OLDEST data per project by commit date; same test set. 100% = the full training set. Our existing per-project models ARE the 100% point (not "80"), so 4 new points per model. Paper finding to compare with: EM +7.4 pp and PR +2.7 pp from 20% to 100%.
- Needed code: an Encoder option (e.g. `train_fraction`) that keeps the validation and test sets of the 0.8 per-project split FIXED and drops the oldest training commits per project until about the fraction of that project's training cases remains (cut at commit boundaries; log the real shares). NOT `train_size` (it would move the test boundary). Prefer deriving the subsets from the already encoded 100% train set (no re-tokenisation); out dirs like ~/eval_results_perproject_trainfrac/<fraction>/<model>/.
- Runs: 3 models x 4 fractions: train (earlier full runs took 1.2 h CodeT5+, 0.2 h PLBART, 1.3 h CodeGen on an H200; smaller subsets are faster) and test (20 min to 1 h). Submit via slurm/submit_pipeline.sh adapted (ask the user first).
- Metrics: EM, BLEU, CodeBLEU for all; corrected plausibility (top 5, `--span-control`) for CodeT5+ first, the others if time allows. Output: a learning curve (EM and PR against training fraction) and the per-fraction table in PIPELINE_NOTES.md.
- Decision: report the curve per model; do not claim a gain from more data beyond what the curve shows (the 5 points per model are few; give the number of training cases per point).

### WP2 Java contrast subset (TaRBench) with local and free OpenRouter LLMs [ ]
- Data: ~/references/TaRBench_download/TaRBench (third-party, CC BY 4.0, do not commit) and the authors' published predictions/verdicts in ~/references/TaRGet_Results_download. Sample 300 test cases: 150 that TaRGet's CodeT5+ solved and 150 it did not, then the same per-group analysis as for Python (rate on each group and a weighted overall estimate with the real group sizes).
- Decisions to take: what "solved" means (published plausible-repair verdict at rank 1, or within the top 5, or exact match); sampling (random with a per-project cap as for Python); which TaRBench split (the test split, newest commits).
- Models: local vLLM (Qwen3-Coder-30B-A3B, gpt-oss-20b; prompt v1 has a Java variant: the "ref" train_fraction switch in llm_test.py - check it), and the free OpenRouter models (particle-1, nemotron-3-ultra, ernie-5.5-flash; refresh the free list first).
- Scoring: first EM, BLEU and CodeBLEU (Java CodeBLEU needs jCodeBLEU and the TaRGET virtual environment); plausibility needs a Java toolchain (JDK 8/11/17, Maven 3.6.3 in user space, clones of the projects, Python 3.8 venv per the TaRGet repo) - not set up; decide the effort or restrict plausibility to a smaller sample. Compare with the published CodeT5+ numbers on the same cases.
- Caveat: TaRBench projects are older and heavily present in training data (stronger contamination than our newest Python cases).

### WP3 Stronger, credit-paid models [ ] (after the WP1/WP2 gate)
- Needs: user adds credit and approves the model list; cost estimate first (tokens per case are in the request logs: about 0.9k in and 4k out per request at high reasoning; Python contrast set 300 cases); per-run caps (`--max-cost`); small pilot (100 cases) before any full run.
- Run on the Python contrast subset and the Java subset; keep rank 1 and add 5 candidates for a rank-5 comparison if affordable. Add cost per solved case to every table.

### WP4 A tool that lets Claude Code defer test updates to a lesser model [ ]
- Idea: after the main task changes source code and tests break, Claude Code calls a tool (MCP server, or a CLI plus a skill/subagent definition) that (1) collects the broken test, the failure output and the source diff, (2) builds our prompt, (3) asks a cheap backend (a free or paid OpenRouter model, or a local vLLM/Ollama model), (4) applies the candidate and RUNS the test (with the span control and assertion check as guard), (5) returns the patch if it passes, otherwise escalates (to the next model, or reports back to the main model).
- Design questions: how Claude Code is told to use it (CLAUDE.md instruction, a skill, a subagent), cost logging per call, where the model runs (the HPC cluster is not reachable from a laptop session; the free API route is), safety (never run untrusted repository code outside a sandbox), failure handling.
- Evaluation: (a) offline cascade simulation on our datasets (cheap model first, escalate on failure: quality and cost against a single strong model); (b) a small live study with real Claude Code sessions if feasible (tokens/dollars of the main model with and without delegation). Cascade theory: the quality estimator is the hard part; here execution provides it.

### WP5 Better context for the lesser model (RAG / auxiliary systems) [~] first ablation (B failure output, C symbol definitions, BC) implemented 2026-11-18, not yet run: see docs/ablation_plan.md
- Current prompt: broken test, broken lines, up to 4 hunks (1024-token window for the fine-tuned models, up to 24k for LLMs). Candidates: retrieval of similar tests and of the definitions of changed symbols (tree-sitter, code search), repository documentation, recent history of the test file, execution feedback (the failing output) in a second attempt.
- Evaluate by ablation on the Python contrast subset: each source of context alone and combined, with the cost of the extra tokens; measure the gain per extra token.

### WP6 Writing [ ]
- Read the UT Institute of Computer Science guidelines (rubric, supervisor/opponent roles, AI-use disclosure, data and ethics) once saved locally; outline: introduction and research questions, related work (docs/thesis_references.md), dataset and pipeline, scoring validity (the weakened-test finding, the audits), experiments (WP1-WP5), limitations (contamination, rank-1 vs rank-5, free-tier availability), conclusions.
- Record the supervisor, the topic registration date and the defence deadline here once known.

## 4. Open decisions (ask the user)
1. Definition of "solved by CodeT5+" for the Java subset (rank 1 vs top 5; plausible vs exact).
2. Java plausibility: build the toolchain, or report EM/CodeBLEU plus the authors' PR on the overlapping cases only.
3. Which paid models and how much credit; local-model cost accounting (GPU hours and energy).
4. Whether the tool in WP4 targets Claude Code only (MCP) or any agent.
5. Location of the saved UT guidelines and the supervisor's name/deadlines.

## 5. Next concrete actions (in order)
1. WP1: implement `train_fraction` in the Encoder (with a unit test on the tiny cases fixture), check the shares per project, then ask the user to confirm the Slurm submissions.
2. WP2: inspect the TaRBench test split and the published verdict files, decide "solved", build the 300-case Java subset (a new script next to build_contrast_subset.py).
3. Keep PIPELINE_NOTES.md current after every result; commit and push docs only when the user asks.

## Status update 2026-11-17 (end of the session): WP1 mostly done
- Data: `data_processing/train_fraction.py` (+ `make_trainfrac_dirs.py`, tests in tests/test_train_fraction.py) derive the 20/40/60/80% subsets of the TRAINING set (newest commits per project,
  total share calibrated; every project keeps its newest commit, so the "20%" point is really about 28% = the floor), validation and test unchanged. Directories: ~/eval_results_perproject_trainfrac/<20|40|60|80>/<model>/0.8/ (job ids and the resubmission history: submitted_jobs.txt there).
- Hyper-parameter check (CodeT5+, full training set, same 1293 test cases): the paper's settings (lr 1e-5, 4 epochs, early stop 1) are NOT better: batch 1 -> EM 17.79 (baseline 18.87), BLEU 40.3, CodeBLEU 47.5; batch 8 -> collapsed (EM 4.33, undertrained); the epoch-5 checkpoint equals the epoch-2 one (EM 18.95). Decision: keep lr 5e-5, batch 8, 10 epochs, early stop 3. Folders: ~/eval_hparam_check/.
- All 24 + 5 + 17 Slurm jobs finished with exit code 0. Learning curve file: docs/results/trainfrac_curve.json (script scripts/thesis_analysis/build_trainfrac_curve.py).
- CodeT5+ complete (EM / plausible @1 / @5 plain): 28%: 9.67 / 6.88 / 11.23; 40%: 15.24 / 12.59 / 18.57; 60%: 14.46 / 13.68 / 20.74; 80%: 18.02 / 16.12 / 26.27; 100%: 18.87 / 15.45 / 26.55. Plausibility rises strongly up to about 80% of the data, then is flat; the paper found PR hardly depended on data size (+2.7 pp vs EM +7.4 pp), ours rises more (+15 pp @5).
- PLBART: EM 5.07 / 6.71 / 8.72 / 7.75 / 7.82 (flat from 60%); CodeGen: EM 8.77 / 11.29 / 11.17 / 11.80 / 12.81. Single runs, noise about +-1 point.
- TO FINISH (plausibility, run natively in a session with >= 6 h left; results are resumable; ask before anything else):
  * CodeGen pass (72% done, about 2 h more): `cd ~/PyTaRGET && PYTHONPATH=. timeout -s INT <secs> ~/venv3.12/bin/python -u -m data_processing.plausibility_test --results ~/eval_trainfrac_plaus --models codegen_f20 codegen_f40 codegen_f60 codegen_f80 --workdir $CLAUDE_JOB_DIR/tmp/plaus_trainfrac --workers 3 --k 5 --out-name plausibility_v3.json --span-control`
  * PLBART pass (about 80 of 1342 cases per set done): same command with --models plbart_f20 plbart_f40 plbart_f60 plbart_f80.
  * `python -I scripts/thesis_analysis/trainfrac_status.py --link` shows the state and builds the layout ~/eval_trainfrac_plaus; then rerun scripts/thesis_analysis/build_trainfrac_curve.py.
- Other open items: the code of WP1 and the scripts are committed locally only (not pushed); the free OpenRouter results are in docs/ already (pushed). Next work packages: WP2 (Java contrast subset), see above.

## WP1 FINAL status (2026-11-17 22:30): everything scored
Complete learning curves (EM / plausible @1 / @5 plain; 1293 / 1342 / 1585 test cases; full table with BLEU, CodeBLEU and checked scores: docs/results/trainfrac_curve.json):
- CodeT5+ 28% / 40% / 60% / 80% / 100%: EM 9.67 / 15.24 / 14.46 / 18.02 / 18.87; plausible @1 6.88 / 12.59 / 13.68 / 16.12 / 15.45; @5 11.23 / 18.57 / 20.74 / 26.27 / 26.55.
- PLBART: EM 5.07 / 6.71 / 8.72 / 7.75 / 7.82; plausible @1 1.48 / 2.96 / 3.13 / 1.65 / 3.06; @5 7.22 / 7.22 / 8.36 / 8.01 / 7.60 (flat from the smallest size on).
- CodeGen: EM 8.77 / 11.29 / 11.17 / 11.80 / 12.81; plausible @1 2.26 / 5.74 / 7.92 / 7.40 / 7.85; @5 15.62 / 15.77 / 16.75 / 16.68 / 17.52 (only the ranking improves).
- Reading: more data helps CodeT5+ clearly up to about 80% of our training set, then flat; CodeGen mostly gets better candidate ORDER; PLBART gains nothing. Single runs, noise about +-1 point. The paper found PR hardly depended on data size (+2.7 pp) while ours grows for CodeT5+ (+15 pp @5): we start from a much smaller training set.
- Nothing is left in WP1 except pushing the code and docs (committed locally only). Next: WP2 (Java contrast subset), after the two decisions listed in section 4.


## Status update 2026-11-18: WP2 done, WP5 started
- WP2 (Java contrast subset, 273 cases without LGPL; 257 with a passing reference): own Java plausibility runner (`data_processing/java_plausibility.py`, validated 66/66 against the authors' published verdicts), results in docs/results/java_contrast273_results.txt and java_contrast273_bleu_codebleu.txt (EM, BLEU, CodeBLEU, plausible@1 per group, weighted). Free models (particle-1 65%, nemotron 62%, ernie 57% weighted plausible@1) beat CodeT5+ published 50.3%; local Qwen3-Coder 40%, gpt-oss-20b 38%. About 75-80% of successes equal the reference (contamination/easy). Caveats in the result files.
- WP3 and WP4 are on hold (user decision 2026-11-18); WP5 first step = docs/ablation_plan.md.

## Status update 2026-11-19: WP5 first results
- Ablation (docs/ablation_plan.md, docs/results/ablation_qwen_results.txt): failure output of the broken test +20 points (Qwen3-Coder) / +27 (particle-1) at ~400 extra tokens; symbol definitions ~0; both together +22 / +31.
- Agent (docs/agent_plan.md): mini-SWE-agent + Qwen3-Coder in an offline sandbox works. First run 65% (71 cases) was inflated by already-repaired sibling tests in the workspace; the sibling-safe control gives 50.5% (54 cases) against 37-39% for Qwen3-Coder with the failure output in the prompt and 56-57% for particle-1 with it (see docs/results/agent_sibling_safe_results.txt).
- Next: sibling-safe agent run and scoring; retry the empty remote answers; more cases for the ablation (nested sample 200) if quota allows; push when the user asks (many local commits).

- 2026-11-19 later: Java agent (Qwen3-Coder, offline Maven) is being prepared: the Java workspace keeps all other tests repaired (the module must compile), so each case records how many other test lines the repair changed; analyse clean cases (0 other lines) separately.

## Status update 2026-11-19 (night): Java agent done
- Java agent (Qwen3-Coder, offline Maven, 59 cases): strict 47.4%, relaxed 57.3% weighted checked plausible@1 vs 41.7% for the same model prompt-only and 55-67% for nemotron/particle (docs/agent_plan.md, docs/results/java_agent_results.txt). Python (sibling-safe): 50.5% vs 36.8% for Qwen with the failure output.
- Cleanup: home file count 3.15M -> ~0.9M (Trash, old mining dirs, caches, scratch); verified dataset backup in ~/backups/2026-11-19/ (64 archives + SHA-256 manifest).
- Open: push of the local commits (the last push attempt was denied by the permission classifier; waiting for the user), delete ~/agent_java/{ws,m2} when no longer needed, retry the empty remote answers, more cases for the ablation and agent runs, WP3 (paid models) and WP6 (writing; UT guidelines still to be saved).

## NEXT SESSION (written 2026-11-19 ~13:15, the session ended with three Slurm jobs queued)
- Queued (independent of any session): Java agent rerun with an 80-call limit and 64k context, job 3748197 -> ~/eval_agent/java_qwen3coder_80 (resets the workspaces itself); Python preparation job 3748198 (CPU, partition amd) -> ~/agent_work3, then the Python agent job 3748199 (afterok:3748198, 80 calls, sibling-safe) -> ~/eval_agent/qwen3coder_sibling_safe_80. Check `squeue -u kandrei`, slurm/logs/*.out.
- Then, in a new session (scoring runs natively and needs network): Java: `python scripts/agent/java_agent_score.py --root ~/agent_java --agent-out ~/eval_agent/java_qwen3coder_80 --out ~/eval_agent_plaus/java_qwen3coder_80.json --workdir <scratch>` and the same with `--relaxed` (out ...80_relaxed.json); then adapt scripts/thesis_analysis/java_agent_report.py to the new files and compare with the 40-call run (47.4% strict / 57.3% relaxed, 59 cases). Python: `scripts/agent/recollect.py`, `score_layout.py` (agent root ~/agent_work3, out ~/eval_agent_plaus/qwen3coder_sibling_safe_80), `plausibility_test.py --results ~/eval_agent_plaus --models qwen3coder_sibling_safe_80 --k 1`, then `agent_vs_ablation.py agent_work3 qwen3coder_sibling_safe_80 "AGENT 80 calls"` (40-call sibling-safe result: 50.5% weighted on 54 cases).
- Question to answer: do 80 calls reduce the 22 (Python) / 7 (Java) call-limit exits and the rule violations, and how much does plausible@1 rise?
- Housekeeping: ~/agent_java (workspaces + Maven repository, ~400k files) and ~/agent_work3 can be deleted after scoring; datasets are backed up in ~/backups/2026-11-19/ (MANIFEST.tsv); home file count ~1.1M (soft limit 2.1M); the last push was d3e66cf, later commits are local until the user says push.

## Status update 2026-11-19 (afternoon): step-limit rerun scored
- The NEXT SESSION block above is done: 80-call agents on Java and Python were run and scored (docs/results/agent_step_limit_results.txt). Java: strict 63.4%, relaxed 70.3% (was 47.4% / 57.3%), on par with the best free prompt-only model; Python: no measurable change (46.1% -> 48.0%).
- Cleanup: Python agent workspaces/environments deleted; ~/agent_java (workspaces + Maven repository, ~520k files) still there until the user confirms (needed only for re-scoring); home file count ~1.2M.
- Open: push (last push d3e66cf; later commits local), prompt/harness changes for the agent (edit discipline), stronger model in the agent (needs credit approval), writing.

- 2026-11-19: one-page summary of agent vs no-agent scores: docs/results/agent_vs_no_agent_summary.txt. Next (user request): Java failure output (run the broken case) and definitions as context -> Java B/C/BC for local Qwen3-Coder and BC for particle-1, to fill the empty cells of that table.

- 2026-11-19 evening: Java context ablation done (docs/results/java_ablation_results.txt): failure output +6 points for Qwen3-Coder (BC +8), +10 for particle-1 BC (72.7%); definitions no help. Summary table updated (docs/results/agent_vs_no_agent_summary.txt): with the context, particle-1 BC (74.5% on the 59 agent cases) is slightly above the 80-call Qwen agent (relaxed 70.3%).

## Status update 2026-11-20: paid OpenRouter models (WP3 first step) done
- Five paid models (Qwen3-Coder-Next, Qwen3-Coder, Devstral-2512, GLM-5.3, Kimi-K2.5) on Java and Python with variants A and B: docs/results/paid_models_results.txt, summary in docs/results/agent_vs_no_agent_summary.txt. Spend 7.87 USD of 10 (Kimi 5.5). Failure output +6..+31 points for all; Kimi and the free particle-1 best (~58-60 Python, ~72-73 Java); Qwen3-Coder with B matches the 80-call local agent at ~0.0004 USD per case. Kimi extension and DeepSeek not run (budget / user decision).
- Remaining credit ~2.1 USD. Open: push (local commits), prompt/harness change for the agent, remote-model agent (needs a harness with API access outside the sandbox), more cases, writing.

- 2026-11-21: consolidated write-up of the paid/free model experiment with tables (A vs B per model and set), the A->B improvements, the cost analysis and the outcome: docs/results/paid_models_summary.md (generated by scripts/thesis_analysis/paid_models_summary_md.py); retroactive token/cost report: docs/results/paid_models_cost.txt. Correction: Kimi costs about 50-105x more per case than Qwen3-Coder (50-62x on the B runs), not 35-90x.
