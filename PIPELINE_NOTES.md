# Pipeline notes: changes and what they did (keep short, append new entries)

## Dataset v2_pr59 (mining -> filtering), numbers for the text
| Stage | Cases | Repos | Commit pairs | Max cases / pair |
|---|---:|---:|---:|---:|
| mined (`annotated_cases_v2_pr59.csv`, 10,086 repos searched) | 16,914 | 477 | 3,748 | 624 |
| `same_pair_error_change_cap` (cap 5), `..._filtered.csv` | 14,414 (-14.8%) | 477 | 3,748 | 182 |
| `environment_errors` | 14,363 (-0.3%) | 476 | 3,739 | 182 |
| `non_repairs`, `..._filtered_cap_clean.csv` (final) | 14,090 (-1.9%) | 476 | 3,691 | 182 |

Chain: `16914 -> 14414 -> 14363 -> 14090`. Reports with every number: `<csv>.report.json|md`.
Run: `python data_processing/filter_cases.py in.csv out.csv --cap 5 --steps same_pair_error_change_cap,environment_errors,non_repairs`
(streams the file, never modifies the input, refuses to overwrite). Details of each step: docstring of `data_processing/filter_cases.py`.
- Why the cap: one source change breaks many tests that are fixed by the same substitution (RxPY: 624 cases in one commit pair, 566 with the same error). Error text alone is not enough (58% of same-pair-same-error groups hide different edits), so the group key is (commit pair, final error line, token-substitution pattern).
- Environment errors are matched strictly (exception class anchored); `FileNotFoundError` is NOT dropped (135 cases stay, mostly tests of files they create themselves).
- Not filtered, known limitations: contamination (65% of cases are 2026 commits, 20% from agent/LLM-tooling repos), mirrored cases, local paths in logs, 31% of cases are tests repaired in more than one pair (split by test/repo if that matters).

## Search (repository_search) fixes
- GitHub ignores the size filter when two size qualifiers are combined (`size:>=A size:<B` returned the unfiltered 183,162 repos for every band): all bands returned the same top repos and the run idled ~12 h per run. Now `size:A..B-1`.
- Walking up from 0 KB found ~7,600 tiny repos (7,146 without tests, ~10 cases/h); starting at 1000 KB (`SIZE_START=1000`) gave ~160-250 cases/h and 45-55% no-test repos. Most case-producing repos are large (median ~8.5 MB).
- All GitHub calls have a timeout (10 s connect, 60 s read) and 5 retries (`http_get`).
- Earlier: per-file blocking, circuit breaker, REQUIRE_SOURCE_CHANGE (see ~/CLAUDE.md of the cluster account for the history).

## Evaluation robustness (data_processing)
- `atomic_io.py`: every saved file / checkpoint directory is written to a temp name and swapped in; SIGTERM becomes a clean interrupt.
- `train.py`: full state after every epoch in `checkpoint-last` (weights, optimizer, scheduler, RNG, stats); `train()` resumes and ends with the same weights as an uninterrupted run; a restart never overwrites `checkpoint-best` (a fresh run keeps the old one as `checkpoint-best.previous`).
- `test.py` (CodeT5+/PLBART/CodeGen) and `llm_test.py`: predictions saved atomically every N cases / every batch, runs continue with the missing cases; LLM sampling seeded per batch (default 1234) so a resumed run equals an uninterrupted one.
- Metrics print "over N/M cases" and warn when the run is incomplete; `login()` only with an explicit token.
- Tests: `tests/test_resume_training.py`, `test_resume_testing.py`, `test_atomic_io.py`, `test_prediction_files.py` run the real Encoder/Trainer/Testers on 28 real cases (`tests/data/tiny_cases.csv`) with tiny random models, including a real `kill -9`.

## Environment (Helios)
- CodeBLEU needs `tree-sitter==0.24.0` (installed in `~/venv3.12` on 2026-11-06 with `pip --python ~/venv3.12/bin/python install tree-sitter==0.24.0`); `tests/test_metrics_environment.py` tells a missing pin from a bug.
- No vLLM installed (venv3.12 has torch 2.9, transformers 4.50.3, no peft); GPU jobs need an explicit subtype: `--gres=gpu:a100-40g:1` (also a100-80g, h200-141g, b200-180g). Compute nodes reach huggingface.co and pypi.org.
- `Encoder` reads the whole CSV with pandas: encode the 6.6 GB file in a high-memory Slurm job, not in a 4 GB session.

## Split and evaluation runs (2026-11-06)
- Split (`Encoder.split`): by commit and in time, train (oldest commits, 80%) < validation (oldest ~25% of the held-out commits) < test; no commit is in two sets (checked on the final data: 11,248 train / 2,842 held-out cases, 0 broken commits in both; 40 held-out cases (1.4%) start from a commit that is the repaired commit of a train case; 94 repos in both, 25% of held-out cases from unseen repos). Before, validation was 25% of every held-out commit, so validation and test shared commits.
- `slurm/`: `submit_pipeline.sh [--gpu TYPE] MODEL...` submits encode -> train -> test per model (models in parallel, `afterok` chain); `smoke.sbatch` runs all three models on a 162-case sample with real models (1 epoch, 20 test cases) and prints time and peak RAM/GPU memory; `run_stage.py` is the entry point (exit status 75 = stopped by the time limit, start the same stage again). Defaults: CSV = final filtered file, OUT = `~/eval_results`, beam 20, 10 epochs, batch 8, lr 5e-5, early stop 3.
- Queue facts (2026-11-06, `sbatch --test-only`): partition `main` was estimated to start in December even for 2 CPUs, `amd` (same nodes) at once, so encode uses `amd`; GPU jobs need a subtype (`--gres=gpu:a100-80g:1`) and were estimated 22 h to 2 days away (b200/h200 first). Estimates change.
- Input window (decided 2026-11-06): CodeT5+ and PLBART 1024 tokens (`Eftt.MODEL_MAX_LENGTH`, `--max-length` overrides), CodeGen its own 2048. On a 162-case sample 41% of the cases fit in 512 (66/162) but 75% in 1024 (121/123 of 162); CodeGen keeps 87% (141/162). So models are tested on different subsets: compare them on the cases every model can encode. Expect roughly 10,600 usable cases at 1024 for the full data (extrapolated, the real encode gives the number).
- Decoder-only fix: the encoder filled all of CodeGen's window with input and left no room for the repair (13 of 141 cases survived, empty validation set); the input now gets two thirds of the window (`Encoder.max_input_tokens`).
- Smoke test submitted 2026-11-06 as job 3478688 on one H200 (`sbatch --gres=gpu:h200-141g:1 slurm/smoke.sbatch`); log `slurm/logs/smoke-3478688.out`, results `~/eval_smoke`. Read the per-stage "finished in ... peak RAM ... peak GPU memory" lines to set the time limits of the full runs.

