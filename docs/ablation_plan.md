# WP5 context ablation: what extra context helps the lesser model? (plan and implementation, written 2026-11-18)

## Question
The prompt of our LLM runs holds the broken test, the broken lines and up to 4 source hunks. Which additional context improves the repair,
and is it worth its tokens? Each source is added ALONE (and then together) to a fixed baseline, on the same cases, with the same model and the
same scoring, so a change in the result is attributable.

## Runs
| Run | Prompt | Source of the extra text |
|---|---|---|
| A | baseline (existing prompt, version 1 of `llm_test.py`) | - (already available: contrast300 results) |
| B | + `### Test Failure Output:` | dataset column `outdated_test_log`: the pytest output of the broken test (cleaned: marker, rulers and machine paths removed; at most 2500 characters, head and tail) |
| C | + `### Definitions of Changed Symbols:` | signatures and first docstring lines of the identifiers of the broken lines, looked up (`ast`) in the project's NON-test source at the repaired commit; at most 8 symbols, 2 definitions each, 3000 characters |
| BC | both | B and C together (is the gain additive or overlapping?) |

Cases: the 300-case Python contrast subset (`~/eval_results_perproject_llm/contrast300`: 150 solved by at least one earlier model, 150 not).
Models: a local one (Qwen3-Coder via vLLM, Slurm job - ask before submitting) and a free OpenRouter one (e.g. particle-1), one candidate each.
Scoring: plausibility with the checked rule (`data_processing/plausibility_test.py`, `--span-control`), as for every other Python result.

## Measures
- plausible@1 per run, per group (solved / unsolved) and weighted with the population sizes of the subset (as in the contrast results);
- extra tokens per case (prompt length of the run minus A) and the **gain per 1000 extra tokens**;
- interaction: BC against B + C (overlap or complement);
- noise: 150 cases per group give about +-2-3 points of sampling error, so a gain below that is not called real without more cases/repeats.

## Leakage rules (why the numbers can be trusted)
1. The failure output is the run of the OLD test (before the repair), so it cannot contain the repaired code.
2. Definitions come from source files only: test directories (`test`, `tests`, `testing`, ...), `test_*.py`, `*_test.py`, `conftest.py` and the case's own
   test file are excluded from the index.
3. `context_ablation build` reports how often the reference repair appears verbatim in the added context (expected ~0; smoke test: 0 of 6).
4. Limit that remains: the definitions are looked up at the repaired commit, i.e. the source state in which the test was repaired. That is the
   state a developer has after changing the source (the source diff is already in the prompt), but a name that exists only after the change appears
   there; names that were renamed away have no definition (then C adds nothing for that case - this is expected, not an error).
5. Contamination of recent models remains a known limitation.

## Implementation
- `data_processing/context_ablation.py`: `failure_context`, `symbols_from_broken`, `index_source`, `definitions_context`, `add_sections`, `attach`,
  and the CLI `build` (partial clone of each project at its repaired commit, resumable, the clone is deleted afterwards, coverage and leak report).
- The runners `vllm_run.py` and `openrouter_run.py` take `--context-variant B|C|BC --context-file <context.json>` (both flags together; the variant is
  written to run_settings.json). The extra sections are inserted before `### Repaired Code:` and the prompt-fitting (drop the lowest-priority hunks) is
  unchanged, so a long extra context costs hunks first.
- Tests: `tests/test_context_ablation.py` (16 tests).

## How to run (nothing has been run yet except a 6-case smoke test of `build`)
```
cd ~/PyTaRGET
# 1) context file for the 300 cases (clones ~100 projects partially; scratch dir, a few minutes, not on a login node)
PYTHONPATH=. ~/venv3.12/bin/python -m data_processing.context_ablation build --split-dir ~/eval_results_perproject_llm/contrast300 \
    --out ~/eval_ablation/context.json --workdir <scratch>
# 2) a free remote model, one variant (repeat with C and BC; the baseline A exists)
HF_HUB_OFFLINE=1 PYTHONPATH=. ~/venv_vllm/bin/python -m data_processing.openrouter_run --model particle-ai/particle-1:free \
    --split-dir ~/eval_results_perproject_llm/contrast300 --out-dir ~/eval_ablation/particle1_B --n 1 --reasoning-effort low \
    --max-new-tokens 8192 --requests-per-minute 8 --workers 2 --context-variant B --context-file ~/eval_ablation/context.json
# 3) local model: same flags on vllm_run.py inside slurm/run_llm.sbatch (EXTRA_ARGS="--context-variant B --context-file ...") - ask before submitting
# 4) plausibility of each run: plausibility_test.py as in PIPELINE_NOTES.md
```

## Open
- Whether D (similar tests from the repository) and a second attempt with the failure output of the first candidate (execution feedback) follow after B/C.
- CodeT5+: the same context would need retraining on the augmented input (1024-token window); see the discussion in the thesis plan.
- Java: the same idea needs the Java failure output (not in TaRBench rows) and a Java definition lookup (tree-sitter); not planned yet.
