STATE of the OpenRouter (free models) evaluation on the contrast subset - written 2026-11-16 ~08:30
=====================================================================================================

WHAT WAS DONE
- Subset: ~/eval_results_perproject_llm/contrast300/  (300 cases of the matched test set: 150 "solved" = at least one of the five
  reference models (CodeT5+, PLBART, CodeGen, Qwen3-Coder, gpt-oss-20b) has a top-5 candidate that passes the CHECKED plausibility
  rule; 150 "unsolved" = none does; at most 4 cases per repository and group). subset_meta.json has, per new ID: matched ID,
  group, repository, solved_by, and the group sizes of the population (solved 438, unsolved 537 of 975 cases scored for all five).
  The two groups are NOT representative: overall rates = (438 * solved-group rate + 537 * unsolved-group rate) / 975.
- Models (all ':free', 1 candidate per case, temperature 0.7, reasoning effort low, prompt version 1 of llm_test.py):
  baidu/ernie-5.5-flash-preview:free, particle-ai/particle-1:free, nvidia/nemotron-3-ultra-550b-a55b:free.
  First pass 8192 tokens, then the empty answers were asked again with 16384 tokens (--retry-empty).
- Runs/answers: ~/eval_openrouter/contrast300_<vendor>_<model>_free/ (test_predictions.json = raw answers, requests.jsonl = one line
  per request without prompt text, run_settings.json, retry_settings.json).
- Plausibility (corrected code, commit dff6219 + uncommitted openrouter_run.py): ~/eval_openrouter_plaus/<name>/0.8/plausibility_v3.json
  (first pass) and <name>_retry/0.8/plausibility_v3.json (only the cases that were empty in the first pass). Analysis merges them:
  python -I scripts/thesis_analysis/or_analyze.py <name> <retry file>   (logic: a retry verdict replaces the
  first-pass verdict of the same case; "checked" = passes AND (test needs the span OR the candidate still asserts)).
- No credit was used: key usage 0 / limit remaining 10 at the end (is_free_tier False = the 10 EUR credit exists).
  The runner refuses any slug without ':free' unless --allow-paid is given.

RESULTS (rank 1, checked scoring; weighted overall estimate with the population sizes above)
  model                      solved group   unsolved group   weighted   cases none of the five solved   answers still empty
  nemotron-3-ultra (free)    95/150 63.3%    13/150  8.7%     33.2%      13                               3
  particle-1 (free)          91/150 60.7%    15/150 10.0%     32.8%      15                               2
  ernie-5.5-flash (free)     77/150 51.3%    11/150  7.3%     27.1%      11                               45
  --- earlier models at rank 1 on the same subset (unsolved group 0 by construction) ---
  Qwen3-Coder                59/150 39.3%      0               17.7%
  gpt-oss-20b                58/150 38.7%      0               17.4%
  CodeT5+                    57/150 38.0%      0               17.1%
  CodeGen                    31/150 20.7%      0                9.3%
  PLBART                     15/150 10.0%      0                4.5%
  Sampling error of the weighted estimate is about +-2 points (150 cases per group).
  The first-pass numbers before the retry were: ernie 24.7%, particle-1 32.5%, nemotron 31.3%.

CAVEATS FOR THE THESIS
- Contamination: the new models are recent and may have seen these public repositories (known limitation).
- Free models are one candidate at temperature 0.7; the earlier models' rank-5 numbers (best of 5) are a different measure.
- Empty answers (reasoning used the whole token limit, or finish=error from the provider) count as failures; ernie still has 45.
- Plausibility = the test passes; 35-40% of cases are "not discriminating" (test passes without the repaired span), handled by the
  checked rule (assertion kept) - see PyTaRGET/PIPELINE_NOTES.md and the audit notes. Equivalent repairs written differently count.
- Code fences in answers: stripped (strip_code_fence in openrouter_run.py); 2 of ernie's first-pass answers had them.

HOW TO RESUME / REPEAT
- cd ~/PyTaRGET; HF_HUB_OFFLINE=1 PYTHONPATH=. ~/venv_vllm/bin/python -u -m data_processing.openrouter_run --model <slug>:free \
    --split-dir ~/eval_results_perproject_llm/contrast300 --out-dir ~/eval_openrouter/contrast300_<vendor>_<model>_free \
    --n 1 --reasoning-effort low --max-new-tokens 8192 --requests-per-minute 8 --workers 2 --max-requests 450
  (same command continues after a stop; add --retry-empty --max-new-tokens 16384 for the empty answers). The key is read from
  ~/.config/openrouter/key (mode 600), never printed.
- Plausibility: python -u -m data_processing.plausibility_test --results ~/eval_openrouter_plaus --models <name> --workdir <scratch>
  --workers 3 --k 1 --out-name plausibility_v3.json --span-control   (run from ~/PyTaRGET with PYTHONPATH=.; the layout dir needs
  splits/test.json (link to contrast300) and test_predictions.json with fences stripped).

NOT DONE / OPEN
- Not committed: data_processing/openrouter_run.py, build_contrast_subset.py (PyTaRGET, branch second-approach); PIPELINE_NOTES.md
  does not yet contain the contrast-subset results.
- Models with more candidates (n=5) for a rank-5 comparison; paid OpenRouter models (need explicit approval and a cost estimate first);
  Gemma-4 free was blocked by 429 (provider capacity) during the pilot.
- The five earlier models were not re-run with the OpenRouter prompt; the comparison is by plausibility on the same cases.
