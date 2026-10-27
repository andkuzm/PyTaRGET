# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PyTaRGET** (Python Test Case Repair using Generative models and EnhancemenT) is a research ML pipeline for automated repair of broken Python (and Java) test cases. It covers three stages: dataset mining from GitHub, model fine-tuning, and evaluation.

Python 3.12+ is required.

---

## Setup

```bash
pip install -r requirements.txt
pip install tree-sitter==0.24.0   # intentional version pin; pip will warn about conflicts — expected
```

For DeepSeek support only (pip version is too old as of the time of writing):
```bash
git clone https://github.com/HuggingFace/transformers.git
cd transformers && pip install -e .
```

The working directory when running code must be **the parent of PyTaRGET** (not inside it), because imports are resolved relative to that root. In Jupyter notebooks the README's sys.path block must be run first; in regular Python/IDE usage the project root is usually picked up automatically.

---

## Main Workflow

All experiment logic lives in `data_processing/encode_tune_test.py` via the `Eftt` class (Encode–Fine-tune–Test).

### Fine-tuned models (`codet5p`, `plbart`, `codegen`)

```python
from data_processing.encode_tune_test import Eftt
from pathlib import Path

experiment = Eftt(
    annotated_cases_path=Path("annotated_cases.csv"),
    out_path=Path("data_processing/results"),
    model="codet5p",       # codet5p | plbart | codegen
    train_size=0.8,
    beam_size=20,
    hftoken=None,
    batch_size=1,
    java=False,
)

experiment.encode()     # preprocess CSV → pickled train/valid/test datasets
experiment.train()      # fine-tune; saves best checkpoint
experiment.validate()   # beam-search inference → BLEU / CodeBLEU / Exact Match
experiment.get_metrics()  # recompute metrics from an existing predictions file
```

### Instruction-tuned LLMs (`qwen`, `qwen3`, `deepseek`, `gemma`)

```python
experiment = Eftt(
    annotated_cases_path=Path("annotated_cases.csv"),
    out_path=Path("data_processing/results"),
    model="qwen",
    train_size=0.0,        # 1-train_size fraction is used as test set
    beam_size=5,
    hftoken="<HuggingFace token>",  # required; gemma needs model-access grant
    batch_size=1,
    java=False,
)

experiment.encode()
experiment.validate_llm()   # loads LLM, generates predictions, appends incrementally
experiment.get_metrics_llm()
```

### Java instruction-tuned evaluation

1. Produce a Java annotated JSON using the [TaRGET](https://github.com/Ahmadreza-SY/TaRGET) pipeline.
2. Reannotate it: `eftt_instance.reannotate("path/to/annotated_java_dataset.json")` → produces `test.json`.
3. Move `test.json` to `data_processing/results/<model>/ref/splits/test.json`.
4. Create `Eftt` with `train_size="ref"` — the string `"ref"` triggers Java prompting (see `llm_test.py:32,48`).
5. Run `validate_llm()` then, in a TaRGET virtual environment (Java CodeBLEU deps), run `get_metrics_llm()` with `java=True`.

---

## Dataset Mining

**Interactive CLI (primary usage):** run the file directly from any working directory:

```bash
python repository_search/GitHubSearch.py
```

It prompts for a GitHub token and a version name, then derives all paths from the CWD:
- `annotated_cases_{version}.csv` — output dataset
- `processed_repositories_{version}.txt` — blacklist (resume file)
- `repos_{version}/` — temporary clone directory

If those files already exist the run resumes automatically, skipping already-processed repos.

**Programmatic use:**

```python
from repository_search.GitHubSearch import GitHubSearch

searcher = GitHubSearch(
    github_token="<token with public read>",
    version="v1",    # all paths derived from CWD + version name
)
searcher.find_and_process_repositories()
```

- Searches ~3,000 repos (3 queries × 1,000, sorted by stars). To go beyond that, tighten the size bounds in `find_and_process_repositories` at `GitHubSearch.py:197`.
- `processed_repositories_{version}.txt` (written to CWD) is a blacklist (`owner/repo|commit` or `owner/repo`). Entries prevent re-processing and guard against repos that break installed modules via `pip install -e .` side-effects.
- Tests inside repos are run via subprocess. To resolve PYTHONPATH inside the subprocess, every repo is installed with `pip install -e .`; this can corrupt packages in the outer environment — the blacklist exists partly for this reason.

---

## Architecture

### Data flow

```
annotated_cases.csv
       │
  encode.py (Eftt.encode)
       │  • filter non-ASCII, NaN
       │  • HunkPrioritizer (TF-IDF) trims hunks to fit context window
       │  • injects special tokens → input/output pairs
       │  • commit-based stratified train/valid/test split
       ▼
  train.pkl / valid.pkl / test.pkl  (pickled Dataset objects)
       │
  train.py (Eftt.train)          ← fine-tuned path
       │  • AdamW lr=5e-5, cosine scheduler
       │  • 10 epochs, early stop after 3 non-improving epochs
       │  • accelerate (DDP-ready), batch_size param
       ▼
  checkpoint-best/
       │
  test.py (Eftt.validate)        ← fine-tuned path
  llm_test.py (Eftt.validate_llm) ← LLM path (float16, appends incrementally)
       │
  test_predictions.json / <model>_llm_test_predictions.json
       │
  CodeBLEU/ or jCodeBLEU/  →  BLEU + CodeBLEU + Exact Match
```

### Results directory layout

```
data_processing/results/
└── <model>/
    └── <train_fraction>/          # e.g. 0.8, 0.0, "ref"
        ├── splits/                # train.json, valid.json, test.json (raw rows)
        ├── train.pkl / valid.pkl / test.pkl
        ├── model/ + tokenizer/ + checkpoint-best/
        ├── training_stats.json
        ├── test_predictions.json
        └── <model>_llm_test_predictions.json
```

Pre-computed results are in `data_processing/results.zip`; extract there to reproduce without re-encoding.

### Special token vocabulary

| Token | Meaning |
|---|---|
| `[<TESTCONTEXT>]` / `[</TESTCONTEXT>]` | Full broken test function |
| `[<BREAKAGE>]` / `[</BREAKAGE>]` | Lines broken by source changes |
| `[<REPAIRCONTEXT>]` / `[</REPAIRCONTEXT>]` | Source code diff |
| `[<HUNK>]` / `[</HUNK>]` | Individual diff hunk |
| `[<REPAIREDTEST>]` / `[</REPAIREDTEST>]` | Target repaired test |

Defined in `data_processing/encode.py` (`Tokens` class); added to tokenizer vocabulary before training.

### Dataset classes (`data_processing/dataset_types.py`)

| Class | Used by |
|---|---|
| `EncDecDataset` | CodeT5+, PLBART |
| `PLBARTDataset` | PLBART (language-id special handling) |
| `CodeGenDataset` | CodeGen (causal LM) |
| `LLMSeqDataset` | All instruction-tuned LLMs |

### CodeBLEU

`data_processing/CodeBLEU/` (Python) and `data_processing/jCodeBLEU/` (Java) implement:

```
CodeBLEU = 0.25 × ngram_BLEU
         + 0.25 × weighted_ngram_BLEU  (keyword-weighted)
         + 0.25 × syntax_match         (tree-sitter AST)
         + 0.25 × dataflow_match       (variable flow)
```

Java CodeBLEU requires the TaRGET virtual environment with its own Java parser dependencies.

### Key design decisions

- **Commit-based split**: entire commits go to one split to prevent data leakage across train/valid/test.
- **Hunk prioritization**: `prioritizer.py` uses TF-IDF cosine similarity to select the most relevant diff hunks when the full diff exceeds token budget.
- **Incremental LLM output**: `validate_llm()` appends predictions as they are produced so interrupted runs can resume without reprocessing.
- **Java language detection**: controlled by the string value `"ref"` passed as `train_size`, not by the `java` bool. The `java` bool only switches the CodeBLEU implementation used for metrics.
