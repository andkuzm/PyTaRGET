"""
Candidates from a model on OpenRouter for the cases of <split_dir>/splits/test.json, with the same prompt (version 1 of
llm_test.py) and the same output format as vllm_run.py (test_predictions.json: ID = position in test.json), so the
plausibility scripts work unchanged.

Resumable: cases already in the file are skipped, so the same command continues where it stopped (also the next day,
after the daily limit of a free model was reached: the run then stops cleanly instead of failing).

The key is read from the environment (OPENROUTER_API_KEY) or from ~/.config/openrouter/key; it is never printed or
written anywhere. WHAT IS SENT: the prompt of each case (test function, broken lines, source diff of a public repository).

  # a fixed random sample of 300 cases (same sample for every model with the same --seed), 3 parallel requests
  python -m data_processing.openrouter_run --model google/gemma-4-31b-it:free --split-dir ~/eval_results_perproject_llm/matched \
      --out-dir ~/eval_openrouter/gemma-4-31b --sample 300 --seed 7 --workers 3 --n 1
"""
import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from data_processing import prediction_files
from data_processing.llm_test import Tester_llm
from data_processing.plausibility_test import iter_rows


def read_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        key = (Path.home() / ".config" / "openrouter" / "key").read_text().strip()
    return key.strip()


class RateLimiter:
    """At most `per_minute` requests per minute over all threads."""

    def __init__(self, per_minute):
        self.gap, self.next, self.lock = 60.0 / per_minute, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.next = max(self.next, now) + self.gap
            delay = self.next - self.gap - now
        if delay > 0:
            time.sleep(delay)


def strip_code_fence(text):
    """A model that answers with a Markdown block (```python ... ```) instead of [REPAIR] brackets: keep the code inside."""
    if "```" not in text:
        return text
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("```")), None)
    if start is None:
        return text
    end = next((i for i in range(len(lines) - 1, start, -1) if lines[i].strip().startswith("```")), None)
    inner = lines[start + 1:end] if end is not None else lines[start + 1:]
    return "\n" + "\n".join(inner).rstrip() if text.startswith("\n") else "\n".join(inner).rstrip()


class DailyLimit(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model slug, e.g. vendor/model:free")
    ap.add_argument("--split-dir", required=True, help="directory with splits/test.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1, help="candidates per case (one request each)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases")
    ap.add_argument("--java", action="store_true", help="Java cases (TaRBench rows): the Java prompt of llm_test.py")
    ap.add_argument("--sample", type=int, default=None, help="a random sample of N cases (fixed by --seed)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default=None, help="low | medium | high (OpenRouter's unified reasoning option)")
    ap.add_argument("--requests-per-minute", type=float, default=15, help="the free tier allows 20")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (paid models only need the rate limit)")
    ap.add_argument("--max-requests", type=int, default=None, help="stop after this many requests (a daily quota)")
    ap.add_argument("--max-cost", type=float, default=0.0, help="stop when the reported cost (USD) exceeds this")
    ap.add_argument("--retry-empty", action="store_true",
                    help="ask again for the cases whose saved answer is empty (e.g. with a larger --max-new-tokens); the old "
                         "record is kept until a new answer replaces it")
    ap.add_argument("--allow-paid", action="store_true",
                    help="allow a model whose slug does not end in ':free' (it spends credit); refused by default")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="only to size the prompt (hunks are dropped to fit)")
    args = ap.parse_args()
    if not args.model.endswith(":free") and not args.allow_paid:
        raise SystemExit(f"{args.model} is not a ':free' model and would spend credit; pass --allow-paid to allow it")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tester = Tester_llm.__new__(Tester_llm)           # prompt building only; no model, no 1.4 GB json in memory
    tester.model_name, tester.tokenizer, tester.train_fraction = "openrouter", tokenizer, ("ref" if args.java else 0.8)
    tester.max_prompt_tokens, tester.is_java = args.max_prompt_tokens, False

    out_dir = Path(args.out_dir)
    out_file = out_dir / "test_predictions.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = prediction_files.load(out_file) if out_file.exists() else []
    seen = {r["ID"] for r in predictions}
    if args.retry_empty:
        seen -= {r["ID"] for r in predictions if not any(x.strip() for x in r["preds"])}

    split = Path(args.split_dir) / "splits" / "test.json"
    chosen = None
    if args.sample is not None:
        total = sum(1 for _ in iter_rows(split))
        chosen = set(random.Random(args.seed).sample(range(total), min(args.sample, total)))

    todo = []
    for i, row in enumerate(iter_rows(split)):
        if args.limit is not None and i >= args.limit:
            break
        if (chosen is not None and i not in chosen) or i in seen:
            continue
        todo.append((i, row["output"], tester.fit_prompt(row)))
        if chosen is not None and len(todo) + len(seen & chosen) >= len(chosen):
            break
    print(f"{len(todo)} cases to do, {len(seen)} already done", flush=True)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=read_key(), max_retries=0)
    limiter, lock = RateLimiter(args.requests_per_minute), threading.Lock()
    state = {"cost": 0.0, "requests": 0, "errors": 0, "stop": False, "tokens_in": 0, "tokens_out": 0}
    log = open(out_dir / "requests.jsonl", "a", encoding="utf-8")        # one line per request, no prompt text

    def request(i, prompt):
        for attempt in range(6):
            if state["stop"]:
                raise DailyLimit("stopped")
            if args.max_requests is not None and state["requests"] >= args.max_requests:
                state["stop"] = True
                raise DailyLimit("request budget reached")
            limiter.wait()
            started = time.time()
            try:
                with lock:
                    state["requests"] += 1
                r = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature, max_tokens=args.max_new_tokens,
                    **({"extra_body": {"reasoning": {"effort": args.reasoning_effort}}} if args.reasoning_effort else {}))
                text = (r.choices[0].message.content or "") if r.choices else ""
                usage = r.usage
                with lock:
                    state["cost"] += float(getattr(usage, "cost", 0) or 0) if usage else 0.0
                    state["tokens_in"] += getattr(usage, "prompt_tokens", 0) or 0
                    state["tokens_out"] += getattr(usage, "completion_tokens", 0) or 0
                    log.write(json.dumps({"id": i, "secs": round(time.time() - started, 1),
                                          "finish": r.choices[0].finish_reason if r.choices else None,
                                          "chars": len(text),
                                          "completion_tokens": getattr(usage, "completion_tokens", None)}) + "\n")
                    log.flush()
                return text
            except DailyLimit:
                raise
            except Exception as e:                       # rate limit, provider hiccup, daily limit of the free tier
                msg = str(e)[:300]
                if "per-day" in msg or "free-models-per-day" in msg:
                    state["stop"] = True
                    raise DailyLimit(msg)
                with lock:
                    state["errors"] += 1
                print(f"case {i} attempt {attempt}: {msg}", flush=True)
                time.sleep(min(60, 5 * 2 ** attempt))
        return None                                     # the provider never answered: the case stays open for a later run

    def work(item):
        i, target, prompt = item
        try:
            texts = [request(i, prompt) for _ in range(args.n)]
        except DailyLimit as e:
            return None, str(e)
        if any(t is None for t in texts):                # provider errors, not an answer of the model: do not record it
            with lock:
                state["failed"] = state.get("failed", 0) + 1
                print(f"case {i}: provider gave no answer after the retries, left open", flush=True)
            return i, None
        preds = [strip_code_fence(tester.postprocess_prediction(t)) for t in texts]
        with lock:
            predictions[:] = [r for r in predictions if r["ID"] != i] + [{"ID": i, "target": target, "preds": preds}]
            prediction_files.save(predictions, out_file)
            print(f"case {i}: saved ({len(predictions)} done, requests {state['requests']}, errors {state['errors']}, "
                  f"cost ${state['cost']:.4f})", flush=True)
            if args.max_cost and state["cost"] > args.max_cost:
                state["stop"] = True
        return i, None

    stopped = None
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, why in pool.map(work, todo):
            if why and not stopped:
                stopped = why
    if stopped:
        print(f"stopped early: {stopped}. Run the same command again to continue.", flush=True)
    print(f"done: {len(predictions)} cases in the file, {state.get('failed', 0)} left open after provider errors, requests {state['requests']}, tokens in {state['tokens_in']} "
          f"out {state['tokens_out']}, cost ${state['cost']:.4f}", flush=True)
    json.dump({"model": args.model, "n": args.n, "temperature": args.temperature, "prompt_version": 1,
               "retry_empty": args.retry_empty,
               "max_prompt_tokens": args.max_prompt_tokens, "reasoning_effort": args.reasoning_effort,
               "max_new_tokens": args.max_new_tokens, "sample": args.sample, "seed": args.seed},
              open(out_dir / ("retry_settings.json" if args.retry_empty else "run_settings.json"), "w"))


if __name__ == "__main__":
    main()
