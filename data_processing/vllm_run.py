"""
Generate repair candidates with a local vLLM model for every case of <split_dir>/splits/test.json.

Reuses the prompt of Tester_llm (same instruction, hunk trimming, [REPAIR] extraction), so the results are comparable
with the other instruction-tuned runs. Output: <out_dir>/test_predictions.json in the format of prediction_files
({"ID": position in test.json, "target", "preds": [n candidates]}), so plausibility_test and the report scripts work
unchanged. Resumable: cases already in the file are skipped.

  python -m data_processing.vllm_run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --split-dir ~/eval_results_perproject/codet5p/0.8 \
      --out-dir ~/eval_results_perproject_llm/qwen3coder --n 5 [--limit 50]
"""
import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from data_processing import prediction_files
from data_processing.llm_test import Tester_llm


PROMPT_V2_INSTRUCTION = (
    "You are given a full Python test function, where some lines are broken by the changes in source code (marked explicitly).\n"
    "Using the source code changes, repair the broken part of the test.\n"
    "Output the repaired version of the broken part: all of its lines that stay in the repaired test (including lines "
    "inside it that do not change), with their original indentation and line breaks. The repaired part may have more or "
    "fewer lines than the broken part, for example when lines are added or removed.\n"
    "Do not output code outside the broken part and do not add explanations.\n"
    "Wrap the repaired lines inside [REPAIR] brackets, and do not add anything else."
)


class TesterV2(Tester_llm):
    """Tester_llm with the instruction of prompt version 2 (same sections, the llm_test.py prompt stays unchanged)."""

    def build_prompt(self, row, max_hunks=None):
        if "REPAIREDTEST" in row["input"]:
            raise ValueError("Test input contains a REPAIREDTEST block (ground-truth leak).")
        test_context, broken_lines, helpful_hunks = self.extract_relevant_code(row["input"], max_hunks)
        return (f"### Instruction:\n{PROMPT_V2_INSTRUCTION}\n\n### Full Test Context:\n{test_context}\n\n"
                f"### Broken Lines:\n{broken_lines}\n\n### Source Code Changes:\n{helpful_hunks}\n\n### Repaired Code:\n")


def reindent(prediction, input_text):
    """
    Shift the prediction so that its first line has the indentation of the first broken line (the models of the
    gpt-oss family drop the leading indentation of the lines they write). Lines keep their relative indentation.
    """
    match = re.search(r"\[<BREAKAGE>](.*?)\[</BREAKAGE>]", input_text, re.DOTALL)
    first_broken = next((l for l in (match.group(1).split("\n") if match else []) if l.strip()), None)
    lines = prediction.split("\n")
    first_pred = next((l for l in lines if l.strip()), None)
    if first_broken is None or first_pred is None:
        return prediction
    shift = (len(first_broken) - len(first_broken.lstrip())) - (len(first_pred) - len(first_pred.lstrip()))
    if shift == 0:
        return prediction
    if shift > 0:
        return "\n".join(" " * shift + l if l.strip() else l for l in lines)
    return "\n".join(l[min(-shift, len(l) - len(l.lstrip())):] if l.strip() else l for l in lines)


def final_answer(text):
    """The answer part of a generation: gpt-oss writes 'analysis...assistantfinal<answer>'; a reasoning that was cut off
    before the answer gives an empty prediction instead of the reasoning text."""
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1]
    return "" if text.lstrip().startswith("analysis") else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-dir", required=True, help="directory with splits/test.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=5, help="candidates per case")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    ap.add_argument("--chunk", type=int, default=64, help="cases per saved chunk")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--reasoning-effort", default=None, help="gpt-oss: low | medium | high (chat template argument)")
    ap.add_argument("--prompt-version", type=int, choices=[1, 2], default=1,
                    help="1 = the instruction of llm_test.py, 2 = full broken part, any number of lines")
    ap.add_argument("--reindent", action="store_true", help="restore the indentation of the first broken line")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tester = (TesterV2 if args.prompt_version == 2 else Tester_llm)("vllm", args.model, args.split_dir, None, tokenizer, 0.8, is_java=False)
    tester.max_prompt_tokens = args.max_prompt_tokens

    out_file = Path(args.out_dir) / "test_predictions.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    if out_file.exists():
        predictions = prediction_files.load(out_file)
        prediction_files.check_matches_test_split(predictions, tester.dataset)
    seen = {r["ID"] for r in predictions}
    rows = list(enumerate(tester.dataset))[: args.limit]
    todo = [(i, r) for i, r in rows if i not in seen]
    print(f"{len(rows)} cases, {len(seen)} done, {len(todo)} to do", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, trust_remote_code=True, seed=1234)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens, seed=1234)

    def chat(prompt):
        # the base prompt ends with '### Repaired Code:\n'; sent as one user turn, the model answers in [REPAIR] brackets
        extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, **extra)

    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        prompts = [chat(tester.fit_prompt(r)) for _, r in chunk]
        outputs = llm.generate(prompts, params)
        for (i, row), out in zip(chunk, outputs):
            # gpt-oss writes its reasoning first ('analysis...assistantfinal<answer>' once special tokens are stripped)
            preds = [tester.postprocess_prediction(final_answer(c.text)) for c in out.outputs]
            if args.reindent:
                preds = [reindent(x, row["input"]) for x in preds]
            predictions.append({"ID": i, "target": row["output"], "preds": preds})
        prediction_files.save(predictions, out_file)
        print(f"saved {len(predictions)} cases", flush=True)

    json.dump({"model": args.model, "n": args.n, "temperature": args.temperature,
               "max_prompt_tokens": args.max_prompt_tokens, "reasoning_effort": args.reasoning_effort,
               "reindent": args.reindent, "prompt_version": args.prompt_version}, open(Path(args.out_dir) / "run_settings.json", "w"))
    bleu, codebleu, em, norm_em = tester.compute_scores(predictions)
    print(f"BLEU={bleu} CodeBLEU={codebleu} EM={em} EM_norm={norm_em} over {len(predictions)}/{len(tester.dataset)}")


if __name__ == "__main__":
    main()
