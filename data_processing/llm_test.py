import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from data_processing.CodeBLEU.bleu import corpus_bleu
from data_processing.CodeBLEU.code_bleu import calc_code_bleu





class Tester_llm:
    def __init__(self, model_name, model_path, dataset_path, token, device="cuda", batch_size=4):
        self.model_name = model_name
        self.model_path = model_path
        self.dataset_path = Path(dataset_path) / "splits" / "test.json"
        self.dataset = json.load(open(self.dataset_path, 'r'))
        self.token = token
        self.device = device
        self.batch_size = batch_size

        # Load model manually (no pipeline)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, token=self.token
        )
        if self.tokenizer.model_max_length > 10000:
            self.tokenizer.model_max_length = 2048
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=self.token
        )
        self.model.eval()  # Important!

    def build_prompt(self, row):
        instruction = (
            "You are given a full Python test function, where some lines are broken (marked explicitly).\n"
            "Using the helpful code changes, repair ONLY the broken lines.\n"
            "Output ONLY the repaired lines, without copying the whole function, and without adding explanations.\n"
            "repaired lines must be wrapped in [<REPAIR>] and [</REPAIR>] brackets."
        )
        test_context, broken_lines, helpful_hunks = self.extract_relevant_code(row["input"])

        if self.model_name in {"llama", "gemma"}:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{instruction}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Full Test Context:\n{test_context}\n\nBroken Lines:\n{broken_lines}\n\nHelpful Code Changes:\n{helpful_hunks}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self.model_name in {"qwen", "deepseek"}:
            return (
                f"### Instruction:\n{instruction}\n\n### Full Test Context:\n{test_context}\n\n### Broken Lines:\n{broken_lines}\n\n### Helpful Code Changes:\n{helpful_hunks}\n\n### Repaired Code:\n"
            )
        else:
            return (
                f"{instruction}\n\nFull Test Context:\n{test_context}\n\nBroken Lines:\n{broken_lines}\n\nHelpful Code Changes:\n{helpful_hunks}\n\nRepaired Code:\n"
            )

    def extract_relevant_code(self, input_text):
        """
        Extract full test context, broken lines, and helpful hunks.
        Returns (full_test_context, broken_lines, helpful_hunks)
        """
        # Extract full [<TESTCONTEXT>] block (including [<BREAKAGE>] etc.)
        testcontext_match = re.search(r"\[<TESTCONTEXT>](.*?)\[<REPAIRCONTEXT>]", input_text, re.DOTALL)
        testcontext_code = testcontext_match.group(1).strip() if testcontext_match else ""
        #print("test code: ", testcontext_code)
        # Extract broken lines
        breakage_match = re.search(r"\[<BREAKAGE>](.*?)\[<\/BREAKAGE>]", input_text, re.DOTALL)
        broken_lines = breakage_match.group(1).strip() if breakage_match else ""
        #print("broken lines: ", broken_lines)

        # Extract all [<HUNK>] repaired pieces
        hunk_matches = re.findall(r"\[<HUNK>](.*?)\[<\/HUNK>]", input_text, re.DOTALL)
        repaired_hunks = "\n\n".join(h.strip() for h in hunk_matches)
        #print("repaired hunks: ", repaired_hunks)

        return testcontext_code, broken_lines, repaired_hunks

    def postprocess_prediction(self, prediction):
        """
        Extract repaired lines enclosed within [<REPAIR>]...</REPAIR>] from model prediction.
        If no brackets are found, return the full prediction as fallback.
        """
        matches = re.findall(r"\[<REPAIR>](.*?)\[<\/REPAIR>]", prediction, re.DOTALL)
        if matches:
            # Join multiple repaired fragments if model predicted several
            repaired_code = "\n".join(m.strip() for m in matches)
            return repaired_code.strip()
        else:
            # Fallback: return everything (maybe model ignored format)
            return prediction.strip()


    def restore_formatting(self, text):
        text = re.sub(r'(?:\s*)<TAB>(?:\s*)', '    ', text)
        text = re.sub(r'(?:\s*)<NL>(?:\s*)', '\n', text)
        return text.rstrip()

    def run(self, out_path=None, max_gen_tokens=256, save_json=True):
        predictions = []
        all_prompts = [self.build_prompt(row) for row in self.dataset]

        for i in tqdm(range(0, len(all_prompts), self.batch_size), desc=f"Testing {self.model_name}"):
            batch_prompts = all_prompts[i:i+self.batch_size]
            batch_rows = self.dataset[i:i+self.batch_size]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, generated in enumerate(decoded_outputs):
                generated = self.restore_formatting(generated)
                generated = self.postprocess_prediction(generated)

                if self.model_name in {"llama", "gemma"}:
                    assistant_tag = "<|start_header_id|>assistant<|end_header_id|>\n"
                    if assistant_tag in generated:
                        generated = generated.split(assistant_tag)[-1].rstrip()

                predictions.append({
                    "ID": batch_rows[j].get("ID", i+j),
                    "target": batch_rows[j]["output"],
                    "preds": [generated.rstrip()]
                })

            torch.cuda.empty_cache()

        if save_json and out_path:
            out_path = Path(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{self.model_name}_llm_test_predictions.json"
            pd.DataFrame(predictions).to_json(out_file, indent=2)
            print(f"Saved predictions to {out_file}")

        bleu, codebleu, em = self.compute_scores(predictions)
        print(f"Evaluation: BLEU={bleu} | CodeBLEU={codebleu} | EM={em}")

        return predictions

    def compute_scores(self, predictions):
        pred_df = pd.DataFrame(predictions)
        eval_size = pred_df["ID"].nunique()
        em_size = 0
        best_preds = []
        targets = []

        for _, row in pred_df.iterrows():
            beam_outputs = row["preds"]
            target = row["target"]
            best_pred = beam_outputs[0]
            for output in beam_outputs:
                if output == target:
                    em_size += 1
                    best_pred = output
                    break
            best_preds.append(best_pred)
            targets.append(target)

        em = round(em_size / eval_size * 100, 2)
        bleu_score = corpus_bleu([[t.split()] for t in targets], [p.split() for p in best_preds])
        code_bleu_score = calc_code_bleu([targets], best_preds, lang="python")
        return round(bleu_score * 100, 2), round(code_bleu_score, 2), em
