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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=self.token
        )
        if self.model.config.is_encoder_decoder:
            self.tokenizer.padding_side = "left"
        self.model.eval()  # Important!

    def build_prompt(self, row):
        instruction = (
            "You are given a full Python test function, where some lines are broken by the changes in source code (marked explicitly).\n"
            "Using the source code changes, repair ONLY the broken lines.\n"
            "Output ONLY the repaired lines.\n"
            "Wrap the repaired lines inside [REPAIR] brackets, and do not add anything else."
        )
        test_context, broken_lines, helpful_hunks = self.extract_relevant_code(row["input"])

        if self.model_name in {"llama3", "llama4", "gemma"}:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{instruction}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Full Test Context:\n{test_context}\n\nBroken Lines:\n{broken_lines}\n\nSource Code Changes:\n{helpful_hunks}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self.model_name in {"qwen", "deepseek"}:
            return (
                f"### Instruction:\n{instruction}\n\n### Full Test Context:\n{test_context}\n\n### Broken Lines:\n{broken_lines}\n\n### Source Code Changes:\n{helpful_hunks}\n\n### Repaired Code:\n"
            )
        else:
            return (
                f"{instruction}\n\nFull Test Context:\n{test_context}\n\nBroken Lines:\n{broken_lines}\n\nSource Code Changes:\n{helpful_hunks}\n\nRepaired Code:\n"
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
        hunk_match = re.findall(r"\[<HUNK>](.*?)\[<\/HUNK>]", input_text, re.DOTALL)
        repaired_hunks = "\n\n".join(h.strip() for h in hunk_match)
        #print("repaired hunks: ", repaired_hunks)

        return testcontext_code, broken_lines, repaired_hunks

    def postprocess_prediction(self, prediction):
        """
        Extract repaired lines enclosed within [<REPAIR>]...</REPAIR>] from model prediction.
        If no brackets are found, return the full prediction as fallback.
        """
        match = re.search(r"\[REPAIR](.*?)\[/REPAIR]", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[REPAIR](.*?)(?:\[/REPAIR]|\[</REPAIR>])", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[REPAIR](.*?)\[<REPAIR>]", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[REPAIR](.*?)\[/?REPAIR", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[REPAIR](.*?)\[", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[REPAIR](.*?)", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        else:
            # Fallback: return everything (maybe model ignored format)
            return prediction.rstrip()

    def run(self, out_path=None, max_gen_tokens=256, save_json=True):
        predictions = []
        all_prompts = [self.build_prompt(row) for row in self.dataset]

        for i in tqdm(range(0, len(all_prompts), self.batch_size), desc=f"Testing {self.model_name}"):
            batch_prompts = all_prompts[i:i + self.batch_size]
            batch_rows = self.dataset[i:i + self.batch_size]

            try:
                outputs = self.safe_generate(batch_prompts, max_gen_tokens)
            except RuntimeError as e:
                print(f"Batch starting at index {i} failed permanently: {e}")
                continue

            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j in range(len(batch_rows)):
                preds = []
                for k in range(4):  # 4 sequences per input
                    idx = j * 4 + k
                    gen = decoded_outputs[idx]
                    if self.model_name == "gemma":
                        if len(gen.split("**Output:**")) > 1:
                            gen = self.postprocess_prediction(gen.split("**Output:**")[1])
                        elif len(gen.split("```")) > 2:
                            gen = self.postprocess_prediction(gen.split("```")[2])
                        elif len(gen.split("**")) > 2:
                            gen = self.postprocess_prediction(gen.split("**")[2])
                        else:
                            gen = self.postprocess_prediction(gen)
                    if self.model_name == "qwen":
                        if len(gen.split("### Repaired Code:")) > 1:
                            gen = self.postprocess_prediction(gen.split("### Repaired Code:")[1])
                        else:
                            gen = self.postprocess_prediction(gen)
                    if self.model_name == "deepseek":
                        if len(gen.split("### Repaired Code:")) > 1:
                            gen = self.postprocess_prediction(gen.split("### Repaired Code:")[1])
                        else:
                            gen = self.postprocess_prediction(gen)
                    preds.append(gen)
                    
                    
                predictions.append({
                    "ID": batch_rows[j].get("ID", i + j),
                    "target": batch_rows[j]["output"],
                    "preds": preds
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

    def safe_generate(self, prompts, max_gen_tokens):
        """
        Handles generation with DeepSeek-specific workaround and fallbacks for OOM.
        """
        if self.model_name == "deepseek":
            outputs = []
            for prompt in prompts:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True,
                                            return_attention_mask=True).to(self.model.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_gen_tokens,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False,
                            num_beams=4,
                            temperature=1.5,
                            num_return_sequences=4,
                        )
                    outputs.append(out)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("OOM on individual DeepSeek prompt. Skipping.")
            return torch.cat(outputs, dim=0) if outputs else torch.empty(0, dtype=torch.long, device=self.model.device)

        # For all other models
        batch_size = len(prompts)
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                    return_attention_mask=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                return self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=4,
                    temperature=1.5,
                    num_return_sequences=4,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError()
            else:
                mid = batch_size // 2
                print(f"OOM with batch size {batch_size}, retrying as two batches of size {mid}")
                first = self.safe_generate(prompts[:mid], max_gen_tokens)
                second = self.safe_generate(prompts[mid:], max_gen_tokens)
                return torch.cat([first, second], dim=0)

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
