import json
import pickle
import re
from pathlib import Path
import pandas as pd
from huggingface_hub import login
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

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

        # Load model and tokenizer manually
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, token=self.token, device_map="auto", torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, token=self.token
        )

        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )

    def build_prompt(self, row):
        if self.model_name in {"llama", "gemma"}:
            system_prompt = "You are a helpful assistant that repairs test code given broken test and code changes."
            user_input = row["input"]
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self.model_name in {"qwen", "deepseek"}:
            return f"### Task:\nRepair the broken test code based on the following input:\n\n{row['input']}\n\n### Repaired Test Code:\n"
        else:
            return f"Repair the following test:\n{row['input']}\n\nRepaired test:\n"

    def restore_formatting(self, text):
        text = re.sub(r'(?:\s*)<TAB>(?:\s*)', '    ', text)
        text = re.sub(r'(?:\s*)<NL>(?:\s*)', '\n', text)
        return text.rstrip()

    def run(self, out_path=None, max_gen_tokens=256, save_json=True):
        predictions = []
        total = len(self.dataset)
        for i in tqdm(range(0, total, self.batch_size), desc=f"Testing {self.model_name}"):
            batch_rows = self.dataset[i:i + self.batch_size]
            prompts = [self.build_prompt(row) for row in batch_rows]

            outputs = self.pipe(
                prompts,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
                eos_token_id=self.pipe.tokenizer.eos_token_id,
                return_full_text=False,
            )

            for j, output in enumerate(outputs):
                print(output)
                generated = output[0]["generated_text"]
                generated = self.restore_formatting(generated)

                if self.model_name in {"llama", "gemma"}:
                    assistant_tag = "<|start_header_id|>assistant<|end_header_id|>\n"
                    if assistant_tag in generated:
                        generated = generated.split(assistant_tag)[-1].rstrip()

                predictions.append({
                    "ID": batch_rows[j].get("ID", i + j),
                    "target": batch_rows[j]["output"],
                    "preds": [generated.rstrip()]
                })

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
