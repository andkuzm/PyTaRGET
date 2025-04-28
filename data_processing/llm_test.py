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
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=self.token
        )
        self.model.eval()  # Important!

    def build_prompt(self, row): #BLEU=2.29 | CodeBLEU=0.49 | EM=0.0
        instruction = (
            "Only repair the specific broken lines inside the given code context. "
            "Output only the corrected lines, without any extra explanation, comments, or code outside the necessary fix."
        )

        if self.model_name in {"llama", "gemma"}:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{instruction}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{row['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self.model_name in {"qwen", "deepseek"}:
            return (
                f"### Instruction:\n{instruction}\n\n### Input:\n{row['input']}\n\n### Corrected Lines:\n"
            )
        else:
            return (
                f"{instruction}\nInput:\n{row['input']}\nCorrected Lines:\n"
            )

    def postprocess_prediction(self, prediction):
        # Remove Markdown code block markers if present
        prediction = prediction.strip()
        if prediction.startswith("```"):
            prediction = prediction.split("```")[1].strip()
        if prediction.startswith("python"):
            prediction = prediction[len("python"):].strip()
        # Also remove trailing ``` if still present
        prediction = prediction.split("```")[0].strip()

        # Remove common unnecessary sections introduced by some models
        prediction = re.sub(r"###.*", "", prediction).strip()

        return prediction


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
