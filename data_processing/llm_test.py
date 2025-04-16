import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from transformers import GenerationConfig
from data_processing.CodeBLEU.bleu import corpus_bleu
from data_processing.CodeBLEU.code_bleu import calc_code_bleu


class Tester_llm:
    def __init__(self, model_name, model, tokenizer, dataset_path, device="cuda"):
        self.model_name = model_name
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_path = Path(dataset_path)
        self.dataset_file = self.dataset_path / "splits" / "test.pkl"

        # Load dataset
        with open(self.dataset_file, "rb") as f:
            self.dataset = pickle.load(f)

    def build_prompt(self, row):
        if self.model_name in {"llama", "gemma"}:
            # ChatML style (used by llama3, gemma)
            system_prompt = "You are a helpful assistant that repairs test code given broken test and code changes."
            user_input = row["input"]
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        elif self.model_name in {"qwen", "deepseek"}:
            # Plain instruction-style prompt
            return f"### Task:\nRepair the broken test code based on the following input:\n\n{row['input']}\n\n### Repaired Test Code:\n"

        else:
            # Fallback: simple instruction prompt
            return f"Repair the following test:\n{row['input']}\n\nRepaired test:\n"

    def run(self, out_path=None, max_gen_tokens=256, save_json=True):
        self.model.eval()
        predictions = []

        for i in tqdm(range(len(self.dataset)), desc=f"Testing {self.model_name}"):
            row = self.dataset[i]
            prompt = self.build_prompt(row)
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids.to(self.device)

            # Generation config
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_gen_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )

            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract only generated continuation if needed
            if self.model_name in {"llama", "gemma"}:
                assistant_tag = "<|start_header_id|>assistant<|end_header_id|>\n"
                generated = generated.split(assistant_tag)[-1].rstrip()

            predictions.append({
                "ID": row["ID"] if "ID" in row else i,
                "target": row["output"],
                "preds": [generated.rstrip()]
            })

        if save_json and out_path:
            out_path = Path(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{self.model_name}_llm_test_predictions.json"
            pd.DataFrame(predictions).to_json(out_file, indent=2)
            print(f"Saved predictions to {out_file}")

            # Compute evaluation scores
        bleu, codebleu, em = self.compute_scores(predictions)
        print(f"Evaluation: BLEU={bleu} | CodeBLEU={codebleu} | EM={em}")

        return predictions

    def compute_scores(self, predictions):
        """
        predictions: list of {"ID", "target", "preds": [<best>, ...]}
        """
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
        print(f"BLEU: {round(bleu_score * 100, 2)} | CodeBLEU: {round(code_bleu_score, 2)} | EM: {em}")
        return round(bleu_score * 100, 2), round(code_bleu_score, 2), em