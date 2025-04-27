import pickle
import re
from pathlib import Path
import pandas as pd
from huggingface_hub import login
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from data_processing.CodeBLEU.bleu import corpus_bleu
from data_processing.CodeBLEU.code_bleu import calc_code_bleu


class Tester_llm:
    def __init__(self, model_name, model_path, model_class, dataset_path, token=None, device="cuda"):
        self.model_name = model_name
        self.model_path = model_path
        self.model_class = model_class
        self.device = device
        self.token = token
        self.dataset_path = Path(dataset_path)
        self.dataset_file = self.dataset_path / "splits" / "test.json"
        self.dataset = pd.read_json(self.dataset_file).to_dict(orient="records")

        # Authenticate if token provided
        if self.token:
            login(self.token)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, token=self.token
        )
        self.model = self.model_class.from_pretrained(
            self.model_path, trust_remote_code=True, token=self.token,
            device_map="auto", torch_dtype="auto",
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

    def run(self, out_path=None, max_gen_tokens=256, save_json=True):
        self.model.eval()
        predictions = []

        for i in tqdm(range(len(self.dataset)), desc=f"Testing {self.model_name}"):
            row = self.dataset[i]
            prompt = self.build_prompt(row)
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids.to(self.device)

            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_gen_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": False,
                }

                # Special handling for DeepSeek models
                if self.model_name.lower() == "deepseek":
                    gen_kwargs["use_cache"] = True
                    gen_kwargs["cache_position"] = torch.arange(0, input_ids.shape[1], device=input_ids.device)

                output_ids = self.model.generate(**gen_kwargs)
            def restore_formatting(text):
                text = re.sub(r'(?:\s*)<TAB>(?:\s*)', '    ', text)
                text = re.sub(r'(?:\s*)<NL>(?:\s*)', '\n', text)
                return text.rstrip()
            generated = restore_formatting(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))

            if self.model_name in {"llama", "gemma"}:
                assistant_tag = "<|start_header_id|>assistant<|end_header_id|>\n"
                generated = generated.split(assistant_tag)[-1].rstrip()

            predictions.append({
                "ID": row.get("ID", i),
                "target": row["output"],
                "preds": [generated.rstrip()]
            })

        if save_json and out_path:
            out_path = Path(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{self.model_name}_llm_test_predictions.json"
            pd.DataFrame(predictions).to_json(out_file, indent=2)
            print(f"Saved predictions to {out_file}")

        # Evaluate
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
