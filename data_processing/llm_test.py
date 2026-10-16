import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from data_processing.metrics import exact_match_flags
from data_processing.CodeBLEU.bleu import corpus_bleu
from data_processing.CodeBLEU.code_bleu import calc_code_bleu

from data_processing.jCodeBLEU.bleu import corpus_bleu as jcorpus_bleu
from data_processing.jCodeBLEU.code_bleu import calc_code_bleu as jcalc_code_bleu





class Tester_llm:
    def __init__(self, model_name, model_path, dataset_path, token, tokenizer, train_fraction, batch_size=4, is_java=False, device="cuda",
                 generation=None):
        self.model_name = model_name
        self.model_path = model_path
        self.dataset_path = Path(dataset_path) / "splits" / "test.json"
        self.dataset = json.load(open(self.dataset_path, 'r'))
        self.token = token
        self.tokenizer = tokenizer
        self.train_fraction = train_fraction
        self.device = device
        self.batch_size = batch_size
        self.is_java = is_java
        self.max_prompt_tokens = 2048
        self.generation = self.resolve_generation(model_name, generation)
        self.model = None
        if "ref" in str(self.train_fraction):
            print("prompting will be done for java dataset")

    def load_model(self):
        """Loaded lazily, so computing metrics from saved predictions does not need a GPU."""
        if self.model is not None or self.is_java:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=self.token
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.model_max_length = self.max_prompt_tokens
        self.model.eval()  # Important!

    @staticmethod
    def resolve_generation(model_name, overrides=None):
        """
        Decoding settings of the instruction-tuned models. The defaults are the ones used for the first results
        (beam search with sampling at temperature 1.5, every beam returned as a candidate); pass e.g.
        {"do_sample": False} for plain beam search, or {"num_beams": 1, "num_return_sequences": 1} for greedy.
        "seed" (optional) seeds torch before sampling.
        """
        settings = {"do_sample": True, "num_beams": 4, "num_return_sequences": 4, "temperature": 1.5, "seed": None}
        if model_name == "deepseek":
            settings.update(num_beams=2, num_return_sequences=2)
        unknown = set(overrides or {}) - set(settings)
        if unknown:
            raise ValueError(f"Unknown generation settings: {sorted(unknown)}")
        settings.update(overrides or {})
        if settings["num_return_sequences"] > settings["num_beams"]:
            raise ValueError("num_return_sequences cannot exceed num_beams")
        return settings

    def generation_kwargs(self):
        settings = self.generation
        kwargs = {
            "do_sample": settings["do_sample"],
            "num_beams": settings["num_beams"],
            "num_return_sequences": settings["num_return_sequences"],
        }
        if settings["do_sample"]:
            kwargs["temperature"] = settings["temperature"]
        return kwargs

    def fit_prompt(self, row):
        """
        Build the prompt, dropping the lowest-priority hunks (they are stored most relevant first)
        until it fits max_prompt_tokens, so the tokenizer never cuts off the '### Repaired Code:' tail.
        """
        prompt = self.build_prompt(row)
        n_hunks = len(re.findall(r"\[<HUNK>](.*?)\[<\/HUNK>]", row["input"], re.DOTALL))
        for kept in range(n_hunks, 1, -1):
            if len(self.tokenizer.encode(prompt)) <= self.max_prompt_tokens:
                return prompt
            prompt = self.build_prompt(row, max_hunks=kept - 1)
        if len(self.tokenizer.encode(prompt)) > self.max_prompt_tokens:
            print("WARNING: prompt exceeds max_prompt_tokens even with a single hunk, it will be truncated.")
        return prompt

    def build_prompt(self, row, max_hunks=None):
        language = "Python"
        if "ref" in str(self.train_fraction):
            language = "Java"
        instruction = (
            f"You are given a full {language} test function, where some lines are broken by the changes in source code (marked explicitly).\n"
            "Using the source code changes, repair ONLY the broken lines.\n"
            "Output ONLY the repaired lines.\n"
            "Wrap the repaired lines inside [REPAIR] brackets, and do not add anything else."
        )
        if "REPAIREDTEST" in row["input"]:
            raise ValueError(
                "Test input contains a REPAIREDTEST block (ground-truth leak). "
                "Re-run reannotate() on this test.json to strip it."
            )
        test_context, broken_lines, helpful_hunks = self.extract_relevant_code(row["input"], max_hunks)
        return (
            f"### Instruction:\n{instruction}\n\n### Full Test Context:\n{test_context}\n\n### Broken Lines:\n{broken_lines}\n\n### Source Code Changes:\n{helpful_hunks}\n\n### Repaired Code:\n"
        )

    def extract_relevant_code(self, input_text, max_hunks=None):
        """
        Extract full test context, broken lines, and helpful hunks (only the first max_hunks, if given).
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
        if max_hunks is not None:
            hunk_match = hunk_match[:max_hunks]
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
        match = re.search(r"\[REPAIREDTEST](.*?)(?:\[/REPAIREDTEST]|\[</REPAIREDTEST>])", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[REPAIRED](.*?)(?:\[/REPAIRED]|\[</REPAIRED>])", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        match = re.search(r"\[.*?REPAIR.*?](.*?)(?:\[.*?/.*?REPAIR.*?])", prediction, re.DOTALL)
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
        match = re.search(r"\[.*?REPAIR.*?](.*?)(?=\[|$)", prediction, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"\[REPAIR](.*?)", prediction, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        else:
            # Fallback: return everything (maybe model ignored format)
            return prediction.rstrip()

    def extract_prediction(self, generated):
        """Cut the echoed prompt off a decoded generation and pull out the repaired lines."""
        parts = generated.split("### Repaired Code:")
        return self.postprocess_prediction(parts[1] if len(parts) > 1 else generated)

    def run(self, out_path=None, max_gen_tokens=256, save_json=True):
        self.load_model()
        if self.generation["seed"] is not None:
            torch.manual_seed(self.generation["seed"])
        print(f"Decoding settings: {self.generation}")
        predictions = []
        seen_ids = set()
        out_file = None

        if save_json and out_path:
            out_path = Path(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{self.model_name}_llm_test_predictions.json"

            # Load existing predictions if file exists
            if out_file.exists():
                print(f"Resuming from existing file: {out_file}")
                existing = pd.read_json(out_file)
                predictions = existing.to_dict("records")
                seen_ids = {int(row["ID"]) for row in predictions}

        # "ID" is the row's position in test.json, so skipped cases never shift the others
        remaining = [(i, row) for i, row in enumerate(self.dataset) if i not in seen_ids]
        failed_ids = []

        for batch_start in tqdm(range(0, len(remaining), self.batch_size), desc=f"Testing {self.model_name}"):
            batch = remaining[batch_start:batch_start + self.batch_size]
            prompts = [self.fit_prompt(row) for _, row in batch]
            results = self.safe_generate(prompts, max_gen_tokens)

            for (idx, row), generated in zip(batch, results):
                if generated is None:
                    failed_ids.append(idx)
                    continue
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions.append({
                    "ID": idx,
                    "target": row["output"],
                    "preds": [self.extract_prediction(gen) for gen in decoded],
                })

            # Save intermediate state
            if out_file is not None:
                pd.DataFrame(predictions).to_json(out_file, indent=2)

            torch.cuda.empty_cache()

        if failed_ids:
            print(f"{len(failed_ids)} cases ran out of memory and have no prediction (they are retried on the next "
                  f"run): IDs {failed_ids}")
        if not predictions:
            print("No predictions to score.")
            return predictions

        # Final scores
        bleu, codebleu, em, norm_em = self.compute_scores(predictions)
        print(f"Evaluation: BLEU={bleu} | CodeBLEU={codebleu} | EM={em} | EM (indentation-normalized)={norm_em} "
              f"(over {len(predictions)}/{len(self.dataset)} cases)")

        return predictions

    def safe_generate(self, prompts, max_gen_tokens):
        """
        Returns one tensor of generated sequences (num_return_sequences x length) per prompt, in order,
        or None for a prompt that cannot be generated even alone (out of memory).
        Handles the DeepSeek-specific workaround and halves the batch on OOM.
        """
        if self.model_name == "deepseek":
            results = []
            for prompt in prompts:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length",
                                            max_length=self.max_prompt_tokens,
                                            return_attention_mask=True).to(self.model.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_gen_tokens,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False,
                            **self.generation_kwargs(),
                        )
                    results.append(out)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("OOM on individual DeepSeek prompt. Skipping.")
                    results.append(None)
            return results

        # For all other models
        num_return_sequences = self.generation["num_return_sequences"]
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_prompt_tokens, return_attention_mask=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **self.generation_kwargs(),
                )
            # generate() returns the sequences of prompt 0 first, then prompt 1, ...
            return list(outputs.reshape(len(prompts), num_return_sequences, -1))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                print("OOM on a single prompt. Skipping.")
                return [None]
            mid = len(prompts) // 2
            print(f"OOM with batch size {len(prompts)}, retrying as two batches of size {mid}")
            return self.safe_generate(prompts[:mid], max_gen_tokens) + self.safe_generate(prompts[mid:], max_gen_tokens)

    def compute_scores(self, predictions):
        if len(predictions) == 0:
            raise ValueError("No predictions to score")
        pred_df = pd.DataFrame(predictions)
        eval_size = len(pred_df["ID"])
        em_size = 0
        norm_em_size = 0
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
            norm_em_size += exact_match_flags(beam_outputs, target)[1]
            best_preds.append(best_pred)
            targets.append(target)

        em = round(em_size / eval_size * 100, 2)
        norm_em = round(norm_em_size / eval_size * 100, 2)
        if self.is_java:
            print("java metrics")
            bleu_score = jcorpus_bleu([[t.split()] for t in targets], [p.split() for p in best_preds])
            code_bleu_score = jcalc_code_bleu([targets], best_preds, lang="java")
        else:
            bleu_score = corpus_bleu([[t.split()] for t in targets], [p.split() for p in best_preds])
            code_bleu_score = calc_code_bleu([targets], best_preds, lang="python")
        return round(bleu_score * 100, 2), round(code_bleu_score*100, 2), em, norm_em
