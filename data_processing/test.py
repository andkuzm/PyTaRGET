import pandas as pd
from pathlib import Path
from accelerate import Accelerator
from datetime import datetime
from tqdm import tqdm

from data_processing.CodeBLEU.bleu import corpus_bleu
from data_processing.CodeBLEU.code_bleu import calc_code_bleu


class Tester:
    def __init__(self, model, model_class, tokenizer, out_path, beam_size, train_fraction):
        self.model = model
        self.model_class = model_class
        self.tokenizer = tokenizer
        self.out_path = Path(out_path)
        self.beam_size = beam_size
        self.train_fraction = train_fraction
        self.checkpoint_dir = self.out_path / self.model / str(self.train_fraction) / "checkpoint-best"
        self.split_dir = self.out_path / self.model / str(self.train_fraction) / "splits"

    def validate(self):
        df = pd.read_json(self.split_dir / "test.json")

        # Load best model checkpoint
        accelerator = Accelerator()
        model = self.model_class.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        model = accelerator.prepare(model)
        model.eval()

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        # Safely get decoder_start_token_id
        if hasattr(model.config, "decoder_start_token_id") and model.config.decoder_start_token_id is not None:
            decoder_sid = model.config.decoder_start_token_id
        else:
            try:
                decoder_sid = self.tokenizer.convert_tokens_to_ids("__python__")
            except:
                decoder_sid = self.tokenizer.pad_token_id  # fallback
        print("decoder_start_token_id:", decoder_sid)
        print("vocab_size:", model.config.vocab_size)
        assert decoder_sid < model.config.vocab_size

        predictions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
            input_ids = self.tokenizer(row["input"], return_tensors="pt", truncation=True, padding=True).input_ids.to(
                accelerator.device)

            outputs = model.generate(
                input_ids=input_ids,
                max_length=self.tokenizer.model_max_length,
                num_beams=int(self.beam_size),
                num_return_sequences=int(self.beam_size),
                early_stopping=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                decoder_start_token_id=decoder_sid
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.append({
                "ID": idx,
                "target": row["output"],
                "preds": [pred.strip() for pred in decoded]
            })

        pred_df = pd.DataFrame(predictions)
        pred_df.to_json(self.out_path / self.model / str(self.train_fraction) / "test_predictions.json", indent=2)

        bleu_score, code_bleu_score, em = self.compute_scores(pred_df)
        print(f"BLEU: {bleu_score} | CodeBLEU: {code_bleu_score} | EM: {em}")

    def compute_scores(self, pred_df):
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
