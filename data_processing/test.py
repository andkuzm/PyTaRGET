import json
import re

import pandas as pd
from pathlib import Path
from accelerate import Accelerator
from datetime import datetime
from tqdm import tqdm

from data_processing import prediction_files
from data_processing.atomic_io import sigterm_as_keyboard_interrupt
from data_processing.metrics import exact_match_flags
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

    def validate(self, save_every=10, resume=True, limit=None):
        """
        Generate the beam candidates of every case of splits/test.json and write test_predictions.json.
        The predictions are saved every `save_every` cases (atomically) and when the run is interrupted, and a run
        that finds a predictions file continues with the cases that have no prediction yet (resume=False starts over).
        limit: only the first `limit` cases (a smoke test); the scores are then over that many cases.
        """
        with sigterm_as_keyboard_interrupt():
            return self._validate(save_every, resume, limit)

    def _validate(self, save_every, resume, limit):
        with open(self.split_dir / "test.json", encoding="utf-8") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
        if limit is not None:
            df = df.head(limit)
        predictions_file = self.out_path / self.model / str(self.train_fraction) / "test_predictions.json"

        predictions = []
        if resume and predictions_file.exists():
            predictions = prediction_files.load(predictions_file)
            prediction_files.check_matches_test_split(predictions, rows)
            predictions = [p for p in predictions if p["ID"] < len(df)]
            print(f"Resuming from {predictions_file}: {len(predictions)} of {len(df)} cases already have predictions")
        done = {p["ID"] for p in predictions}

        # Load best model checkpoint
        accelerator = Accelerator()
        model = self.model_class.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        model = accelerator.prepare(model)
        model.eval()

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        generation_args = dict(
            max_length=self.tokenizer.model_max_length,
            num_beams=int(self.beam_size),
            num_return_sequences=int(self.beam_size),
            early_stopping=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        if model.config.is_encoder_decoder:
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
            generation_args["decoder_start_token_id"] = decoder_sid

        unsaved = 0
        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
                if int(idx) in done:
                    continue
                input_ids = self.tokenizer(row["input"], return_tensors="pt", truncation=True, padding=True).input_ids.to(
                    accelerator.device)

                outputs = model.generate(input_ids=input_ids, **generation_args)
                outputs = self.new_tokens(model, outputs, input_ids)
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.append({
                    "ID": int(idx),
                    "target": row["output"],
                    "preds": [pred.replace("<TAB>", "    ").replace("<NL>", "\n") for pred in decoded]
                })
                unsaved += 1
                if unsaved >= save_every:
                    prediction_files.save(predictions, predictions_file)
                    unsaved = 0
        finally:
            if unsaved:                       # also when interrupted: keep what was generated
                prediction_files.save(predictions, predictions_file)

        prediction_files.save(predictions, predictions_file)
        bleu_score, code_bleu_score, em, norm_em = self.compute_scores(pd.DataFrame(predictions))
        text, warning = prediction_files.describe(len(predictions), len(df))
        print(f"BLEU: {bleu_score} | CodeBLEU: {code_bleu_score} | EM: {em} | EM (indentation-normalized): {norm_em} ({text})")
        if warning:
            print(warning)
        return bleu_score, code_bleu_score, em, norm_em

    @staticmethod
    def new_tokens(model, outputs, input_ids):
        """Decoder-only models (CodeGen) return prompt + continuation; keep only the continuation."""
        if model.config.is_encoder_decoder:
            return outputs
        return outputs[:, input_ids.shape[1]:]

    def compute_scores(self, pred_df):
        pred_df = pd.DataFrame(pred_df)
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
        bleu_score = corpus_bleu([[t.split()] for t in targets], [p.split() for p in best_preds])
        code_bleu_score = calc_code_bleu([targets], best_preds, lang="python")
        return round(bleu_score * 100, 2), round(code_bleu_score * 100, 2), em, norm_em
