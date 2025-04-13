import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.optim._multi_tensor import AdamW
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from transformers.models.bart.modeling_bart import shift_tokens_right


class Trainer:

    def __init__(self, model, model_path, tokenizer, out_path, model_class, train_fraction, seed=1234, grad_accum_steps=1):
        self.model = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.out_path = Path(out_path)
        self.model_class = model_class
        self.train_fraction = train_fraction
        self.seed = seed
        self.grad_accum_steps = grad_accum_steps
        self.set_seed()

    def set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train(self):
        # Load datasets
        train_dataset = pickle.load(open(self.out_path / self.model / str(self.train_fraction) / "train.pkl", "rb"))
        eval_dataset = pickle.load(open(self.out_path / self.model / str(self.train_fraction) / "valid.pkl", "rb"))

        # Model loading
        model_dir = self.out_path / self.model / str(self.train_fraction) / "model"

        model = self.model_class.from_pretrained(self.model_path, trust_remote_code=True)
        model.resize_token_embeddings(len(self.tokenizer))
        model.save_pretrained(model_dir)

        # Optimizer and scheduler
        train_steps = len(train_dataset) * 10 // (8 * self.grad_accum_steps)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, train_steps)

        # Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=self.grad_accum_steps)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        if self.model=="plbart":
            collate_fn = make_collate_fnplb(self.tokenizer, model)
        else:
            collate_fn = make_collate_fn(self.tokenizer)

        # Training params
        batch_size = 1
        eval_batch_size = 1
        epochs = 4
        best_loss = float("inf")
        best_epoch = 0
        early_stop = 1
        stats = {"epochs": [], "train_set_size": len(train_dataset), "valid_set_size": len(eval_dataset)}

        id_check = self.tokenizer.convert_tokens_to_ids("__python__")
        vocab_size = model.config.vocab_size
        print(f"__python__ id: {id_check}, Model vocab size: {vocab_size}")
        assert id_check < vocab_size, "Token ID out of range after resizing!"
        print("Training started.")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            for step, batch in enumerate(train_loader, 1):
                with accelerator.accumulate(model):
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    print("processing batch", step)
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / step
            print(f"Epoch {epoch} | Training loss: {avg_loss:.4f}")
            valid_loss = self.validate(model, eval_dataset, accelerator, eval_batch_size, collate_fn)
            print(f"Epoch {epoch} | Validation loss: {valid_loss:.4f}")

            # Save stats
            stats["epochs"].append({
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "valid_loss": round(valid_loss, 4),
                "timestamp": datetime.now().isoformat()
            })

            # Checkpoint
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                model.save_pretrained(self.out_path / self.model / str(self.train_fraction) / "checkpoint-best")
                print(f"Best checkpoint updated: epoch {epoch}, loss {valid_loss:.4f}")

            if (epoch - best_epoch) >= early_stop:
                print(f"Early stopping at epoch {epoch}, no improvement for {early_stop} epochs.")
                break

        # Final stats
        with open(self.out_path / self.model / str(self.train_fraction) / "training_stats.json", "w") as f:
            import json
            json.dump(stats, f, indent=2)
        print("Training completed.")

    def validate(self, model, eval_dataset, accelerator, batch_size, collate_fn):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
            for step, batch in enumerate(eval_loader, 1):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                print("validating batch", step)
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += accelerator.gather_for_metrics(loss).sum().item()
        return total_loss / step

def make_collate_fnplb(tokenizer, model):
    def collate_fn(batch):
        input_ids = [item["input_ids"].squeeze(0) for item in batch]
        attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
        labels = [item["labels"].squeeze(0) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Truncate to max positional embedding length
        max_len = model.config.max_position_embeddings  # usually 1024
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.convert_tokens_to_ids("__python__")
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }
    return collate_fn

def make_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [item['input_ids'].squeeze(0) for item in batch]
        attention_masks = [item['attention_mask'].squeeze(0) for item in batch]
        labels = [item['labels'].squeeze(0) for item in batch]

        return {
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            'attention_mask': pad_sequence(attention_masks, batch_first=True, padding_value=0),
            'labels': pad_sequence(labels, batch_first=True, padding_value=-100)
        }
    return collate_fn
