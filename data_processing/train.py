import json
import os
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from transformers.models.bart.modeling_bart import shift_tokens_right

from data_processing.atomic_io import (atomic_directory, atomic_save_pretrained, atomic_write_json,
                                       recover_directory, sigterm_as_keyboard_interrupt)

CHECKPOINT_LAST = "checkpoint-last"
CHECKPOINT_BEST = "checkpoint-best"


class Trainer:
    """
    Fine-tunes a seq2seq / causal model on <out_path>/<model>/<train_fraction>/{train,valid}.pkl.

    Interruptions (Slurm time limit, node failure, Ctrl-C) cost at most the epoch in progress:
      - after every epoch the full training state (weights, optimizer, scheduler, random state, statistics) is
        saved to `checkpoint-last` with an atomic directory swap, and `training_stats.json` is rewritten;
      - `train()` called again continues from there and, from the same seed, ends with exactly the weights an
        uninterrupted run would have;
      - `checkpoint-best` is only replaced when the validation loss improves on the best loss of the whole run,
        never by a restart. A fresh run (resume=False, or no saved state) keeps an existing `checkpoint-best`
        as `checkpoint-best.previous` instead of overwriting it.
    """

    def __init__(self, model, model_path, tokenizer, out_path, model_class, train_fraction, seed=1234,
                 grad_accum_steps=1, epochs=10, batch_size=8, eval_batch_size=8, early_stop=3, learning_rate=5e-5):
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.experiment_dir = self.out_path / self.model / str(self.train_fraction)
        self.set_seed()

    def set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    # ---- saved state -------------------------------------------------------------------------------------------
    @staticmethod
    def random_state():
        return {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(), "python": random.getstate()}

    @staticmethod
    def restore_random_state(state):
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])

    def save_last(self, accelerator, model, optimizer, scheduler, state):
        """The complete training state after an epoch, swapped in atomically."""
        if not accelerator.is_main_process:
            return
        with atomic_directory(self.experiment_dir / CHECKPOINT_LAST) as directory:
            accelerator.unwrap_model(model).save_pretrained(directory)
            torch.save(optimizer.state_dict(), directory / "optimizer.pt")
            torch.save(scheduler.state_dict(), directory / "scheduler.pt")
            torch.save(self.random_state(), directory / "rng_state.pt")
            (directory / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

    def load_last(self):
        """The saved training state, or None when there is none (or it is incomplete)."""
        directory = self.experiment_dir / CHECKPOINT_LAST
        state_file = directory / "trainer_state.json"
        needed = ["optimizer.pt", "scheduler.pt", "rng_state.pt", "trainer_state.json"]
        if not all((directory / name).exists() for name in needed):
            return None
        return json.loads(state_file.read_text(encoding="utf-8"))

    def keep_previous_best(self):
        """A fresh run must not destroy the best checkpoint of an earlier one."""
        best = self.experiment_dir / CHECKPOINT_BEST
        if best.exists():
            previous = self.experiment_dir / (CHECKPOINT_BEST + ".previous")
            shutil.rmtree(previous, ignore_errors=True)
            os.replace(best, previous)
            print(f"Kept the existing {CHECKPOINT_BEST} as {previous.name}")

    def write_stats(self, stats):
        atomic_write_json(self.experiment_dir / "training_stats.json", stats, indent=2)

    def should_stop(self, epoch, best_epoch):
        return (epoch - best_epoch) >= self.early_stop

    # ---- training ----------------------------------------------------------------------------------------------
    def train(self, resume=True):
        """Train (or continue training) the model. resume=False starts over."""
        with sigterm_as_keyboard_interrupt():
            self._train(resume)

    def _train(self, resume):
        directory = self.experiment_dir
        for name in ("model", CHECKPOINT_BEST, CHECKPOINT_LAST):
            recover_directory(directory / name)

        # Load datasets
        train_dataset = pickle.load(open(directory / "train.pkl", "rb"))
        eval_dataset = pickle.load(open(directory / "valid.pkl", "rb"))
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            raise ValueError(f"Cannot train: {len(train_dataset)} training and {len(eval_dataset)} validation samples in "
                             f"{directory}. Training needs both (encode with a train fraction between 0 and 1 and at "
                             f"least two held-out commits).")

        stats_file = directory / "training_stats.json"
        if resume and stats_file.exists() and json.loads(stats_file.read_text(encoding="utf-8")).get("completed"):
            print("Training already completed, nothing to do (pass resume=False to train again).")
            return
        saved = self.load_last() if resume else None
        if saved is not None:
            stats = saved["stats"]
            if stats["train_set_size"] != len(train_dataset) or stats["valid_set_size"] != len(eval_dataset):
                raise ValueError(
                    f"{CHECKPOINT_LAST} was saved for {stats['train_set_size']} train / {stats['valid_set_size']} valid "
                    f"samples but the datasets now have {len(train_dataset)} / {len(eval_dataset)}: the data was "
                    f"encoded again. Delete {directory / CHECKPOINT_LAST} or train with resume=False.")
            model = self.model_class.from_pretrained(directory / CHECKPOINT_LAST, trust_remote_code=True)
            start_epoch, best_loss, best_epoch = saved["epoch"] + 1, saved["best_loss"], saved["best_epoch"]
            print(f"Resuming after epoch {saved['epoch']} (best so far: epoch {best_epoch}, loss {best_loss:.4f})")
        else:
            self.keep_previous_best()
            model = self.model_class.from_pretrained(self.model_path, trust_remote_code=True)
            model.resize_token_embeddings(len(self.tokenizer))
            atomic_save_pretrained(model, directory / "model")
            start_epoch, best_loss, best_epoch = 1, float("inf"), 0
            stats = {"epochs": [], "train_set_size": len(train_dataset), "valid_set_size": len(eval_dataset),
                     "completed": False}
            self.write_stats(stats)     # a stale "completed" of an earlier run must not survive a fresh start

        # Optimizer and scheduler
        train_steps = len(train_dataset) * self.epochs // (self.batch_size * self.grad_accum_steps)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, train_steps)

        # Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=self.grad_accum_steps)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        if saved is not None:
            optimizer.load_state_dict(torch.load(directory / CHECKPOINT_LAST / "optimizer.pt", map_location="cpu"))
            scheduler.load_state_dict(torch.load(directory / CHECKPOINT_LAST / "scheduler.pt", map_location="cpu"))
            self.restore_random_state(torch.load(directory / CHECKPOINT_LAST / "rng_state.pt", weights_only=False))
        if self.model == "plbart":
            collate_fn = make_collate_fnplb(self.tokenizer, model)
        else:
            collate_fn = make_collate_fn(self.tokenizer)

        id_check = self.tokenizer.convert_tokens_to_ids("__python__")
        vocab_size = model.config.vocab_size
        print(f"__python__ id: {id_check}, Model vocab size: {vocab_size}")
        assert id_check < vocab_size, "Token ID out of range after resizing!"
        print("Training started.")
        stopped_early = saved is not None and self.should_stop(saved["epoch"], best_epoch)
        for epoch in range(start_epoch, self.epochs + 1):
            if stopped_early:
                break
            model.train()
            total_loss = 0
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            for step, batch in enumerate(train_loader, 1):
                with accelerator.accumulate(model):
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / step
            print(f"Epoch {epoch} | Training loss: {avg_loss:.4f}")
            valid_loss = self.validate(model, eval_dataset, accelerator, self.eval_batch_size, collate_fn)
            print(f"Epoch {epoch} | Validation loss: {valid_loss:.4f}")

            stats["epochs"].append({
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "valid_loss": round(valid_loss, 4),
                "timestamp": datetime.now().isoformat()
            })

            # Checkpoint: the best weights, then the full state to continue from, then the statistics
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                if accelerator.is_main_process:
                    atomic_save_pretrained(accelerator.unwrap_model(model), directory / CHECKPOINT_BEST)
                print(f"Best checkpoint updated: epoch {epoch}, loss {valid_loss:.4f}")
            self.save_last(accelerator, model, optimizer, scheduler,
                           {"epoch": epoch, "best_loss": best_loss, "best_epoch": best_epoch, "stats": stats,
                            "completed": False})
            self.write_stats(stats)

            if self.should_stop(epoch, best_epoch):
                print(f"Early stopping at epoch {epoch}, no improvement for {self.early_stop} epochs.")
                stopped_early = True
                break

        # Final stats
        stats["completed"] = True
        stats["stopped_early"] = stopped_early
        self.write_stats(stats)
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
