import sys

import pandas as pd
from transformers import (
    PLBartTokenizer,
    AutoTokenizer
)
import inspect

class Tokens:
    BREAKAGE_START = "[<BREAKAGE>]"
    BREAKAGE_END = "[</BREAKAGE>]"
    TEST_CONTEXT_START = "[<TESTCONTEXT>]"
    TEST_CONTEXT_END = "[</TESTCONTEXT>]"
    REPAIR_CONTEXT_START = "[<REPAIRCONTEXT>]"
    REPAIR_CONTEXT_END = "[</REPAIRCONTEXT>]"
    DELETE_START = "[<DEL>]"
    DELETE_END = "[</DEL>]"
    ADD_START = "[<ADD>]"
    ADD_END = "[</ADD>]"
    HUNK = "[<HUNK>]"
    HUNK_END = "[</HUNK>]"

    # if args.model == "plbart":
    #     args.model_class = PLBartForConditionalGeneration
    #     args.model_tokenizer_class = PLBartTokenizer
    #     args.dataset_class = PLBARTDataset
    # elif args.model == "codegen":
    #     args.model_class = CodeGenForCausalLM
    #     args.model_tokenizer_class = AutoTokenizer
    #     args.dataset_class = CodeGenDataset
    # elif args.model == "codet5p":
    #     args.model_class = AutoModelForSeq2SeqLM
    #     args.model_tokenizer_class = AutoTokenizer
    #     args.dataset_class = EncDecDataset

class Encoder:
    def __init__(self, annotated_cases_path, out_path, model, train_size, tokenizer=None):
        self.annotated_cases_path = annotated_cases_path
        self.out_path = out_path
        self.model = model
        self.train_size = train_size
        if model == "plbart":
            self.tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
        if model == "codegen":
            self.tokenizer = AutoTokenizer.from_pretrained("salesforce/codegen-350M-multi")
        if model == "codet5p":
            self.tokenizer = AutoTokenizer.from_pretrained("salesforce/codet5p-770m")

    def encode(self):
        ds = self.preprocess_dataset()
        if len(ds) <= 1:
            print("No data found")
            sys.exit()
        ds = self.create_inputs_and_outputs(ds)
        train_ds, valid_ds, test_ds = self.split(ds)

    def create_tokenizer(self):
        new_special_tokens = {
            "additional_special_tokens": self.tokenizer.additional_special_tokens
            + [v for k, v in inspect.getmembers(Tokens) if not k.startswith("_")]
        }
        self.tokenizer.add_special_tokens(new_special_tokens)
        self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        self.tokenizer.save_pretrained(str(self.out_path / "tokenizer"))

    def split(self, ds):
        # Ensure dataset is sorted by commit order (older commits first)
        ds = ds.sort_values("broken_hash").reset_index(drop=True)

        # Compute absolute number of training samples based on fraction
        num_train_samples = int(len(ds) * self.train_size)

        # Identify commits until we reach required train size
        unique_commits = ds["broken_hash"].unique()
        train_commits = []
        current_size = 0

        for commit in unique_commits:
            commit_size = len(ds[ds["broken_hash"] == commit])
            if current_size + commit_size > num_train_samples:
                break  # Stop once we have enough samples
            train_commits.append(commit)
            current_size += commit_size

        # Assign entries based on commit inclusion
        train_ds = ds[ds["broken_hash"].isin(train_commits)]
        eval_ds = ds[~ds["broken_hash"].isin(train_commits)]

        # Stratified split for validation & test sets
        grouped = eval_ds.groupby("broken_hash")
        valid_ds_list, test_ds_list = [], []

        for _, group in grouped:
            split_idx = int(0.25 * len(group))  # 25% to validation, 75% to test
            valid_ds_list.append(group.iloc[:split_idx])
            test_ds_list.append(group.iloc[split_idx:])

        valid_ds = pd.concat(valid_ds_list).reset_index(drop=True)
        test_ds = pd.concat(test_ds_list).reset_index(drop=True)

        return train_ds, valid_ds, test_ds



    def preprocess_dataset(self):
        ds = pd.read_csv(self.annotated_cases_path)

        return ds

    def create_inputs_and_outputs(self, ds):
        return ds