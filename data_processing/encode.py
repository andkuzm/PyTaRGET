import pickle
import re
import sys

import pandas as pd

from data_processing.prioritizer import HunkPrioritizer


class Encoder:
    def __init__(self, annotated_cases_path, out_path, train_size, tokenizer, dataset_class, model, Tokens):
        self.annotated_cases_path = annotated_cases_path
        self.out_path = out_path
        self.train_size = train_size
        self.tokenizer = tokenizer
        self.dataset_class = dataset_class
        self.model = model
        self.Tokens = Tokens

    def encode(self):
        ds = self.preprocess_dataset()
        if len(ds) <= 1:
            print("No data found")
            sys.exit()
        prioritizer = HunkPrioritizer(self.tokenizer, ds)
        ds = prioritizer.prioritize_hunks_prep()
        ds = ds[ds["prioritized_changes"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        print("dropped rows with empty coverage, remaining rows:", len(ds))
        ds = self.create_inputs_and_outputs(ds)
        train_ds, valid_ds, test_ds = self.split(ds)

        print("Creating datasets")
        train_file = self.out_path / self.model / str(self.train_size) / "train.pkl"
        valid_file = self.out_path / self.model / str(self.train_size) / "valid.pkl"
        test_file = self.out_path / self.model / str(self.train_size) / "test.pkl"
        og_ds_s = len(train_ds) + len(valid_ds) + len(test_ds)
        train_ds = self.dataset_class(train_ds, self.tokenizer, "train", self.out_path / self.model / str(self.train_size))
        valid_ds = self.dataset_class(valid_ds, self.tokenizer, "valid", self.out_path / self.model / str(self.train_size))
        test_ds = self.dataset_class(test_ds, self.tokenizer, "test", self.out_path / self.model / str(self.train_size))
        new_ds_s = len(train_ds) + len(valid_ds) + len(test_ds)
        valid_per = round(100 * new_ds_s / og_ds_s, 1)
        print(
            f"{valid_per} % ({new_ds_s}/{og_ds_s}) samples had less than max_length ({self.tokenizer.model_max_length}) tokens."
        )
        print("Pickling datasets")
        pickle.dump(train_ds, open(str(train_file), "wb"))
        pickle.dump(valid_ds, open(str(valid_file), "wb"))
        pickle.dump(test_ds, open(str(test_file), "wb"))

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
        # Read the dataset
        ds = pd.read_csv(self.annotated_cases_path, delimiter='|')

        # Remove rows with empty values
        ds = ds.dropna()
        print("empty values dropped, remaining dataset length:", len(ds))

        # Remove rows with invalid characters (this assumes that all rows should be valid UTF-8)
        ds = ds[ds.apply(lambda row: row.astype(str).apply(lambda x: x.isascii()).all(), axis=1)]
        print("incorrectly formatted rows dropped, remaining dataset length:", len(ds))

        pattern = r"(?s)\[<TESTCONTEXT>\](.*?)\[</TESTCONTEXT>\]"
        ds = ds[ds['annotated_code'].str.contains(pattern, regex=True, na=False)]
        print("empty context containing rows dropped, remaining dataset length:", len(ds))
        # Check token length constraints for testcontext
        max_token_length = self.tokenizer.model_max_length
        ds = ds[ds.apply(self.check_testcontext_length, axis=1, max_token_length=max_token_length)]
        print("rows with oversized tests dropped, remaining dataset length:", len(ds))

        return ds

    def check_testcontext_length(self, row, max_token_length):
        # Extract the testcontext
        testcontext = self.extract_testcontext(row["annotated_code"])
        # Tokenize and check if testcontext length exceeds half of the max token length
        testcontext_tokens = self.tokenizer.encode(testcontext.replace("\t", "<TAB>").replace("    ", "<TAB>").replace("\n", "<NL>"))
        if len(testcontext_tokens) > (max_token_length // 2):
            return False  # Remove this row
        return True

    def extract_testcontext(self, annotated_code):
        # Regex to find the content between [<TESTCONTEXT>] and [</TESTCONTEXT>]
        testcontext_pattern = r"\[<TESTCONTEXT>\](.*?)\[</TESTCONTEXT>\]"

        # Find the content
        match = re.search(testcontext_pattern, annotated_code, re.DOTALL)

        if match:
            return match.group(1).rstrip()  # Return the content inside the TESTCONTEXT
        else:
            return None  # If no TESTCONTEXT is found

    def create_inputs_and_outputs(self, ds):

        # Step 1: Select change hunks for each row
        ds_selected_changes = [self.select_changes(row) for _, row in ds.iterrows()]

        # Step 2: Log inclusion stats
        all_change_cnt = sum(len(row["prioritized_changes"]) for _, row in ds.iterrows())
        included_change_cnt = sum(len(selected[1]) for selected in ds_selected_changes)
        included_change_p = round(100 * included_change_cnt / all_change_cnt, 1)
        print(f"In total, {included_change_p} % of covered changed documents are included in the input.")

        # Step 3: Assign inputs and outputs
        ds["input"] = [selected[0] for selected in ds_selected_changes]
        ds["output"] = ds.apply(lambda row: self.create_output(row), axis=1)

        # Step 4: Cleanup
        ds["prioritized_changes"].apply(
            lambda changes: [c.pop("annotated_doc_seq") for c in changes if "annotated_doc_seq" in c])
        ds["input"] = ds["input"].str.rstrip()
        ds["output"] = ds["output"].str.rstrip()
        return ds

    def create_output(self, row):
        repaired_code = self.get_repaired_code(row)
        if not repaired_code:
            repaired_code = "# Deleted"  # Python-style fallback
        return repaired_code

    def get_repaired_code(self, row):
        code_str = row["annotated_code"]
        pattern = r"\[<REPAIREDTEST>\](.*?)\[</REPAIREDTEST>\]"
        match = re.search(pattern, code_str, re.DOTALL)
        if match:
            return match.group(1).rstrip()
        return ""

    def select_changes(self, row):
        pr_changes_cnt = len(row["prioritized_changes"])
        selected_changes = []
        test_context = self.create_test_context(row)
        test_context_e = self.tokenizer.encode(test_context.replace("\t", "<TAB>").replace("    ", "<TAB>").replace("\n", "<NL>"))
        for i in range(pr_changes_cnt):
            row["prioritized_changes"][i]["selected"] = False
            new_selected_changes = selected_changes + [row["prioritized_changes"][i]]
            # The +2 is for Tokens.TEST_CONTEXT and Tokens.REPAIR_CONTEXT
            new_input_len = len(test_context_e) + sum(len(c["annotated_doc_seq"]) for c in new_selected_changes) + 2
            max_input_length = self.tokenizer.model_max_length
            if new_input_len <= max_input_length:
                selected_changes = new_selected_changes
                row["prioritized_changes"][i]["selected"] = True

        if len(selected_changes) == 0:
            selected_changes = [row["prioritized_changes"][0]]
        return (self.create_input(test_context, selected_changes), selected_changes)

    def create_input(self, test_context, selected_changes):
        return "".join(
            [self.Tokens.TEST_CONTEXT, test_context, self.Tokens.REPAIR_CONTEXT] +
            [change["annotated_doc"] for change in selected_changes]
        )

    def create_test_context(self, row):
        code_str = row["annotated_code"]
        test_code = self.extract_testcontext(code_str)

        # Inject breakage tokens directly
        test_code = test_code.replace("[<BREAKAGE>]", self.Tokens.BREAKAGE_START)
        test_code = test_code.replace("[</BREAKAGE>]", self.Tokens.BREAKAGE_END)

        # Clean and process each line
        lines = test_code.split("\n")
        processed_lines = [
            line.rstrip()
            for line in lines
            if not line.strip().startswith("#") and line.strip()
        ]

        test_context = "\n".join(processed_lines)
        return test_context