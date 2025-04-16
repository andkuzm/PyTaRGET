import difflib
import re
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class HunkPrioritizer:
    def __init__(self, tokenizer, ds):
        self.tokenizer = tokenizer
        self.ds = ds

    def extract_hunks_from_code(self, code_str):
        # Extract all [<HUNK>]...</HUNK>] blocks
        hunk_blocks = re.findall(r"\[<HUNK>](.*?)\[</HUNK>]", code_str, re.DOTALL)
        return [{"annotated_doc": "[<HUNK>]" + h + "[</HUNK>]"} for h in hunk_blocks]

    def prioritize_hunks_prep(self):
        """
        Extracts and prioritizes all hunks from the dataset row-by-row,
        using test breakage and test source context from each row.
        Rows with invalid or empty TF-IDF inputs are skipped.
        """
        self.ds["prioritized_changes"] = pd.Series([None] * len(self.ds), dtype=object)
        valid_indices = []

        for idx, row in tqdm(self.ds.iterrows(), total=len(self.ds), desc="Filtering and prioritizing hunks"):
            code_str = row["annotated_code"]
            hunks = self.extract_hunks_from_code(code_str)
            test_breakage_code = self.extract_test_breakage(code_str)
            test_source_code = self.extract_testcontext(code_str)

            if not test_source_code.strip():
                continue

            # Step 1: Create change documents and filter empty ones
            change_docs = []
            for hunk in hunks:
                doc = self.create_changed_document(hunk)
                if len(doc["annotated_doc_seq"]) > 0:
                    change_docs.append(doc)

            if not change_docs:
                continue

            try:
                _ = self.get_tfidf_sim(test_breakage_code, change_docs)
            except Exception:
                continue

            prioritized = self.prioritize_hunks(hunks, test_breakage_code, test_source_code)
            self.ds.at[idx, "prioritized_changes"] = prioritized
            valid_indices.append(idx)

        self.ds = self.ds.loc[valid_indices].reset_index(drop=True)
        print(f"Remaining rows after filtering: {len(self.ds)}")

        return self.ds

    def prioritize_hunks(self, hunks, test_breakage_code, test_source_code):
        """
        Prioritizes hunks based on:
        - TF-IDF similarity to test breakage
        - TF-IDF similarity to covered test source
        - Repetition count (line-level)
        """
        change_docs = []
        line_repeat = defaultdict(int)  # Track repetition at the line level

        # Step 1: Process each hunk
        for hunk in hunks:
            change_doc = self.create_changed_document(hunk)
            change_docs.append(change_doc)

            # Extract changed lines from the hunk
            del_lines, add_lines = self.extract_hunk_lines(hunk)
            diff_lines = self.extract_diff_lines(del_lines, add_lines)
            for line in diff_lines:
                line_repeat[line] += 1

        # Step 2: Assign repetition scores
        for change_doc in change_docs:
            del_lines, add_lines = self.extract_hunk_lines({"annotated_doc": change_doc["annotated_doc"]})
            diff_lines = self.extract_diff_lines(del_lines, add_lines)
            change_doc["repeat"] = sum(line_repeat[line] for line in diff_lines if line in line_repeat)

        # Step 3: Remove duplicates
        change_docs = self.remove_duplicate_change_documents(change_docs)

        # Step 4: Compute TF-IDF similarity
        tfidf_breakage = self.get_tfidf_sim(test_breakage_code, change_docs)
        tfidf_testsrc = self.get_tfidf_sim(test_source_code, change_docs)

        # Step 5: Assign scores to hunks
        for i, changed_doc in enumerate(change_docs):
            changed_doc["tfidf_breakage"] = tfidf_breakage[i]
            changed_doc["tfidf_testsrc"] = tfidf_testsrc[i]

        # Step 6: Sort hunks by priority
        prioritized_hunks = sorted(
            change_docs,
            key=lambda doc: (-round(doc["tfidf_breakage"], 1),
                             -doc["repeat"],
                             -round(doc["tfidf_testsrc"], 2),
                             doc["annotated_doc"])  # Tie-breaker: alphabetical order
        )

        return prioritized_hunks

    def create_changed_document(self, hunk):
        """Creates an annotated document for a hunk."""
        if "annotated_doc" not in hunk:
            hunk["annotated_doc"] = self.create_hunk_document(hunk)
        if "annotated_doc_seq" not in hunk:
            hunk["annotated_doc_seq"] = self.tokenizer.encode(hunk["annotated_doc"])
        return {"annotated_doc": hunk["annotated_doc"], "annotated_doc_seq": hunk["annotated_doc_seq"]}

    def create_hunk_document(self, hunk):
        """Converts hunk to a structured format."""
        src_lines, tgt_lines = self.extract_hunk_lines(hunk)
        src_annotated_doc = "DEL " + " ".join(src_lines) if src_lines else ""
        tgt_annotated_doc = "ADD " + " ".join(tgt_lines) if tgt_lines else ""
        return f"HUNK {src_annotated_doc} {tgt_annotated_doc} HUNK_END"

    def extract_hunk_lines(self, hunk):
        del_match = re.search(r"\[<DEL>](.*?)\[</DEL>]", hunk["annotated_doc"], re.DOTALL)
        add_match = re.search(r"\[<ADD>](.*?)\[</ADD>]", hunk["annotated_doc"], re.DOTALL)

        del_lines = del_match.group(1).rstrip().split("\n") if del_match else []
        add_lines = add_match.group(1).rstrip().split("\n") if add_match else []

        if not del_lines and not add_lines:
            # No DEL/ADD, fallback to body content
            # Extract everything between [<HUNK>] and [</HUNK>]
            raw = re.search(r"\[<HUNK>].*?\n(.*?)\[</HUNK>]", hunk["annotated_doc"], re.DOTALL)
            context_lines = raw.group(1).rstrip().split("\n") if raw else []
            add_lines = context_lines

        return del_lines, add_lines

    def extract_diff_lines(self, del_lines, add_lines):
        """Extracts only changed lines (+/-) from a unified diff string."""
        diff_lines = list(difflib.unified_diff(del_lines, add_lines, lineterm=""))
        return [line for line in diff_lines if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]

    def remove_duplicate_change_documents(self, change_docs):
        """Removes duplicate hunks based on annotated_doc content."""
        seen = set()
        unique_docs = []
        for doc in change_docs:
            if doc["annotated_doc"] not in seen:
                unique_docs.append(doc)
                seen.add(doc["annotated_doc"])
        return unique_docs

    def get_tfidf_sim(self, target, changes):
        vectorizer = TfidfVectorizer(tokenizer=lambda t: t, lowercase=False, token_pattern=None)
        tokenized_docs = [
            tokens for tokens in [self.tokenizer.encode(target)] + [c.get("annotated_doc_seq", []) for c in changes]
            if isinstance(tokens, list) and len(tokens) > 0
        ]
        vectors = vectorizer.fit_transform(tokenized_docs)
        dense = vectors.todense()
        cosine_sim = (dense * dense[0].T).T.tolist()[0]
        return [cosine_sim[i + 1] for i in range(len(changes))]

    def extract_testcontext(self, code_str):
        """Extracts the full test context block from annotated_code."""
        match = re.search(r"\[<TESTCONTEXT>](.*?)\[</TESTCONTEXT>]", code_str, re.DOTALL)
        return match.group(1).rstrip() if match else ""

    def extract_test_breakage(self, code_str):
        """Extracts the failing test lines from the BREAKAGE block in annotated_code."""
        match = re.search(r"\[<BREAKAGE>](.*?)\[</BREAKAGE>]", code_str, re.DOTALL)
        return match.group(1).rstrip() if match else ""
