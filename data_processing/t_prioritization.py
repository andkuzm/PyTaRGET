from data_processing.prioritizer import HunkPrioritizer


# Dummy tokenizer for demonstration purposes.
class DummyTokenizer:
    def encode(self, text):
        # Simple whitespace tokenization.
        return text.split()


if __name__ == "__main__":
    # Create a dummy tokenizer instance.
    tokenizer = DummyTokenizer()

    # Instantiate the HunkPrioritizer.
    prioritizer = HunkPrioritizer(tokenizer, None)

    # Define three sample hunks:
    # Hunk 1: A change for function f1 with a specific DEL/ADD block.
    sample_hunk1 = {
        "annotated_doc": "\n".join([
            "[<DEL>]",
            "def f1(x):",
            "    result = f2(x)",
            "[</DEL>]",
            "[<ADD>]",
            "def f1(a):",
            "    with pysnooper.snoop():",
            "        result = f2(a)",
            "[</ADD>]",
            "    return result"
        ])
    }

    # Hunk 2: Similar to hunk 1, with extra context (a comment) but identical DEL/ADD blocks.
    sample_hunk2 = {
        "annotated_doc": "\n".join([
            "# Extra context comment for f1",
            "[<DEL>]",
            "def f1(x):",
            "    result = f2(x)",
            "[</DEL>]",
            "[<ADD>]",
            "def f1(a):",
            "    with pysnooper.snoop():",
            "        result = f2(a)",
            "[</ADD>]",
            "    return result"
        ])
    }

    # Hunk 3: A different change for function f1 (different internal call).
    sample_hunk3 = {
        "annotated_doc": "\n".join([
            "def f1(a):",
            "    with pysnooper.snoop():",
            "        value = f3(a)",
            "    return value"
        ])
    }

    # For demonstration, we use a list with all three sample hunks.
    hunks = [sample_hunk1, sample_hunk2, sample_hunk3]

    # Define sample test breakage code and test source code for TF-IDF similarity.
    test_breakage_code = "def f1(x): result = f2(x)"
    test_source_code = "def f1(a): with pysnooper.snoop(): result = f2(a)"

    # Process and prioritize hunks.
    prioritized = prioritizer.prioritize_hunks(hunks, test_breakage_code, test_source_code)

    # Output the prioritized hunks.
    for i, doc in enumerate(prioritized):
        print(f"Priority {i + 1}:")
        print("Annotated Doc:")
        print(doc["annotated_doc"])
        print("Aggregate Repeat:", doc["repeat"])
        print("TF-IDF Breakage:", doc["tfidf_breakage"])
        print("TF-IDF TestSrc:", doc["tfidf_testsrc"])
        print("Tokenized:", doc["annotated_doc_seq"])
        print("-" * 40)
