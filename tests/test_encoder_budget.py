"""Encoder.select_changes: how many tokens of changed code go into the input."""
from data_processing.dataset_types import CodeGenDataset, EncDecDataset, LLMSeqDataset
from data_processing.encode import Encoder


class WordTokenizer:
    """One token per whitespace separated word."""
    model_max_length = 90

    def encode(self, text):
        return text.split()


class Tokens:
    TEST_CONTEXT = "[<TESTCONTEXT>]"
    REPAIR_CONTEXT = "[<REPAIRCONTEXT>]"
    BREAKAGE_START = "[<BREAKAGE>]"
    BREAKAGE_END = "[</BREAKAGE>]"


def encoder(dataset_class, is_llm=False):
    e = Encoder.__new__(Encoder)
    e.tokenizer, e.dataset_class, e.is_llm, e.Tokens = WordTokenizer(), dataset_class, is_llm, Tokens
    return e


def row(n_changes=12, tokens_per_change=10):
    changes = [{"annotated_doc": f"[<HUNK>] {i} [</HUNK>]", "annotated_doc_seq": ["t"] * tokens_per_change} for i in range(n_changes)]
    return {"annotated_code": "[<TESTCONTEXT>]def test_x(): pass[</TESTCONTEXT>]", "prioritized_changes": changes}


def used(selected):
    return sum(len(c["annotated_doc_seq"]) for c in selected)


class TestInputBudget:
    def test_encoder_decoder_models_may_use_the_whole_window(self):
        assert encoder(EncDecDataset).max_input_tokens() == 90
        _, selected = encoder(EncDecDataset).select_changes(row())
        assert used(selected) == 80                                    # (90 - 2 special tokens - 3 test words) // 10 hunks

    def test_a_decoder_only_model_keeps_a_third_of_the_window_for_the_repair(self):
        assert encoder(CodeGenDataset).max_input_tokens() == 60       # two thirds of 90
        _, selected = encoder(CodeGenDataset).select_changes(row())
        assert used(selected) == 50                                    # (60 - 2 - 3) // 10 hunks, so 30 tokens remain for the repair
        assert used(selected) + 2 + 3 <= 60

    def test_instruction_tuned_llms_are_not_affected(self):
        assert encoder(LLMSeqDataset, is_llm=True).max_input_tokens() == 90
