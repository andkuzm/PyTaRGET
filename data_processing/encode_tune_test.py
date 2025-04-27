import inspect

from huggingface_hub import login
from transformers import (
    PLBartTokenizer,
    AutoTokenizer, PLBartForConditionalGeneration, CodeGenForCausalLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM
)

from data_processing.encode import Encoder
from data_processing.llm_test import Tester_llm
from data_processing.test import Tester
from data_processing.train import Trainer
from dataset_types import EncDecDataset, PLBARTDataset, CodeGenDataset, LLMSeqDataset


class Tokens:
    BREAKAGE_START = "[<BREAKAGE>]"
    BREAKAGE_END = "[</BREAKAGE>]"
    TEST_CONTEXT = "[<TESTCONTEXT>]"
    TEST_CONTEXT_END = "[</TESTCONTEXT>]"
    REPAIR_CONTEXT = "[<REPAIRCONTEXT>]"
    REPAIR_CONTEXT_END = "[</REPAIRCONTEXT>]"
    DELETE_START = "[<DEL>]"
    DELETE_END = "[</DEL>]"
    ADD_START = "[<ADD>]"
    ADD_END = "[</ADD>]"
    HUNK = "[<HUNK>]"
    HUNK_END = "[</HUNK>]"

class Eftt:
    def __init__(self, annotated_cases_path, out_path, model, train_size, beam_size=5, hftoken=None):
        self.annotated_cases_path = annotated_cases_path
        self.out_path = out_path
        self.model = model
        self.train_size = train_size
        self.hftok = hftoken
        self.beam_size = beam_size
        if model == "plbart":
            self.model_path = "uclanlp/plbart-base"
            self.model_class = PLBartForConditionalGeneration
            self.tokenizer = PLBartTokenizer.from_pretrained(self.model_path)
            self.create_tokenizer()
            self.dataset_class = PLBARTDataset
        if model == "codegen":
            self.model_path = "salesforce/codegen-350M-multi"
            self.model_class = CodeGenForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.create_tokenizer()
            self.dataset_class = CodeGenDataset
        if model == "codet5p":
            self.model_path = "salesforce/codet5p-770m"
            self.model_class = AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.create_tokenizer()
            self.dataset_class = EncDecDataset
        if model == "llama":
            login(token=self.hftok)
            self.model_path = "meta-llama/Llama-3-70B-Instruct"
            self.model_class = AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, token=True
            )
            self.create_tokenizer_llm()
            self.dataset_class = LLMSeqDataset

        if model == "qwen":
            login(token=self.hftok)
            self.model_path = "Qwen/Qwen2.5-7B-Instruct"
            self.model_class = AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, token=True
            )
            self.create_tokenizer_llm()
            self.dataset_class = LLMSeqDataset

        if model == "gemma":
            login(token=self.hftok)
            self.model_path = "google/gemma-7b-it"
            self.model_class = AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, token=True
            )
            self.create_tokenizer_llm()
            self.dataset_class = LLMSeqDataset

        if model == "deepseek":
            login(token=self.hftok)
            self.model_path = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" #deepseek-ai/deepseek-coder-33b-instruct
            self.model_class = AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, token=True
            )
            self.create_tokenizer_llm()
            self.dataset_class = LLMSeqDataset

        # Gemini and Claude are API-based (not HuggingFace-hosted), so you’ll need API wrappers instead
        # if model == "gemini":
        #     self.model_path = "gemini"
        #     self.model_class = None
        #     self.tokenizer = None
        #     self.create_tokenizer_llm()
        #     self.dataset_class = LLMSeqDataset
        #
        # if model == "claude-sonnet":
        #     self.model_path = "claude-sonnet"
        #     self.model_class = None
        #     self.tokenizer = None
        #     self.create_tokenizer_llm()
        #     self.dataset_class = LLMSeqDataset
        # self.beam_size = beam_size

    def full_cycle(self):
        self.encode()
        self.train()
        self.validate()

    def encode(self):
        encoder = Encoder(self.annotated_cases_path, self.out_path, self.train_size, self.tokenizer, self.dataset_class, self.model, Tokens)
        encoder.encode()

    def train(self):
        trainer = Trainer(self.model, self.model_path, self.tokenizer, self.out_path, self.model_class, self.train_size)
        trainer.train()

    def validate(self):
        tester = Tester(self.model, self.model_class, self.tokenizer, self.out_path, self.beam_size, self.train_size)
        tester.validate()

    def validate_llm(self):
        tester = Tester_llm(
            model_name=self.model,
            model_path=self.model_path,
            model_class=self.model_class,
            dataset_path=self.out_path / self.model / str(self.train_size),
            token=self.hftok  # Pass the Hugging Face token if available
        )
        tester.run(out_path=self.out_path / self.model / str(self.train_size))

    def create_tokenizer(self):
        new_special_tokens = {
            "additional_special_tokens": self.tokenizer.additional_special_tokens
                                         + [v for k, v in inspect.getmembers(Tokens) if not k.startswith("_")]
        }

        # Only override if it's a placeholder (commonly very large int)
        if self.tokenizer.model_max_length > 10000:
            self.tokenizer.model_max_length = 512

        print("Final tokenizer model_max_length:", self.tokenizer.model_max_length)
        self.tokenizer.add_special_tokens(new_special_tokens)
        self.tokenizer.add_tokens(["<TAB>", "<NL>"])
        self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        self.tokenizer.save_pretrained(str(self.out_path / self.model / str(self.train_size) / "tokenizer"))

    def create_tokenizer_llm(self):
        new_special_tokens = {
            "additional_special_tokens": list({
                v for k, v in inspect.getmembers(Tokens) if not k.startswith("_")
            })
        }

        num_added_tokens = self.tokenizer.add_special_tokens(new_special_tokens)
        num_added_tokens += self.tokenizer.add_tokens(["<TAB>", "<NL>"])
        if num_added_tokens > 0 and hasattr(self.model_class, "resize_token_embeddings"):
            model = self.model_class.from_pretrained(self.model_path, trust_remote_code=True)
            model.resize_token_embeddings(len(self.tokenizer))
            model.save_pretrained(str(self.out_path / self.model / str(self.train_size) / "checkpoint-best"))

        # Only override if the value is clearly a placeholder (very high number)
        if self.tokenizer.model_max_length > 10000:
            self.tokenizer.model_max_length = 2048

        print("Final LLM tokenizer model_max_length:", self.tokenizer.model_max_length)
        self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        self.tokenizer.save_pretrained(str(self.out_path / self.model / str(self.train_size) / "tokenizer"))