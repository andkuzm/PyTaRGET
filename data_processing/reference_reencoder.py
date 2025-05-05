import json
from pathlib import Path
from tqdm import tqdm

class Re_encoder:
    def __init__(self, original_ds, out_path, tokenizer):
        with open(original_ds, "r", encoding="utf-8") as f:
            self.original_ds = json.load(f)
        self.out_path = Path(out_path)
        self.tokenizer = tokenizer

    def decode_reencode(self):
        ds = self.decode(self.original_ds)
        self.save(ds, self.out_path)

    def decode(self, original_ds):
        new_rows = []
        for row in tqdm(original_ds, desc="Decoding and preparing"):
            input_ids = row.get("input_ids", row.get("input"))
            label_ids = row.get("labels", row.get("output"))

            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            if "[<TESTCONTEXT>]" in input_text and "[<REPAIRCONTEXT>]" in input_text and "[</TESTCONTEXT>]" not in input_text:
                input_text = input_text.replace("[<REPAIRCONTEXT>]", "[</TESTCONTEXT>][<REPAIRCONTEXT>]")
            output_text = self.tokenizer.decode(label_ids, skip_special_tokens=False)

            new_rows.append({
                "input": input_text.strip(),
                "output": output_text.strip()
            })
        return new_rows

    def save(self, ds, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ds, f, indent=2, ensure_ascii=False)
