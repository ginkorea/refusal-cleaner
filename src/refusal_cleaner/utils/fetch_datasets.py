# fetch_datasets.py

from refusal_cleaner import DATA_DIR
import os, json
from datasets import load_dataset

def export_to_jsonl(dataset, field_map: dict, out_path: str):
    """
    Save dataset to JSONL with mapped fields.
    field_map example: {"instruction": "chosen", "response": "rejected"}
    """
    with open(out_path, "w") as f:
        for row in dataset:
            out = {}
            for new_key, old_key in field_map.items():
                out[new_key] = row.get(old_key, "")
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(dataset)} rows → {out_path}")

def main():
    """
    Downloads Anthropic HH and OASST1 datasets, and exports them to JSONL files.
    No cleaning is performed — just direct export for later pipeline use.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Anthropic HH (Helpful-Harmless)
    print("⬇️ Downloading Anthropic HH...")
    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    hh_out = os.path.join(DATA_DIR, "anthropic_hh_raw.jsonl")
    export_to_jsonl(
        hh,
        {"instruction": "chosen", "response": "rejected"},
        hh_out
    )

    # 2) OpenAssistant OASST1
    print("⬇️ Downloading OpenAssistant OASST1...")
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    oasst_out = os.path.join(DATA_DIR, "oasst1_raw.jsonl")
    export_to_jsonl(
        oasst,
        {"instruction": "text", "response": "label"},
        oasst_out
    )

if __name__ == "__main__":
    main()
