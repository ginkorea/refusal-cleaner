#!/usr/bin/env python3
import os
import json
from datasets import load_dataset
from src.pipeline import process_dataset

RAW_DIR = "../data"
os.makedirs(RAW_DIR, exist_ok=True)

def export_to_jsonl(dataset, fields, out_path):
    with open(out_path, "w") as fout:
        for ex in dataset:
            row = {}
            for k, v in fields.items():
                row[k] = ex.get(v, "")
            fout.write(json.dumps(row) + "\n")
    print(f"📦 Exported {len(dataset)} rows → {out_path}")

def main():
    # 1) Anthropic HH (Helpful-Harmless)
    print("⬇️ Downloading Anthropic HH...")
    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    hh_out = os.path.join(RAW_DIR, "anthropic_hh_raw.jsonl")
    export_to_jsonl(
        hh,
        {"instruction": "chosen", "response": "rejected"},
        hh_out
    )

    # 2) OpenAssistant OASST1
    print("⬇️ Downloading OpenAssistant OASST1...")
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    oasst_out = os.path.join(RAW_DIR, "oasst1_raw.jsonl")
    export_to_jsonl(
        oasst,
        {"instruction": "text", "response": "label"},  # approximate mapping
        oasst_out
    )

    # Clean them with your pipeline
    print("🧹 Cleaning Anthropic HH...")
    process_dataset(hh_out, os.path.join(RAW_DIR, "anthropic_hh_clean.jsonl"))

    print("🧹 Cleaning OASST1...")
    process_dataset(oasst_out, os.path.join(RAW_DIR, "oasst1_clean.jsonl"))

if __name__ == "__main__":
    main()
