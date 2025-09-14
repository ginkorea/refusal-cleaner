from refusal_cleaner import DATA_DIR

def main():
    """
    Downloads Anthropic HH and OASST1 datasets, exports them to JSONL,
    and runs the cleaning pipeline.
    """
    # Ensure data directory exists
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

    # Clean them with the pipeline
    print("🧹 Cleaning Anthropic HH...")
    process_dataset(hh_out, os.path.join(DATA_DIR, "anthropic_hh_clean.jsonl"))

    print("🧹 Cleaning OASST1...")
    process_dataset(oasst_out, os.path.join(DATA_DIR, "oasst1_clean.jsonl"))
