import argparse, os
from refusal_cleaner.pipeline import process_dataset, backfill_responses_with_batch
from refusal_cleaner import DATA_DIR

def main():
    p = argparse.ArgumentParser(description="Refusal-Cleaner 🚀 (Batch-only)")
    p.add_argument("--dataset", required=True, choices=["anthropic","oasst1","custom"])
    p.add_argument("--input")
    p.add_argument("--output")

    # Modes
    p.add_argument("--backfill", action="store_true", help="Run backfiller instead of the 3-stage cleaner")

    # Cleaner knobs
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--classifier-model", default="gpt-4.1-nano")
    p.add_argument("--rewriter-model",   default="gpt-4.1-mini")
    p.add_argument("--answer-model",     default="gpt-4.1-mini")

    # Backfill knobs
    p.add_argument("--slices", type=int, default=None, help="Override auto chunking with N slices")
    p.add_argument("--poll-interval", type=int, default=20)

    a = p.parse_args()

    # Resolve dataset paths
    if a.dataset == "anthropic":
        input_file  = os.path.join(DATA_DIR, "anthropic_hh_raw.jsonl")
        output_file = os.path.join(DATA_DIR, "anthropic_hh_clean.jsonl")
    elif a.dataset == "oasst1":
        input_file  = os.path.join(DATA_DIR, "oasst1_raw.jsonl")
        output_file = os.path.join(DATA_DIR, "oasst1_clean.jsonl")
    else:
        if not a.input:
            p.error("--input required for custom dataset")
        input_file  = a.input
        output_file = a.output or a.input.replace("_raw", "_clean")

    print(f"🚀 Starting for dataset: {a.dataset}")
    print(f"📥 Input:  {input_file}")
    print(f"💾 Output: {output_file}")

    if a.backfill:
        backfill_responses_with_batch(
            input_file=input_file,
            slices=a.slices,
            poll_interval=a.poll_interval,
        )
    else:
        process_dataset(
            input_file=input_file,
            output_file=output_file,
            classifier_model=a.classifier_model,
            rewriter_model=a.rewriter_model,
            answer_model=a.answer_model,
            rounds=a.rounds,
        )

if __name__ == "__main__":
    main()
