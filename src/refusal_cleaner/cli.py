import argparse
import os
from refusal_cleaner.pipeline import process_dataset, backfill_responses_with_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def main():
    parser = argparse.ArgumentParser(description="Compliant Dataset Cleaning CLI 🚀")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["anthropic", "oasst1", "custom"],
        help="Which dataset to process."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Custom input JSONL file (required if --dataset=custom)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output JSONL file (required if --dataset=custom)"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run backfill mode instead of cleaning."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds for batch jobs."
    )

    args = parser.parse_args()

    # Resolve dataset paths
    if args.dataset == "anthropic":
        input_file = os.path.join(DATA_DIR, "anthropic_hh_raw.jsonl")
        output_file = os.path.join(DATA_DIR, "anthropic_hh_clean.jsonl")
    elif args.dataset == "oasst1":
        input_file = os.path.join(DATA_DIR, "oasst1_raw.jsonl")
        output_file = os.path.join(DATA_DIR, "oasst1_clean.jsonl")
    elif args.dataset == "custom":
        if not args.input or not args.output:
            parser.error("--input and --output are required when --dataset=custom")
        input_file, output_file = args.input, args.output
    else:
        raise ValueError("Invalid dataset selection.")

    print(f"🚀 Starting for dataset: {args.dataset}")
    print(f"📥 Input:  {input_file}")
    print(f"💾 Output: {output_file}")

    if args.backfill:
        print("🔄 Running backfill mode...")
        backfill_responses_with_batch(input_file, poll_interval=args.poll_interval)
    else:
        print("🌀 Running full cleaning mode...")
        process_dataset(input_file, output_file, poll_interval=args.poll_interval)
