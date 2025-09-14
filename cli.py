#!/usr/bin/env python3
import argparse
from src.pipeline import process_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Compliant Dataset Cleaning CLI 🚀"
    )

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
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to process per batch (default=100)"
    )

    args = parser.parse_args()

    if args.dataset == "anthropic":
        input_file = "data/anthropic_hh_raw.jsonl"
        output_file = "data/anthropic_hh_clean.jsonl"
    elif args.dataset == "oasst1":
        input_file = "data/oasst1_raw.jsonl"
        output_file = "data/oasst1_clean.jsonl"
    elif args.dataset == "custom":
        if not args.input or not args.output:
            parser.error("--input and --output are required when --dataset=custom")
        input_file = args.input
        output_file = args.output
    else:
        raise ValueError("Invalid dataset selection.")

    print(f"🚀 Starting cleaning for dataset: {args.dataset}")
    print(f"📥 Input:  {input_file}")
    print(f"💾 Output: {output_file}")

    process_dataset(input_file, output_file, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
