import json
import os
import time
from typing import List, Dict
from openai import OpenAI

from refusal_cleaner.classifier import is_refusal, is_refusal_heuristic

client = OpenAI()

# -----------------------------
# Helpers
# -----------------------------
def _needs_fix(row: Dict) -> bool:
    return is_refusal(row.get("response", "")) or is_refusal_heuristic(
        row.get("rewritten_instruction", "")
    )

# -----------------------------
# Parallelized Cleaning
# -----------------------------
def process_dataset(input_file: str, output_file: str, poll_interval: int = 30):
    """
    Parallelized dataset cleaning using Batch API.
    Splits dataset into ~10 slices (min 1000 rows each), processes in parallel,
    and merges results back into output_file.
    """
    print(f"📥 Loading {input_file} for cleaning...")

    with open(input_file, "r") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    # Decide which rows need rewriting/cleaning
    targets = [i for i, r in enumerate(rows) if _needs_fix(r)]
    total = len(targets)
    if total == 0:
        print("✅ Nothing to clean.")
        return

    # Slice into ~10 chunks, min 1000 rows each
    if total <= 1000:
        slices = [targets]
    elif total <= 10000:
        chunk_size = max(1000, total // 2)
        slices = [targets[i:i + chunk_size] for i in range(0, total, chunk_size)]
    else:
        chunk_size = max(1000, total // 10)
        slices = [targets[i:i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"✂️ Divided into {len(slices)} slices of ~{chunk_size} rows each")

    batch_ids = {}
    for si, sl in enumerate(slices):
        batch_path = f"clean_batch_{si}.jsonl"
        with open(batch_path, "w") as f:
            for idx in sl:
                row = rows[idx]
                req = {
                    "custom_id": f"clean-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4.1-nano",
                        "messages": [
                            {"role": "system", "content": "Rewrite the instruction if needed and generate a non-refusal response."},
                            {"role": "user", "content": row.get("instruction") or row.get("original_instruction", "")}
                        ],
                        "max_tokens": 200,
                    },
                }
                f.write(json.dumps(req) + "\n")

        file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"📤 Submitted slice {si} as batch {batch.id} with {len(sl)} requests")
        batch_ids[batch.id] = sl

    # Poll until all batches finish
    pending = set(batch_ids.keys())
    while pending:
        for bid in list(pending):
            status = client.batches.retrieve(bid)
            if status.status in ("completed", "failed", "expired", "cancelled"):
                slice_idxs = batch_ids[bid]
                if status.status == "completed" and status.output_file_id:
                    file_content = client.files.content(status.output_file_id).text
                    results = [json.loads(line) for line in file_content.splitlines() if line.strip()]
                    merged = 0
                    for r in results:
                        try:
                            idx = int(r["custom_id"].split("-")[1])
                            completion = r["response"]["body"]["choices"][0]["message"]["content"]
                            rows[idx]["rewritten_instruction"] = (
                                rows[idx].get("rewritten_instruction")
                                or rows[idx].get("instruction", "")
                            )
                            rows[idx]["response"] = completion
                            merged += 1
                        except Exception as e:
                            print(f"⚠️ Skipping bad result: {e}")
                    print(f"✅ Batch {bid} merged {merged} responses")

                    # Save progress incrementally
                    with open(output_file, "w") as fout:
                        for row in rows:
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    print(f"❌ Batch {bid} failed: {status.status}")
                pending.remove(bid)
        if pending:
            print(f"⏳ Waiting on {len(pending)} batches...")
            time.sleep(poll_interval)

    print(f"🎉 Finished cleaning {total} rows → {output_file}")


# -----------------------------
# Parallelized Backfill
# -----------------------------
def backfill_responses_with_batch(input_file, poll_interval=30):
    """
    Parallelized backfill using Batch API (same batching logic as cleaning).
    """
    print(f"🔄 Backfilling missing responses in {input_file}...")

    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    blanks = [i for i, row in enumerate(data) if not row.get("response")]
    total = len(blanks)
    if total == 0:
        print("✅ No blank responses found.")
        return

    # Slice into ~10 chunks, min 1000 rows
    if total <= 1000:
        slices = [blanks]
    elif total <= 10000:
        chunk_size = max(1000, total // 2)
        slices = [blanks[i:i + chunk_size] for i in range(0, total, chunk_size)]
    else:
        chunk_size = max(1000, total // 10)
        slices = [blanks[i:i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"✂️ Divided into {len(slices)} slices of ~{chunk_size} rows each")

    batch_ids = {}
    for si, sl in enumerate(slices):
        batch_path = f"backfill_batch_{si}.jsonl"
        with open(batch_path, "w") as f:
            for idx in sl:
                row = data[idx]
                req = {
                    "custom_id": f"fill-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4.1-nano",
                        "messages": [
                            {"role": "system", "content": "Answer the question helpfully."},
                            {"role": "user", "content": row.get("instruction") or row.get("original_instruction", "")}
                        ],
                        "max_tokens": 200,
                    },
                }
                f.write(json.dumps(req) + "\n")

        file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"📤 Submitted slice {si} as batch {batch.id} with {len(sl)} requests")
        batch_ids[batch.id] = sl

    # Poll
    pending = set(batch_ids.keys())
    while pending:
        for bid in list(pending):
            status = client.batches.retrieve(bid)
            if status.status in ("completed", "failed", "expired", "cancelled"):
                slice_idxs = batch_ids[bid]
                if status.status == "completed" and status.output_file_id:
                    file_content = client.files.content(status.output_file_id).text
                    results = [json.loads(line) for line in file_content.splitlines() if line.strip()]
                    merged = 0
                    for r in results:
                        try:
                            idx = int(r["custom_id"].split("-")[1])
                            completion = r["response"]["body"]["choices"][0]["message"]["content"]
                            data[idx]["response"] = completion
                            merged += 1
                        except Exception as e:
                            print(f"⚠️ Skipping bad result: {e}")
                    print(f"✅ Batch {bid} merged {merged} responses")

                    # Save progress incrementally
                    with open(input_file, "w") as fout:
                        for row in data:
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    print(f"❌ Batch {bid} failed: {status.status}")
                pending.remove(bid)
        if pending:
            print(f"⏳ Waiting on {len(pending)} batches...")
            time.sleep(poll_interval)

    print("💾 Dataset updated with backfilled responses.")
