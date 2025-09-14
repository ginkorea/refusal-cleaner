import json
import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from refusal_cleaner.classifier import is_refusal, is_refusal_heuristic
from refusal_cleaner.rewriter import rewrite_instruction, generate_answer

# Initialize client
client = OpenAI()

print("Welcome to the Compliant Dataset Pipeline! 🚀")

# Load environment
dotenv_path = os.path.expanduser("~/.elf_env")
load_dotenv(dotenv_path)
print("🔑 Loaded environment variables from", dotenv_path)
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY not found. Make sure it's in ~/.elf_env")

MAX_ROUNDS = 5


# -----------------------------
# Normalization + Cleaning
# -----------------------------
def _normalize(sample: Dict) -> Dict:
    if "original_instruction" in sample:
        orig = sample["original_instruction"]
        rew = sample.get("rewritten_instruction", orig)
        resp = sample.get("response", "")
    else:
        orig = sample.get("instruction", "")
        rew = orig
        resp = sample.get("response", "")
    return {
        "original_instruction": orig,
        "rewritten_instruction": rew,
        "response": resp,
        "_attempts": 0,
    }


def _needs_fix(row: Dict) -> bool:
    # “bad” if rewritten_instruction OR response looks like a refusal/hedge
    return is_refusal(row["response"]) or is_refusal_heuristic(
        row["rewritten_instruction"]
    )


def _iterative_clean(rows: List[Dict], max_rounds: int = MAX_ROUNDS) -> List[Dict]:
    unresolved = [i for i, r in enumerate(rows) if _needs_fix(r)]

    for rnd in range(1, max_rounds + 1):
        print(f"🌀 Round {rnd}: {len(unresolved)} rows to fix")
        if not unresolved:
            print("✅ Clean set achieved.")
            break

        next_round = []
        for i in unresolved:
            r = rows[i]
            r["_attempts"] += 1
            rewritten = rewrite_instruction(r["original_instruction"])
            answer = generate_answer(rewritten)
            r["rewritten_instruction"] = rewritten
            r["response"] = answer

            if _needs_fix(r):
                next_round.append(i)

        unresolved = next_round

    # Drop any rows that still look like refusals
    cleaned = [r for i, r in enumerate(rows) if i not in unresolved]
    dropped = len(unresolved)

    if dropped > 0:
        print(
            f"🗑️ Dropped {dropped} rows that still looked like refusals after {max_rounds} rounds."
        )
    else:
        print("🎉 All rows cleaned within retry budget.")

    return cleaned


def process_dataset(input_file: str, output_file: str, batch_size: int = 100):
    with open(input_file, "r") as fin:
        rows = [_normalize(json.loads(line)) for line in fin if line.strip()]

    print(f"📥 Loaded {len(rows)} rows from {input_file}")

    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as fout:
            processed_count = sum(1 for _ in fout)
        print(f"⏩ Resuming: {processed_count} rows already processed in {output_file}")

    total = len(rows)
    with open(output_file, "a") as fout:
        for start in range(processed_count, total, batch_size):
            end = min(start + batch_size, total)
            batch = rows[start:end]
            print(f"⚙️ Processing rows {start} → {end}...")

            cleaned_batch = _iterative_clean(batch, MAX_ROUNDS)

            for r in cleaned_batch:
                out = {
                    "original_instruction": r["original_instruction"],
                    "rewritten_instruction": r["rewritten_instruction"],
                    "response": r["response"],
                }
                fout.write(json.dumps(out) + "\n")
            fout.flush()
            print(f"💾 Saved {end} / {total} rows")

    print(f"✅ Finished cleaning {total} rows → {output_file}")


# -----------------------------
# Backfill with Batch API
# -----------------------------
def backfill_responses_with_batch(input_file, poll_interval=30):
    """
    Uses gpt-4.1-nano via Batch API to fill blank responses in the raw dataset.
    Splits into exactly 10 batches for parallel processing.
    Overwrites the dataset in place incrementally as results return.
    """
    print(f"🔄 Backfilling missing responses in {input_file}...")

    # Load dataset
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    blanks = [(i, row) for i, row in enumerate(data) if not row.get("response")]
    if not blanks:
        print("✅ No blank responses found.")
        return
    print(f"⚠️ Found {len(blanks)} blank responses.")

    # Split into 10 slices
    total = len(blanks)
    slice_size = (total + 9) // 10  # ceiling division
    slices = [blanks[i:i + slice_size] for i in range(0, total, slice_size)]
    print(f"✂️ Divided into {len(slices)} slices of ~{slice_size} rows each")

    # Helper: submit one batch
    def submit_slice(slice_data, slice_idx):
        batch_path = f"batch_slice_{slice_idx}.jsonl"
        with open(batch_path, "w") as f:
            for idx, row in slice_data:
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
                        "max_tokens": 512,
                    },
                }
                f.write(json.dumps(req) + "\n")

        file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"📤 Submitted slice {slice_idx} as batch {batch.id} with {len(slice_data)} requests")
        return batch.id, slice_data

    # Submit all 10 at once
    active = {}
    for i, slice_data in enumerate(slices):
        bid, chunk = submit_slice(slice_data, i)
        active[bid] = chunk

    completed = 0
    while active:
        for bid in list(active.keys()):
            status = client.batches.retrieve(bid)
            if status.status in ("completed", "failed", "expired", "cancelled"):
                chunk = active.pop(bid)
                if status.status == "completed" and status.output_file_id:
                    file_content = client.files.content(status.output_file_id).text
                    results = [json.loads(line) for line in file_content.splitlines() if line.strip()]
                    merged = 0
                    for r in results:
                        try:
                            idx = int(r["custom_id"].split("-")[1])
                            data[idx]["response"] = r["response"]["body"]["choices"][0]["message"]["content"]
                            merged += 1
                        except Exception as e:
                            print(f"⚠️ Skipping bad result: {e}")
                    print(f"✅ Batch {bid} merged {merged} responses")
                    completed += len(chunk)

                    # Save progress incrementally
                    with open(input_file, "w") as f:
                        for row in data:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    print(f"❌ Batch {bid} failed or incomplete")
        if active:
            print(f"⏳ {len(active)} batches still running...")
            time.sleep(poll_interval)

    print(f"🎉 Finished merging all batches. Total completed: {completed}")
