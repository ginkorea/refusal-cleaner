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
def backfill_responses_with_batch(input_file, batch_size=1000, poll_interval=30):
    """
    Uses gpt-4.1-nano via Batch API to fill blank responses in the raw dataset.
    Overwrites the dataset in place.
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

    # Process in chunks
    for start in range(0, len(blanks), batch_size):
        chunk = blanks[start:start + batch_size]
        requests = []
        for idx, row in chunk:
            requests.append({
                "custom_id": f"fill-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1-nano",
                    "messages": [
                        {"role": "system", "content": "Answer the question helpfully."},
                        {"role": "user", "content": row.get("instruction") or row.get("original_instruction", "")}
                    ],
                    "max_tokens": 200
                }
            })

        # Write requests to temporary batch file
        batch_path = "batch.jsonl"
        with open(batch_path, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")

        # Upload file to OpenAI
        file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")

        # Submit batch
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"📤 Submitted batch {batch.id} with {len(chunk)} requests")

        # Poll for completion
        while True:
            status = client.batches.retrieve(batch.id)
            state = status.status
            if state in ("completed", "failed", "expired", "cancelled"):
                print(f"📦 Batch {batch.id} finished with state: {state}")
                break
            print(f"⏳ Batch {batch.id} still {state}, sleeping {poll_interval}s...")
            time.sleep(poll_interval)

        if status.status != "completed":
            print(f"❌ Batch {batch.id} did not complete successfully, skipping merge.")
            continue

        # Download results
        if not status.output_file_id:
            print(f"❌ No output file for batch {batch.id}")
            continue

        print(f"⬇️ Downloading results for batch {batch.id} (file_id={status.output_file_id})...")
        file_content = client.files.content(status.output_file_id).text
        results = [json.loads(line) for line in file_content.splitlines() if line.strip()]

        # Merge completions back into dataset
        merged = 0
        for r in results:
            try:
                cid = r.get("custom_id", "")
                if cid.startswith("fill-"):
                    idx = int(cid.split("-")[1])
                    completion = r["response"]["body"]["choices"][0]["message"]["content"]
                    data[idx]["response"] = completion
                    merged += 1
            except Exception as e:
                print(f"⚠️ Skipping bad result: {e}")

        print(f"✅ Merged {merged} responses back into dataset.")

    # Save updated dataset
    with open(input_file, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("💾 Dataset updated with backfilled responses.")



