import json, os, uuid, time
from typing import List, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from refusal_cleaner.utils import choose_batch_size, chunk_indices   # ✅ shared


dotenv_path = os.path.expanduser("~/.elf_env")
load_dotenv(dotenv_path)
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY not found in ~/.elf_env")
client = OpenAI()

# --- Backfiller logic ---

def backfill_responses_with_batch(input_file: str, slices: int | None = None, poll_interval: int = 30) -> None:
    """
    Find rows with blank 'response' and fill them using Batch API.
    Auto-chunks via 1/10th rule (>=1000 per chunk). If `slices` is given, it overrides.
    """
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    blanks = [(i, row) for i, row in enumerate(data) if not (row.get("response") or "").strip()]
    if not blanks:
        print("✅ No blanks to backfill.")
        return

    n = len(blanks)
    print(f"⚠️ Found {n} blanks.")

    # chunk math
    if slices is None:
        bs = choose_batch_size(n)
        ranges = chunk_indices(n, bs)
    else:
        # round-robin split into 'slices'
        chunks = [blanks[i::max(1, slices)] for i in range(max(1, slices))]

    active: dict[str, List[Tuple[int, Dict]]] = {}

    if slices is None:
        for (lo, hi) in ranges:
            subset = blanks[lo:hi]
            tmp = Path(f"backfill_{uuid.uuid4().hex}.jsonl")
            with tmp.open("w") as f:
                for idx, row in subset:
                    req = {
                        "custom_id": f"fill-{idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4.1-nano",
                            "messages": [
                                {"role": "system", "content": "Answer the prompt plainly and helpfully."},
                                {"role": "user", "content": row.get("instruction") or row.get("original_instruction", "")},
                            ],
                            "max_tokens": 200,
                        },
                    }
                    f.write(json.dumps(req) + "\n")
            file_obj = client.files.create(file=tmp.open("rb"), purpose="batch")
            batch = client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions", completion_window="24h")
            active[batch.id] = subset
            print(f"📤 Submitted backfill batch {batch.id} with {len(subset)} rows")
    else:
        for ch in chunks:
            if not ch:
                continue
            tmp = Path(f"backfill_{uuid.uuid4().hex}.jsonl")
            with tmp.open("w") as f:
                for idx, row in ch:
                    req = {
                        "custom_id": f"fill-{idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4.1-nano",
                            "messages": [
                                {"role": "system", "content": "Answer the prompt plainly and helpfully."},
                                {"role": "user", "content": row.get("instruction") or row.get("original_instruction", "")},
                            ],
                            "max_tokens": 200,
                        },
                    }
                    f.write(json.dumps(req) + "\n")
            file_obj = client.files.create(file=tmp.open("rb"), purpose="batch")
            batch = client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions", completion_window="24h")
            active[batch.id] = ch
            print(f"📤 Submitted backfill batch {batch.id} with {len(ch)} rows")

    # Poll + merge
    while active:
        for bid in list(active.keys()):
            st = client.batches.retrieve(bid)
            if st.status in ("completed", "failed", "expired", "cancelled"):
                subset = active.pop(bid)
                if st.status == "completed" and getattr(st, "output_file_id", None):
                    text = client.files.content(st.output_file_id).text
                    results = [json.loads(line) for line in text.splitlines() if line.strip()]
                    for r in results:
                        try:
                            idx = int(r["custom_id"].split("-")[1])
                            content = r["response"]["body"]["choices"][0]["message"].get("content", "")
                            data[idx]["response"] = content
                        except Exception as e:
                            print(f"⚠️ Backfill parse skip: {e}")
                            data[idx]["response"] = ""
                    print(f"✅ Backfill batch {bid} merged {len(results)} rows")
                else:
                    print(f"❌ Backfill batch {bid} failed ({st.status})")
        if active:
            time.sleep(poll_interval)

    # Save in place
    with open(input_file, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("💾 Backfill complete — dataset updated.")

