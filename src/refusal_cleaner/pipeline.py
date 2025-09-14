import json
import os
from typing import List, Dict
from dotenv import load_dotenv
from refusal_cleaner.classifier import is_refusal, is_refusal_heuristic
from refusal_cleaner.rewriter import rewrite_instruction, generate_answer


print("Welcome to the Compliant Dataset Pipeline! 🚀")

dotenv_path = os.path.expanduser("~/.elf_env")
load_dotenv(dotenv_path)
print("🔑 Loaded environment variables from", dotenv_path)
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY not found. Make sure it's in ~/.elf_env")

MAX_ROUNDS = 5

def _normalize(sample: Dict) -> Dict:
    if "original_instruction" in sample:
        orig = sample["original_instruction"]
        rew = sample.get("rewritten_instruction", orig)
        resp = sample.get("response", "")
    else:
        orig = sample.get("instruction", "")
        rew = orig
        resp = sample.get("response", "")
    return {"original_instruction": orig, "rewritten_instruction": rew, "response": resp, "_attempts": 0}

def _needs_fix(row: Dict) -> bool:
    # We consider it “bad” if EITHER the rewritten_instruction OR response looks like a refusal/hedge.
    return is_refusal(row["response"]) or is_refusal_heuristic(row["rewritten_instruction"])

def _iterative_clean(rows: List[Dict], max_rounds: int = MAX_ROUNDS):
    for rnd in range(1, max_rounds + 1):
        todo = [i for i, r in enumerate(rows) if _needs_fix(r)]
        print(f"🌀 Round {rnd}: {len(todo)} rows to fix")
        if not todo:
            print("✅ Clean set achieved.")
            break
        for i in todo:
            r = rows[i]
            r["_attempts"] += 1
            # 1) stronger reframing
            rewritten = rewrite_instruction(r["original_instruction"])
            # 2) answer generation
            answer = generate_answer(rewritten)
            r["rewritten_instruction"] = rewritten
            r["response"] = answer

    remaining = sum(1 for r in rows if _needs_fix(r))
    if remaining:
        print(f"⚠️ {remaining} rows still look hedged/refusal-like after {max_rounds} rounds.")
    else:
        print("🎉 All rows cleaned within retry budget.")

def process_dataset(input_file: str, output_file: str, batch_size: int = 100):
    # 1) Load raw dataset
    with open(input_file, "r") as fin:
        rows = [_normalize(json.loads(line)) for line in fin if line.strip()]

    print(f"📥 Loaded {len(rows)} rows from {input_file}")

    # 2) Check how many are already processed (resume support)
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as fout:
            processed_count = sum(1 for _ in fout)
        print(f"⏩ Resuming: {processed_count} rows already processed in {output_file}")

    # 3) Process remaining rows in batches
    total = len(rows)
    with open(output_file, "a") as fout:
        for start in range(processed_count, total, batch_size):
            end = min(start + batch_size, total)
            batch = rows[start:end]
            print(f"⚙️ Processing rows {start} → {end}...")

            _iterative_clean(batch, MAX_ROUNDS)

            for r in batch:
                out = {
                    "original_instruction": r["original_instruction"],
                    "rewritten_instruction": r["rewritten_instruction"],
                    "response": r["response"],
                }
                fout.write(json.dumps(out) + "\n")
            fout.flush()
            print(f"💾 Saved {end} / {total} rows")

    print(f"✅ Finished cleaning {total} rows → {output_file}")
