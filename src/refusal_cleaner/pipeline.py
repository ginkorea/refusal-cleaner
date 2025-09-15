import json, os, time, uuid
from typing import List, Dict, Iterable, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Stage builders/parsers
from refusal_cleaner.classifier import build_classifier_request, parse_classifier_result
from refusal_cleaner.rewriter import (
    build_rewrite_request, parse_rewrite_result,
    build_answer_request,  parse_answer_result,
)
from refusal_cleaner.backfiller import backfill_responses_with_batch as backfill_batch
from refusal_cleaner.utils import choose_batch_size, chunk_indices   # ✅ use shared helpers

# ---------- OpenAI client ----------
dotenv_path = os.path.expanduser("~/.elf_env")
load_dotenv(dotenv_path)
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY not found in ~/.elf_env")
client = OpenAI()

# ---------- IO helpers ----------
def _normalize(sample: Dict) -> Dict:
    orig = sample.get("original_instruction") or sample.get("instruction", "")
    return {
        "original_instruction": orig,
        "rewritten_instruction": sample.get("rewritten_instruction", orig),
        "response": sample.get("response", ""),
    }

def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [_normalize(json.loads(line)) for line in f if line.strip()]

def _dump_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            out = {
                "original_instruction": r["original_instruction"],
                "rewritten_instruction": r.get("rewritten_instruction", r["original_instruction"]),
                "response": r.get("response", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

# ---------- Batch plumbing ----------
def _submit_batch(requests: List[Dict]) -> str:
    tmp = Path(f"batch_{uuid.uuid4().hex}.jsonl")
    with tmp.open("w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    file_obj = client.files.create(file=tmp.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=file_obj.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"📤 Submitted batch {batch.id} with {len(requests)} rows")
    return batch.id

def _poll_batches(active: Dict[str, Dict], poll_interval: int = 20) -> Dict[str, Dict]:
    """
    active: { batch_id: {'kind': 'classify'|'rewrite'|'answer'} }
    returns mapping batch_id -> dict(status='completed'|'failed', 'results': [raw_json_lines] or [])
    """
    done = {}
    while active:
        for bid in list(active.keys()):
            st = client.batches.retrieve(bid)
            if st.status in ("completed", "failed", "expired", "cancelled"):
                meta = active.pop(bid)
                if st.status == "completed" and getattr(st, "output_file_id", None):
                    text = client.files.content(st.output_file_id).text
                    lines = [json.loads(l) for l in text.splitlines() if l.strip()]
                    done[bid] = {"status": "completed", "kind": meta["kind"], "results": lines}
                    print(f"✅ Batch {bid} ({meta['kind']}) merged {len(lines)} rows")
                else:
                    done[bid] = {"status": "failed", "kind": meta["kind"], "results": []}
                    print(f"❌ Batch {bid} ({meta['kind']}) failed ({st.status})")
        if active:
            time.sleep(poll_interval)
    return done

# ---------- Stage runners ----------
def _run_stage_classify(rows: List[Dict], indices: List[int], model: str) -> Tuple[Dict[int, bool], int, int]:
    """
    Returns:
      - dict {row_idx: is_refusal_bool}
      - skipped (regex count)
      - used_api (API count)
    """
    if not indices:
        return {}, 0, 0

    n = len(indices)
    bs = choose_batch_size(n)
    out: Dict[int, bool] = {}
    batches = {}
    skipped = 0

    for lo, hi in chunk_indices(n, bs):
        to_submit: List[Tuple[int, dict]] = []
        for pos in range(lo, hi):
            idx = indices[pos]
            text = rows[idx]["response"] or rows[idx]["rewritten_instruction"]
            req = build_classifier_request(row_id=f"classify-{idx}", text=text, model=model)

            if req.get("method") == "SKIP":
                # Regex pre-filter: immediate refusal result (still flagged for rewrite/answer)
                out[idx] = True
                skipped += 1
            else:
                to_submit.append((idx, req))

        if to_submit:
            reqs = [r for _, r in to_submit]
            bid = _submit_batch(reqs)
            batches[bid] = {"kind": "classify", "map": [i for i, _ in to_submit]}

    # Poll API batches
    results = _poll_batches(batches)
    for bid, info in results.items():
        if info["status"] != "completed":
            continue
        for r in info["results"]:
            try:
                ridx = int(r["custom_id"].split("-")[1])
                out[ridx] = parse_classifier_result(r)
            except Exception as e:
                print(f"⚠️ classify parse error: {e}")

    total = len(indices)
    used_api = total - skipped
    print(f"📊 Classification summary → {total} total | {skipped} via regex | {used_api} via API")

    return out, skipped, used_api




def _run_stage_rewrite(rows: List[Dict], indices: List[int], model: str) -> Dict[int, str]:
    """Returns {row_idx: rewritten_text}"""
    if not indices:
        return {}
    n = len(indices)
    bs = choose_batch_size(n)
    out: Dict[int, str] = {}
    batches = {}
    for lo, hi in chunk_indices(n, bs):
        reqs = []
        for pos in range(lo, hi):
            idx = indices[pos]
            reqs.append(build_rewrite_request(row_id=f"rewrite-{idx}",
                                              text=rows[idx]["original_instruction"],
                                              model=model))
        bid = _submit_batch(reqs)
        batches[bid] = {"kind": "rewrite"}
    results = _poll_batches(batches)
    for info in results.values():
        if info["status"] != "completed":
            continue
        for r in info["results"]:
            try:
                ridx = int(r["custom_id"].split("-")[1])
                out[ridx] = parse_rewrite_result(r)
            except Exception as e:
                print(f"⚠️ rewrite parse error: {e}")
    return out


def _run_stage_answer(rows: List[Dict], indices: List[int], model: str) -> Dict[int, str]:
    """Returns {row_idx: answer_text}"""
    if not indices:
        return {}
    n = len(indices)
    bs = choose_batch_size(n)
    out: Dict[int, str] = {}
    batches = {}
    for lo, hi in chunk_indices(n, bs):
        reqs = []
        for pos in range(lo, hi):
            idx = indices[pos]
            prompt = rows[idx]["rewritten_instruction"]
            reqs.append(build_answer_request(row_id=f"answer-{idx}", text=prompt, model=model))
        bid = _submit_batch(reqs)
        batches[bid] = {"kind": "answer"}
    results = _poll_batches(batches)
    for info in results.values():
        if info["status"] != "completed":
            continue
        for r in info["results"]:
            try:
                ridx = int(r["custom_id"].split("-")[1])
                out[ridx] = parse_answer_result(r)
            except Exception as e:
                print(f"⚠️ answer parse error: {e}")
    return out


# ---------- Public API ----------
def process_dataset(
    input_file: str,
    output_file: str,
    classifier_model: str = "gpt-4.1-nano",
    rewriter_model:   str = "gpt-4.1-mini",
    answer_model:     str = "gpt-4.1-mini",
    rounds: int = 3,
) -> None:
    rows = _load_jsonl(input_file)
    print(f"📥 Loaded {len(rows)} rows from {input_file}")

    total_regex = 0
    total_api = 0
    modified_idx: set[int] = set(range(len(rows)))   # only round 1 starts full

    for round_idx in range(1, rounds + 1):
        if not modified_idx:
            print(f"\n🔁 Round {round_idx}: nothing left to check, stopping early.")
            break

        print(f"\n🔁 Round {round_idx} starting with {len(modified_idx)} rows")

        cls_map, skipped, used_api = _run_stage_classify(rows, sorted(modified_idx), classifier_model)
        total_regex += skipped
        total_api += used_api

        flagged = [i for i, is_ref in cls_map.items() if is_ref]
        print(f"⚠️ Classifier flagged {len(flagged)} rows")

        if not flagged:
            modified_idx.clear()
            continue

        rew_map = _run_stage_rewrite(rows, flagged, rewriter_model)
        print(f"✍️ Rewrote {len(rew_map)} rows")
        for i, new_text in rew_map.items():
            if new_text:
                rows[i]["rewritten_instruction"] = new_text

        ans_map = _run_stage_answer(rows, flagged, answer_model)
        print(f"💬 Answered {len(ans_map)} rows")
        for i, answer in ans_map.items():
            rows[i]["response"] = answer

        # Next round: only recheck what was just changed
        modified_idx = set(flagged)

    # Final pass on ALL rows
    print("\n🔍 Final refusal pass and drop")
    final_map, skipped, used_api = _run_stage_classify(rows, list(range(len(rows))), classifier_model)
    total_regex += skipped
    total_api += used_api

    keep_rows = [r for i, r in enumerate(rows) if final_map.get(i) is False]
    dropped = len(rows) - len(keep_rows)
    print(f"🗑 Dropped {dropped} refusals; kept {len(keep_rows)}")

    print(f"\n📊 Cumulative classifier usage across all rounds:")
    print(f"   - {total_regex} rows handled via regex pre-filter")
    print(f"   - {total_api} rows handled via API")

    _dump_jsonl(output_file, keep_rows)
    print(f"✅ Finished → {output_file}")


# Keep original CLI flag behavior by re-exporting backfill under pipeline
def backfill_responses_with_batch(input_file: str, slices: int | None = None, poll_interval: int = 30) -> None:
    """
    Thin wrapper so existing CLI flag --backfill still works.
    If slices is None, backfiller computes auto chunking via the same 1/10th rule.
    """
    backfill_batch(input_file=input_file, slices=slices, poll_interval=poll_interval)
