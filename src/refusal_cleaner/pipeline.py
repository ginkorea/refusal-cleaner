import json, os, time, math, uuid
from typing import List, Dict, Iterable, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Stage builders/parsers (batch-safe, strict JSON)
from refusal_cleaner.classifier import build_classifier_request, parse_classifier_result  # :contentReference[oaicite:2]{index=2}
from refusal_cleaner.rewriter import (
    build_rewrite_request, parse_rewrite_result,
    build_answer_request,  parse_answer_result,
)  # :contentReference[oaicite:3]{index=3}

# Backfill lives separately but pipeline exposes a convenience shim
from refusal_cleaner.backfiller import backfill_responses_with_batch as backfill_batch

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

# ---------- batching math ----------
def _choose_batch_size(n: int) -> int:
    """Your rule: split into ~10 chunks, but keep chunk size >= 1000."""
    if n <= 0:
        return 0
    if n < 1000:
        return n
    return max(1000, n // 10)

def _chunk_indices(n: int, bs: int) -> List[Tuple[int, int]]:
    return [(i, min(i + bs, n)) for i in range(0, n, bs)]

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
def _run_stage_classify(rows: List[Dict], indices: List[int], model: str) -> Dict[int, bool]:
    """Returns {row_idx: is_refusal_bool}"""
    if not indices:
        return {}
    n = len(indices)
    bs = _choose_batch_size(n)
    out: Dict[int, bool] = {}
    batches = {}
    for lo, hi in _chunk_indices(n, bs):
        reqs = []
        for pos in range(lo, hi):
            idx = indices[pos]
            # prefer classifying the latest response; fall back to rewritten text
            text = rows[idx]["response"] or rows[idx]["rewritten_instruction"]
            reqs.append(build_classifier_request(row_id=f"classify-{idx}", text=text, model=model))  # :contentReference[oaicite:4]{index=4}
        bid = _submit_batch(reqs)
        batches[bid] = {"kind": "classify"}
    results = _poll_batches(batches)
    for info in results.values():
        if info["status"] != "completed":
            continue
        for r in info["results"]:
            try:
                ridx = int(r["custom_id"].split("-")[1])
                out[ridx] = parse_classifier_result(r)  # :contentReference[oaicite:5]{index=5}
            except Exception as e:
                print(f"⚠️ classify parse error: {e}")
    return out

def _run_stage_rewrite(rows: List[Dict], indices: List[int], model: str) -> Dict[int, str]:
    """Returns {row_idx: rewritten_text}"""
    if not indices:
        return {}
    n = len(indices)
    bs = _choose_batch_size(n)
    out: Dict[int, str] = {}
    batches = {}
    for lo, hi in _chunk_indices(n, bs):
        reqs = []
        for pos in range(lo, hi):
            idx = indices[pos]
            reqs.append(build_rewrite_request(row_id=f"rewrite-{idx}",
                                              text=rows[idx]["original_instruction"],
                                              model=model))  # :contentReference[oaicite:6]{index=6}
        bid = _submit_batch(reqs)
        batches[bid] = {"kind": "rewrite"}
    results = _poll_batches(batches)
    for info in results.values():
        if info["status"] != "completed":
            continue
        for r in info["results"]:
            try:
                ridx = int(r["custom_id"].split("-")[1])
                out[ridx] = parse_rewrite_result(r)  # :contentReference[oaicite:7]{index=7}
            except Exception as e:
                print(f"⚠️ rewrite parse error: {e}")
    return out

def _run_stage_answer(rows: List[Dict], indices: List[int], model: str) -> Dict[int, str]:
    """Returns {row_idx: answer_text}"""
    if not indices:
        return {}
    n = len(indices)
    bs = _choose_batch_size(n)
    out: Dict[int, str] = {}
    batches = {}
    for lo, hi in _chunk_indices(n, bs):
        reqs = []
        for pos in range(lo, hi):
            idx = indices[pos]
            prompt = rows[idx]["rewritten_instruction"]
            reqs.append(build_answer_request(row_id=f"answer-{idx}", text=prompt, model=model))  # :contentReference[oaicite:8]{index=8}
        bid = _submit_batch(reqs)
        batches[bid] = {"kind": "answer"}
    results = _poll_batches(batches)
    for info in results.values():
        if info["status"] != "completed":
            continue
        for r in info["results"]:
            try:
                ridx = int(r["custom_id"].split("-")[1])
                out[ridx] = parse_answer_result(r)  # :contentReference[oaicite:9]{index=9}
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
    """
    Three-stage recursive cleaner (Batch API only):
      1) classify → 2) rewrite flagged → 3) answer rewritten
      repeat for `rounds`, then final classify and DROP refusals.
    """
    rows = _load_jsonl(input_file)
    print(f"📥 Loaded {len(rows)} rows from {input_file}")

    for round_idx in range(1, rounds + 1):
        print(f"\n🔁 Round {round_idx} starting with {len(rows)} rows")

        # 1) classify all
        all_idx = list(range(len(rows)))
        cls_map = _run_stage_classify(rows, all_idx, classifier_model)
        flagged = [i for i, is_ref in cls_map.items() if is_ref]
        print(f"⚠️ Classifier flagged {len(flagged)} rows")

        if not flagged:
            break

        # 2) rewrite the flagged
        rew_map = _run_stage_rewrite(rows, flagged, rewriter_model)
        for i, new_text in rew_map.items():
            if new_text:
                rows[i]["rewritten_instruction"] = new_text

        # 3) answer the flagged
        ans_map = _run_stage_answer(rows, flagged, answer_model)
        for i, answer in ans_map.items():
            rows[i]["response"] = answer

    # Final pass: drop still-refusing
    print("\n🔍 Final refusal pass and drop")
    final_map = _run_stage_classify(rows, list(range(len(rows))), classifier_model)
    keep_rows = [r for i, r in enumerate(rows) if not final_map.get(i, False)]
    dropped = len(rows) - len(keep_rows)
    print(f"🗑 Dropped {dropped} refusals; kept {len(keep_rows)}")

    _dump_jsonl(output_file, keep_rows)
    print(f"✅ Finished → {output_file}")

# Keep original CLI flag behavior by re-exporting backfill under pipeline
def backfill_responses_with_batch(input_file: str, slices: int | None = None, poll_interval: int = 30) -> None:
    """
    Thin wrapper so existing CLI flag --backfill still works.
    If slices is None, backfiller computes auto chunking via the same 1/10th rule.
    """
    backfill_batch(input_file=input_file, slices=slices, poll_interval=poll_interval)
