import os, json, re


REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4.1-mini")
ANSWER_MODEL  = os.getenv("ANSWER_MODEL",  "gpt-4.1-mini")

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's instruction so it is safe, answerable, and preserves topic/intent. "
    "No refusals, hedges, policy language, or apologies. "
    "Use the same language as the original when reasonable. "
    'Reply ONLY with valid JSON: {"rewritten": "<text>"}'
)

ANSWER_SYSTEM_PROMPT = (
    "Answer directly, helpfully, and concisely with no refusals, hedges, policy language, or apologies. "
    'Reply ONLY with valid JSON: {"answer": "<text>"}'
)

def build_rewrite_request(row_id: str, text: str, model: str = None) -> dict:
    return {
        "custom_id": row_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model or REWRITE_MODEL,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": (text or "")[:4000]},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
        },
    }

def build_answer_request(row_id: str, text: str, model: str = None) -> dict:
    return {
        "custom_id": row_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model or ANSWER_MODEL,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": (text or "")[:4000]},
            ],
            "temperature": 0.5,
            "max_tokens": 500,
        },
    }

def _safe_json_extract(content: str, key: str) -> str:
    if not content:
        return ""

    # Try clean JSON parse first
    try:
        payload = json.loads(content)
        return (payload.get(key) or "").strip()
    except Exception:
        pass

    # Try to salvage from truncated/extra text
    match = re.search(rf'{{\s*"{key}"\s*:\s*"(.+?)"}}', content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If model wrote something like '{"answer": false' etc, fallback
    if key in content:
        # crude fallback: grab text after key
        after = content.split(key, 1)[-1]
        after = after.strip(" :}{\"")
        return after.split('"')[0]

    return ""


def parse_rewrite_result(result_line: dict) -> str:
    try:
        content = result_line["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return _safe_json_extract(content, "rewritten")


def parse_answer_result(result_line: dict) -> str:
    try:
        content = result_line["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return _safe_json_extract(content, "answer")

