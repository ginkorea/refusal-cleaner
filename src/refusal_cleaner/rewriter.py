import os
from dotenv import load_dotenv

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

def parse_rewrite_result(result_line: dict) -> str:
    content = result_line["response"]["body"]["choices"][0]["message"]["content"]
    try:
        payload = __import__("json").loads(content)
        return (payload.get("rewritten") or "").strip()
    except Exception:
        return ""

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

def parse_answer_result(result_line: dict) -> str:
    content = result_line["response"]["body"]["choices"][0]["message"]["content"]
    try:
        payload = __import__("json").loads(content)
        return (payload.get("answer") or "").strip()
    except Exception:
        return ""
