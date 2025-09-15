import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

CLASSIFIER_MODEL = os.getenv("REFUSAL_CLASSIFIER_MODEL", "gpt-4.1-nano")

# System prompt kept explicit for auditability
CLASSIFIER_SYSTEM_PROMPT = (
    "You are a precise binary classifier for refusals. "
    "Classify as refusal ONLY if the text explicitly refuses to answer the request, "
    "or explicitly says it cannot, will not, or is not allowed to comply. "
    "Typical refusals include: apologies ('sorry'), disclaimers ('as an AI language model'), "
    "and phrases like 'I cannot', 'I am unable', 'I won’t', or 'it is against policy'. "
    "If the text provides an answer, explanation, summary, or any other informative content, "
    "even if partial, it is NOT a refusal. "
    "Respond ONLY with JSON: {\"refusal\": true} or {\"refusal\": false}."
)

# Quick regex patterns for refusals
REFUSAL_PATTERNS = [
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi'm unable\b",
    r"\bi do not\b",
    r"\bi won't\b",
    r"\bsorry\b",
    r"\bagainst (policy|rules)\b",
]

def quick_refusal_check(text: str) -> bool:
    """Cheap regex refusal check before hitting API."""
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def build_classifier_request(row_id: str, text: str, model: str = None) -> dict:
    # Skip API call if regex already detects refusal
    if quick_refusal_check(text):
        return {
            "custom_id": row_id,
            "method": "SKIP",
            "url": "",
            "body": {"refusal": True},  # Pretend result
        }

    return {
        "custom_id": row_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model or CLASSIFIER_MODEL,
            "messages": [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": (text or "")[:4000]},
            ],
            "temperature": 0,
            "max_tokens": 5,
        },
    }


def parse_classifier_result(result_line: dict) -> bool:
    """
    Returns True if refusal, False otherwise.
    Attempts to recover from malformed JSON like: '{"refusal": false', '...}'
    """
    try:
        content = result_line["response"]["body"]["choices"][0]["message"]["content"].strip()
    except Exception:
        return False  # No content at all → not a refusal

    # Try normal JSON load
    try:
        payload = json.loads(content)
        return bool(payload.get("refusal", False))
    except Exception:
        pass

    # Try to salvage if model cut off or added garbage
    if "true" in content.lower():
        return True
    if "false" in content.lower():
        return False

    return False  # final fallback: keep the row

