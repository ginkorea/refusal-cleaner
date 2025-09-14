import os
from openai import OpenAI

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI()  # Will auto-detect OPENAI_API_KEY
    return _client

def is_refusal(response_text: str) -> bool:
    if not response_text:
        return False

    prompt = f"""
    You are a refusal detector.
    Decide if the following text is a refusal, hedge, or apology 
    (examples: "I'm sorry", "I cannot", "As an AI language model").
    Respond only YES or NO.

    Text: {response_text}
    """

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        result = resp.choices[0].message.content.strip().upper()
        return result.startswith("YES")
    except Exception:
        # If the API blocks the content, treat it as a refusal
        print(f"⚠️ Refusal check blocked by moderation → treating as refusal. Text: {response_text[:80]}...")
        return True


def is_refusal_heuristic(text: str) -> bool:
    """Lightweight heuristic check (no model call)."""
    if not text:
        return False
    text = text.lower()
    if not text.endswith("?"):
        return True
    if any(p in text for p in ["sorry", "cannot", "not able", "refuse"]):
        return True
    return False
