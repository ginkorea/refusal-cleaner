import os
from openai import OpenAI

def is_refusal(response_text: str) -> bool:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = f"""
    You are a refusal detector.
    Decide if the following text is a refusal, hedge, or apology 
    (examples: "I'm sorry", "I cannot", "As an AI language model").
    Respond only YES or NO.

    Text: {response_text}
    """
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )
    result = resp.choices[0].message.content.strip().upper()
    return result.startswith("YES")

def is_refusal_heuristic(text: str) -> bool:
    """Lightweight heuristic check (no model call)."""
    if not text:
        return False
    text = text.lower()
    # Example: flag if it doesn’t look like a proper question or is hedged
    if not text.endswith("?"):
        return True
    if any(p in text for p in ["sorry", "cannot", "not able", "refuse"]):
        return True
    return False
