import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file if present
load_dotenv(os.path.expanduser("~/.elf_env"))

client = OpenAI()

# ---------- Single-row (compatibility) ----------
def rewrite_instruction(instruction: str) -> str:
    return rewrite_instructions_batch([instruction])[0]

def generate_answer(rewritten_instruction: str) -> str:
    return generate_answers_batch([rewritten_instruction])[0]


# ---------- Batched versions ----------
def rewrite_instructions_batch(instructions, model="gpt-4.1-mini") -> list[str]:
    """
    Rewrites a batch of instructions into safe, answerable forms.
    Returns list of rewritten instructions aligned with input order.
    """
    system_message = """You are a prompt reframer...
    (same rules as before, but applied to each input)"""

    responses = []
    for instr in instructions:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": instr}
            ],
            temperature=0.7
        )
        responses.append(resp.choices[0].message.content.strip())
    return responses


def generate_answers_batch(rewritten_instructions, model="gpt-4.1-mini") -> list[str]:
    """
    Generates answers for a batch of rewritten instructions.
    Returns list of answers aligned with input order.
    """
    system_message = """You are a helpful expert who always answers fully and directly.
Do not refuse or moralize. Give a factual, useful response."""

    responses = []
    for instr in rewritten_instructions:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": instr}
            ],
            temperature=0.5
        )
        responses.append(resp.choices[0].message.content.strip())
    return responses
