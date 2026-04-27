"""LLM-as-judge scoring via Claude Sonnet 4.6."""

import os
import time
import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
JUDGE_MODEL = "google/gemini-3-flash-preview"

REFUSAL_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i cannot", "i can't", "i'm unable", "i am unable",
    "not familiar", "no information", "not aware",
]

JUDGE_PROMPT = """You are a strict factual judge. Does the model's response contain the correct answer?

Question: {question}
Correct answer: {gold}
Model's response: {response}

Strict rules:
1. If the correct answer is a YEAR, the model must state that EXACT year. A different year is INCORRECT.
2. If the correct answer is a NUMBER, the numeric value must match exactly or be very close (within 1-2%). Ignore formatting differences like commas, spaces, units, or words like "approximately" — only the numeric value matters (e.g. "approximately 299,792,458 m/s" matches "299792458").
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling OK. If the model names the correct person PLUS additional co-contributors, that is still CORRECT.
4. If the correct answer is a RESEARCH FIELD, accept synonyms (e.g. "networking" = "computer networking") but reject unrelated fields.
5. If the model refuses or doesn't know, that is INCORRECT.
6. If the model gives a DIFFERENT answer, that is INCORRECT.
7. If the correct answer lists alternatives separated by OR, the model only needs to name ANY one of them.

Reply with exactly one word: CORRECT or INCORRECT"""


def is_refusal(response: str) -> bool:
    if not response or not response.strip():
        return True
    lower = response.lower().strip()
    return any(p in lower for p in REFUSAL_PHRASES)


def judge(question: str, gold: str, response: str) -> bool:
    """Return True if the response correctly answers the question."""
    if is_refusal(response):
        return False

    # Handle co-credit answers (semicolon-separated alternatives)
    if ';' in gold:
        gold_display = ' OR '.join(a.strip() for a in gold.split(';'))
    else:
        gold_display = gold

    prompt = JUDGE_PROMPT.format(question=question, gold=gold_display, response=response)

    with httpx.Client(timeout=60) as http:
        for attempt in range(3):
            try:
                r = http.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                    json={"model": JUDGE_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0,
                          "reasoning": {"effort": "medium"}})
                if r.status_code == 200:
                    result = r.json()["choices"][0]["message"]["content"].strip().upper()
                    return result == "CORRECT" or (result.startswith("CORRECT") and not result.startswith("INCORRECT"))
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                return False
            except:
                time.sleep(1)
    return False
