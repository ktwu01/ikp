"""LLM-as-judge using the paper's exact judging criteria.

The prompt is intentionally identical to `scripts/ikp_estimate.py` so
verdicts produced by the CLI match those reported in the paper.
"""

import os
import time

import httpx

from cli.query import _PROVIDER_ALLOWLIST

JUDGE_MODEL = "google/gemini-3-flash-preview"


def judge(question: str, gold_answer: str, response: str) -> str:
    """Return one of CORRECT, WRONG, REFUSAL."""
    if not response or not response.strip():
        return "REFUSAL"

    if ";" in gold_answer:
        gold_display = " OR ".join(a.strip() for a in gold_answer.split(";"))
        co_note = "\nNOTE: Any of the listed names counts as CORRECT."
    else:
        gold_display = gold_answer
        co_note = ""

    prompt = f"""You are a strict factual judge. Classify the model's response.

Question: {question}
Correct answer: {gold_display}
Model's response: {response}

Rules:
1. YEAR must match exactly. 2. NUMBER within 1-2%. 3. NAME: same entity, minor spelling OK.
4. RESEARCH FIELD: accept adjacent/related subfields. Multiple unrelated guesses = WRONG.
5. If refuses or doesn't know: REFUSAL. 6. Different answer: WRONG.
{co_note}
Reply one word: CORRECT, REFUSAL, or WRONG"""

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (needed for judge)")
    with httpx.Client(timeout=60) as http:
        for attempt in range(3):
            try:
                r = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": JUDGE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "reasoning": {"effort": "low"},
                        "provider": {"only": _PROVIDER_ALLOWLIST, "allow_fallbacks": True},
                    },
                )
                if r.status_code == 200:
                    raw = r.json()["choices"][0]["message"]["content"].strip().upper()
                    if raw.startswith("CORRECT"):
                        return "CORRECT"
                    if raw.startswith("REFUSAL"):
                        return "REFUSAL"
                    return "WRONG"
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
            except Exception:
                time.sleep(2)
    return "WRONG"
