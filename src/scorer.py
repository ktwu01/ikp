"""Scoring engine for IKP probe responses.

Uses LLM-as-judge for all scoring decisions. The judge model evaluates
whether the model's response correctly answers the factual question,
given the gold answer. This avoids all edge cases with string matching,
numeric tolerance, field adjacency, etc.

For efficiency, the judge model should be small and cheap (e.g., a local
Ollama model or a cheap API model).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_refusal(response: str) -> bool:
    """Detect if the model refused to answer."""
    if not response or not response.strip():
        return True
    response_lower = response.lower().strip()
    # Short responses that are pure refusal
    refusal_phrases = [
        "i don't know", "i do not know",
        "i'm not sure", "i am not sure",
        "i cannot", "i can't",
        "i'm unable", "i am unable",
        "not familiar", "no information",
        "not aware", "unknown",
    ]
    for phrase in refusal_phrases:
        if phrase in response_lower:
            return True
    return False


def score_with_llm_judge(
    question: str,
    gold_answer: str,
    model_response: str,
    judge_fn,
) -> bool:
    """Score a response using an LLM as judge.

    Args:
        question: The probe question
        gold_answer: The verified correct answer
        model_response: The model's response to evaluate
        judge_fn: A callable(prompt) -> str that queries the judge LLM

    Returns:
        True if the judge determines the response is correct
    """
    if not model_response or not model_response.strip():
        return False

    if is_refusal(model_response):
        return False

    prompt = f"""You are a strict factual judge. Does the model's response contain the correct answer?

Question: {question}
Correct answer: {gold_answer}
Model's response: {model_response}

Strict rules:
1. If the correct answer is a YEAR (like 1931), the model must state that EXACT year. A different year (e.g. 1960 instead of 1931) is INCORRECT.
2. If the correct answer is a NUMBER, it must match exactly or be very close (within 1-2%).
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling differences are OK.
4. If the correct answer is a RESEARCH FIELD, accept synonyms (e.g. "networking" = "computer networking") but reject unrelated fields. The model must demonstrate it actually knows the person's field, not just guess randomly.
5. If the model clearly does not know the answer, gives an unrelated response, or says "I don't know", that is INCORRECT.
6. If the model gives a DIFFERENT year, a DIFFERENT name, or a DIFFERENT field from the correct answer, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT"""

    try:
        result = judge_fn(prompt).strip().upper()
        # Must check for exact "CORRECT" — not substring of "INCORRECT"
        if result == "CORRECT":
            return True
        if result.startswith("CORRECT"):
            return True
        # Handle cases like "CORRECT." or "CORRECT - the answer matches"
        if result.split()[0] == "CORRECT" if result else False:
            return True
        return False
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return False


def create_ollama_judge(model: str = "qwen3:4b"):
    """Create a judge function using a local Ollama model."""
    import httpx

    def judge_fn(prompt: str) -> str:
        http = httpx.Client(timeout=60)
        try:
            r = http.post("http://localhost:11434/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            })
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "")
            return ""
        finally:
            http.close()

    return judge_fn


def create_openrouter_judge(client=None, model: str = "anthropic/claude-sonnet-4"):
    """Create a judge function using an OpenRouter model.

    Uses direct httpx calls (no rate limiter) for parallel execution.
    """
    import os
    import httpx

    api_key = os.environ.get("OPENROUTER_API_KEY")

    def judge_fn(prompt: str) -> str:
        http = httpx.Client(timeout=60)
        try:
            for attempt in range(3):
                r = http.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0, "max_tokens": 10})
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                elif r.status_code == 429:
                    import time; time.sleep(2 * (attempt + 1))
                else:
                    return ""
            return ""
        except:
            return ""
        finally:
            http.close()

    return judge_fn
