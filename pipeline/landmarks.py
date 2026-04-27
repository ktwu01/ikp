"""Landmark model definitions and query functions.

Six landmark models define the boundaries between 7 tiers.
A probe's tier is the smallest landmark that answers it correctly.
"""

import os
import re
import time
import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."

LANDMARKS = [
    {"name": "qwen2.5-0.5b",   "id": "qwen2.5:0.5b",                   "type": "ollama",     "thinking": False, "tier": "T1"},
    {"name": "qwen2.5-7b",     "id": "qwen/qwen-2.5-7b-instruct",      "type": "openrouter", "thinking": False, "tier": "T2"},
    {"name": "qwen3-32b",      "id": "qwen/qwen3-32b",                  "type": "openrouter", "thinking": True,  "tier": "T3"},
    {"name": "qwen3-235b",     "id": "qwen/qwen3-235b-a22b",            "type": "openrouter", "thinking": True,  "tier": "T4"},
    {"name": "kimi-k2.5",      "id": "moonshotai/kimi-k2.5",            "type": "openrouter", "thinking": True,  "tier": "T5"},
    {"name": "gemini-3.1-pro", "id": "google/gemini-3.1-pro-preview",   "type": "openrouter", "thinking": False, "tier": "T6"},
]

TIER_NAMES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned.startswith('<think>'):
        end = cleaned.find('</think>')
        cleaned = cleaned[end + 8:].strip() if end >= 0 else ''
    return cleaned or text


def query_ollama(model_id: str, question: str) -> str:
    with httpx.Client(timeout=120) as http:
        try:
            r = http.post("http://localhost:11434/api/chat", json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "options": {"temperature": 0},
            })
            return r.json().get("message", {}).get("content", "") if r.status_code == 200 else ""
        except:
            return ""


def query_openrouter(model_id: str, question: str, is_thinking: bool = False) -> str:
    with httpx.Client(timeout=120) as http:
        for attempt in range(3):
            try:
                r = http.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model_id,
                          "messages": [{"role": "system", "content": SYSTEM_MSG},
                                       {"role": "user", "content": question}],
                          "temperature": 0})
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    content = msg.get("content") or ""
                    # For thinking models: OpenRouter puts thinking in 'reasoning'
                    # field. The 'content' field has the final answer if the model
                    # finished reasoning within the token budget.
                    if not content and msg.get("reasoning"):
                        # Token budget was too small — fallback to reasoning text
                        content = msg["reasoning"]
                    return content
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                return ""
            except:
                time.sleep(1)
        return ""


def query_landmark(landmark: dict, question: str) -> str:
    """Query a landmark model and return the cleaned response."""
    if landmark["type"] == "ollama":
        resp = query_ollama(landmark["id"], question)
    else:
        resp = query_openrouter(landmark["id"], question, is_thinking=landmark["thinking"])
    if landmark["thinking"] and resp:
        resp = strip_thinking(resp)
    return resp
