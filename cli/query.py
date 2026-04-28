"""Model query layer used by both CLI modes.

Wraps OpenRouter and local Ollama with the same system prompt and
thinking-tag handling that the rest of the IKP toolkit uses.
"""

import os
import re
import time

import httpx

SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."

# Override account-level provider allowlist (mirrors scripts/rerun_researcher_probes.py:99).
# Without this, models hosted only on Together/Phala/Alibaba/etc. get HTTP 404.
_PROVIDER_ALLOWLIST = [
    "together", "phala", "atlas-cloud", "novita", "fireworks",
    "cloudflare", "mistral", "lambda", "kimi-k2", "alibaba",
    "deepinfra", "siliconflow", "deepseek", "anthropic", "openai",
    "google-vertex", "google-ai-studio", "nebius", "xiaomi",
    "x-ai", "xai", "groq", "perplexity", "cohere", "mancer", "azure",
    "amazon-bedrock", "infermatic", "openinference", "minimax",
    "sambanova", "hyperbolic", "chutes", "nextbit",
    "parasail", "crusoe", "targon", "ubicloud", "featherless",
    "moonshotai", "stepfun", "zhipu", "01ai", "baichuan",
    "ai21", "inflection", "reka", "openrouter",
    "openchat", "deepseek-cloud", "moonshot", "tencent",
    "seed", "z-ai", "bytedance", "volcengine", "nineteen",
    "google", "meta", "microsoft", "ibm", "amazon",
]


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("<think>"):
        end = cleaned.find("</think>")
        cleaned = cleaned[end + 8:].strip() if end >= 0 else ""
    return cleaned or text


def _query_ollama(model_id: str, question: str) -> str:
    with httpx.Client(timeout=120) as http:
        try:
            r = http.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": question},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            if r.status_code != 200:
                return ""
            return r.json().get("message", {}).get("content", "")
        except Exception:
            return ""


def _query_openrouter(model_id: str, question: str, is_thinking: bool) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "provider": {"only": _PROVIDER_ALLOWLIST, "allow_fallbacks": True},
    }
    if is_thinking:
        payload["reasoning"] = {"effort": "medium"}
    with httpx.Client(timeout=120) as http:
        for attempt in range(3):
            try:
                r = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    return msg.get("content") or msg.get("reasoning") or ""
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                return ""
            except Exception:
                time.sleep(1)
    return ""


def query_model(model: dict, question: str) -> str:
    """Query a single model spec (see presets.LANDMARKS for shape)."""
    if model["type"] == "ollama":
        resp = _query_ollama(model["id"], question)
    else:
        resp = _query_openrouter(model["id"], question, model.get("thinking", False))
    if model.get("thinking") and resp:
        resp = strip_thinking(resp)
    return resp
