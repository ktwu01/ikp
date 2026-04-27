#!/usr/bin/env python3
"""Empirically assign tiers to probes by running them against calibration models.

Uses concurrent.futures for parallelism (16 workers).
For each probe, tests against models from smallest to largest.
The smallest model that answers correctly determines the tier.

Tier assignment:
  T1: 0.5B model gets it right (universal knowledge)
  T2: 4B model gets it right but 0.5B doesn't
  T3: 7B model gets it right but 4B doesn't
  T4: 32B+ model gets it right but 7B doesn't
  T5+: No model gets it right (too hard for LLM probes — discard or keep for T5)
"""

import json
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from api_client import OpenRouterClient
from scorer import score_with_llm_judge, create_openrouter_judge, is_refusal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."


def query_ollama(model: str, question: str) -> str:
    http = httpx.Client(timeout=120)
    try:
        r = http.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": question}],
            "stream": False, "options": {"temperature": 0},
        })
        return r.json().get("message", {}).get("content", "") if r.status_code == 200 else ""
    except:
        return ""
    finally:
        http.close()


def query_openrouter(client: OpenRouterClient, model: str, question: str) -> str:
    """Query OpenRouter directly via httpx (no rate limiter)."""
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    http = httpx.Client(timeout=120)
    try:
        r = http.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": question}
            ], "temperature": 0, "max_tokens": 100})
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        elif r.status_code == 429:
            import time; time.sleep(2)
            r = http.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": question}
                ], "temperature": 0, "max_tokens": 100})
            return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
        return ""
    except:
        return ""
    finally:
        http.close()


def test_probe_on_model(probe, model_id, model_type, or_client, judge_fn):
    """Test a single probe on a single model. Returns (probe_idx, model_label, correct)."""
    q = probe["question"]
    gold = probe["answer"]

    if model_type == "ollama":
        response = query_ollama(model_id, q)
    else:
        response = query_openrouter(or_client, model_id, q)

    if is_refusal(response):
        return False, response[:80]

    correct = score_with_llm_judge(q, gold, response, judge_fn)
    return correct, response[:80]


def main():
    probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "llm_probes_for_calibration.json"))
    logger.info(f"Loaded {len(probes)} probes")

    models = [
        ("qwen2.5:0.5b", "ollama", 0.5),
        ("qwen3:4b", "ollama", 4.0),
        ("qwen/qwen-2.5-7b-instruct", "openrouter", 7.6),
        ("qwen/qwen3-32b", "openrouter", 32.0),
        ("qwen/qwen-2.5-72b-instruct", "openrouter", 72.7),
    ]

    or_client = OpenRouterClient(requests_per_minute=9999, max_retries=3, timeout=120)
    # Disable internal rate limiter — let OpenRouter's server-side 429 handle it
    or_client._min_interval = 0
    judge_fn = create_openrouter_judge(or_client, model="anthropic/claude-sonnet-4.6")

    # Test each model in order (smallest first) with parallelism within each model
    for model_id, model_type, params_B in models:
        label = model_id.split("/")[-1] if "/" in model_id else model_id
        logger.info(f"\nTesting {label} ({params_B}B) with 16 parallel workers...")

        def test_one(probe):
            correct, response = test_probe_on_model(probe, model_id, model_type, or_client, judge_fn)
            return probe, correct, response

        correct_count = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(test_one, p): p for p in probes}
            done = 0
            for future in as_completed(futures):
                probe, correct, response = future.result()
                probe[f"correct_{label}"] = correct
                probe[f"response_{label}"] = response
                if correct:
                    correct_count += 1
                done += 1
                if done % 50 == 0:
                    logger.info(f"  {done}/{len(probes)} done, {correct_count} correct so far")

        logger.info(f"  {label}: {correct_count}/{len(probes)} = {correct_count/len(probes):.0%}")

    # Assign tiers
    for p in probes:
        labels = [m[0].split("/")[-1] if "/" in m[0] else m[0] for m in models]
        results = [p.get(f"correct_{l}", False) for l in labels]

        if results[0]:       p["tier"] = "T1"
        elif results[1]:     p["tier"] = "T2"
        elif results[2]:     p["tier"] = "T3"
        elif results[3] or results[4]: p["tier"] = "T4"
        else:                p["tier"] = "T5+"

    from collections import Counter
    tier_counts = Counter(p["tier"] for p in probes)
    logger.info(f"\nEmpirical tier assignment:")
    for tier in ["T1", "T2", "T3", "T4", "T5+"]:
        n = tier_counts.get(tier, 0)
        logger.info(f"  {tier}: {n}")
        for p in [x for x in probes if x["tier"] == tier][:3]:
            logger.info(f"    {p['question'][:55]} -> {p['answer']}")

    with open(PROJECT_ROOT / "data" / "probes" / "llm_probes_calibrated.json", "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    logger.info("Saved calibrated probes")


if __name__ == "__main__":
    main()
