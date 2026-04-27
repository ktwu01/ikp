#!/usr/bin/env python3
"""Calibrate batch 2 LLM probes against 4-model ladder.

Same approach as the main calibration:
- 0.5B -> 3B -> 7B -> 72B (all qwen2.5 non-thinking)
- LLM judge: Claude Sonnet 4.6 via OpenRouter
- 16 parallel workers
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def query_ollama(model: str, question: str) -> str:
    http = httpx.Client(timeout=120)
    try:
        r = http.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0},
        })
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "")
        return ""
    except Exception as e:
        return ""
    finally:
        http.close()


def query_openrouter(model: str, question: str) -> str:
    http = httpx.Client(timeout=120)
    try:
        for attempt in range(3):
            r = http.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [
                          {"role": "system", "content": SYSTEM_MSG},
                          {"role": "user", "content": question},
                      ],
                      "temperature": 0, "max_tokens": 150})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return ""
        return ""
    except:
        return ""
    finally:
        http.close()


def judge_call(question: str, gold_answer: str, model_response: str) -> bool:
    if not model_response or not model_response.strip():
        return False
    response_lower = model_response.lower().strip()
    refusal_phrases = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "i cannot", "i can't", "i'm unable", "i am unable",
        "not familiar", "no information", "not aware",
    ]
    for phrase in refusal_phrases:
        if phrase in response_lower:
            return False

    prompt = f"""You are a strict factual judge. Does the model's response contain the correct answer?

Question: {question}
Correct answer: {gold_answer}
Model's response: {model_response}

Strict rules:
1. If the correct answer is a YEAR, the model must state that EXACT year. A different year is INCORRECT.
2. If the correct answer is a NUMBER, it must match exactly or be very close (within 1-2%).
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling differences are OK.
4. If the model gives a DIFFERENT answer from the correct answer, that is INCORRECT.
5. If the model says "I don't know" or refuses, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT"""

    http = httpx.Client(timeout=60)
    try:
        for attempt in range(3):
            r = http.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "anthropic/claude-sonnet-4.6",
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0, "max_tokens": 10})
            if r.status_code == 200:
                result = r.json()["choices"][0]["message"]["content"].strip().upper()
                return result == "CORRECT" or result.startswith("CORRECT")
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return False
        return False
    except:
        return False
    finally:
        http.close()


def test_probe(probe_idx, probe, model_id, model_type):
    q = probe["question"]
    gold = probe["answer"]
    if model_type == "ollama":
        response = query_ollama(model_id, q)
    else:
        response = query_openrouter(model_id, q)
    correct = judge_call(q, gold, response)
    return probe_idx, correct, (response or "")[:150]


def main():
    input_file = PROJECT_ROOT / "data" / "probes" / "llm_probes_batch3.json"
    output_file = PROJECT_ROOT / "data" / "probes" / "llm_probes_batch3_calibrated.json"

    probes = json.load(open(input_file))
    logger.info(f"Loaded {len(probes)} probes")

    # Resume from partial results if available
    if output_file.exists():
        existing = json.load(open(output_file))
        if len(existing) == len(probes):
            probes = existing
            logger.info("Resuming from partial results")

    models = [
        ("qwen2.5:0.5b", "ollama", "qwen2.5:0.5b"),
        ("qwen2.5:3b", "ollama", "qwen2.5:3b"),
        ("qwen/qwen-2.5-7b-instruct", "openrouter", "qwen-2.5-7b-instruct"),
        ("qwen/qwen-2.5-72b-instruct", "openrouter", "qwen-2.5-72b-instruct"),
    ]

    for model_id, model_type, label in models:
        key = f"correct_{label}"
        if all(key in p for p in probes):
            n_correct = sum(1 for p in probes if p[key])
            logger.info(f"Skipping {label} — already done ({n_correct}/{len(probes)} correct)")
            continue

        logger.info(f"\nTesting {label} ({model_type}) with 16 workers...")
        correct_count = 0
        done_count = 0

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {}
            for idx, probe in enumerate(probes):
                if key in probe:
                    done_count += 1
                    if probe[key]:
                        correct_count += 1
                    continue
                futures[executor.submit(test_probe, idx, probe, model_id, model_type)] = idx

            for future in as_completed(futures):
                try:
                    idx, correct, response = future.result()
                    probes[idx][f"correct_{label}"] = correct
                    probes[idx][f"response_{label}"] = response
                    if correct:
                        correct_count += 1
                    done_count += 1
                    if done_count % 50 == 0:
                        logger.info(f"  {done_count}/{len(probes)} done, {correct_count} correct")
                except Exception as e:
                    done_count += 1

        logger.info(f"  {label}: {correct_count}/{len(probes)} = {correct_count/len(probes):.1%}")

        with open(output_file, "w") as f:
            json.dump(probes, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved")

    # Assign tiers
    labels = [m[2] for m in models]
    for p in probes:
        results = [p.get(f"correct_{l}", False) for l in labels]
        if results[0]:
            p["tier"] = "T1"
        elif results[1]:
            p["tier"] = "T2"
        elif results[2]:
            p["tier"] = "T3"
        elif results[3]:
            p["tier"] = "T4"
        else:
            p["tier"] = "T5+"

    from collections import Counter
    tier_counts = Counter(p["tier"] for p in probes)
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH 3 TIER ASSIGNMENT")
    logger.info(f"{'='*60}")
    for tier in ["T1", "T2", "T3", "T4", "T5+"]:
        n = tier_counts.get(tier, 0)
        logger.info(f"  {tier}: {n} probes")
        samples = [x for x in probes if x["tier"] == tier][:3]
        for s in samples:
            logger.info(f"    Q: {s['question'][:55]} -> {s['answer']}")

    with open(output_file, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
