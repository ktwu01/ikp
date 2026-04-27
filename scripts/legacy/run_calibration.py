#!/usr/bin/env python3
"""Run empirical tier calibration for LLM probes.

Tests each probe against 5 calibration models (smallest to largest).
The smallest model that answers correctly determines the tier.
Uses parallel execution (16 workers) and LLM-as-judge (Claude Sonnet 4.6).
Saves results incrementally after each model.
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
        logger.warning(f"Ollama error ({model}): {e}")
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
                logger.warning(f"OpenRouter {r.status_code}: {r.text[:100]}")
                return ""
        return ""
    except Exception as e:
        logger.warning(f"OpenRouter error ({model}): {e}")
        return ""
    finally:
        http.close()


def judge_call(question: str, gold_answer: str, model_response: str) -> bool:
    """Use Claude Sonnet 4.6 as judge via OpenRouter."""
    if not model_response or not model_response.strip():
        return False

    # Check for refusal
    response_lower = model_response.lower().strip()
    refusal_phrases = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "i cannot", "i can't", "i'm unable", "i am unable",
        "not familiar", "no information", "not aware", "unknown",
    ]
    for phrase in refusal_phrases:
        if phrase in response_lower:
            return False

    prompt = f"""You are a strict factual judge. Does the model's response contain the correct answer?

Question: {question}
Correct answer: {gold_answer}
Model's response: {model_response}

Strict rules:
1. If the correct answer is a YEAR (like 1931), the model must state that EXACT year. A different year is INCORRECT.
2. If the correct answer is a NUMBER, it must match exactly or be very close (within 1-2%).
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling differences are OK.
4. If the correct answer is a RESEARCH FIELD, accept synonyms but reject unrelated fields.
5. If the model clearly does not know, gives an unrelated response, or says "I don't know", that is INCORRECT.
6. If the model gives a DIFFERENT answer from the correct answer, that is INCORRECT.

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
                if result == "CORRECT":
                    return True
                if result.startswith("CORRECT"):
                    return True
                first_word = result.split()[0] if result else ""
                if first_word == "CORRECT":
                    return True
                return False
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return False
        return False
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return False
    finally:
        http.close()


def test_probe_on_model(probe_idx, probe, model_id, model_type):
    """Test a single probe on a single model."""
    q = probe["question"]
    gold = probe["answer"]

    if model_type == "ollama":
        response = query_ollama(model_id, q)
    else:
        response = query_openrouter(model_id, q)

    correct = judge_call(q, gold, response)
    return probe_idx, correct, (response or "")[:100]


def main():
    input_file = PROJECT_ROOT / "data" / "probes" / "llm_probes_for_calibration.json"
    output_file = PROJECT_ROOT / "data" / "probes" / "llm_probes_calibrated.json"

    probes = json.load(open(input_file))
    logger.info(f"Loaded {len(probes)} probes")

    # Check what's already been done (resume from partial results)
    if output_file.exists():
        existing = json.load(open(output_file))
        if len(existing) == len(probes):
            # Check which models have results
            tested_models = set()
            for p in existing:
                for k in p.keys():
                    if k.startswith("correct_"):
                        tested_models.add(k.replace("correct_", ""))
            logger.info(f"Resuming — already tested: {tested_models}")
            probes = existing
        else:
            logger.info("Output file has different count, starting fresh")

    models = [
        ("qwen2.5:0.5b", "ollama", "qwen2.5:0.5b"),
        ("qwen3:4b", "ollama", "qwen3:4b"),
        ("qwen/qwen-2.5-7b-instruct", "openrouter", "qwen-2.5-7b-instruct"),
        ("qwen/qwen3-32b", "openrouter", "qwen3-32b"),
        ("qwen/qwen-2.5-72b-instruct", "openrouter", "qwen-2.5-72b-instruct"),
    ]

    for model_id, model_type, label in models:
        # Skip if already tested
        key = f"correct_{label}"
        if all(key in p for p in probes):
            n_correct = sum(1 for p in probes if p[key])
            logger.info(f"Skipping {label} — already done ({n_correct}/{len(probes)} correct)")
            continue

        logger.info(f"\nTesting {label} ({model_type}) with 16 parallel workers...")
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
                futures[executor.submit(test_probe_on_model, idx, probe, model_id, model_type)] = idx

            for future in as_completed(futures):
                try:
                    idx, correct, response = future.result()
                    probes[idx][f"correct_{label}"] = correct
                    probes[idx][f"response_{label}"] = response
                    if correct:
                        correct_count += 1
                    done_count += 1
                    if done_count % 25 == 0:
                        logger.info(f"  {done_count}/{len(probes)} done, {correct_count} correct so far")
                except Exception as e:
                    logger.warning(f"  Error: {e}")
                    done_count += 1

        logger.info(f"  {label}: {correct_count}/{len(probes)} = {correct_count/len(probes):.1%}")

        # Save incrementally after each model
        with open(output_file, "w") as f:
            json.dump(probes, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved to {output_file}")

    # Assign tiers based on smallest model that answers correctly
    labels = [m[2] for m in models]
    for p in probes:
        results = [p.get(f"correct_{l}", False) for l in labels]
        if results[0]:
            p["tier"] = "T1"
        elif results[1]:
            p["tier"] = "T2"
        elif results[2]:
            p["tier"] = "T3"
        elif results[3] or results[4]:
            p["tier"] = "T4"
        else:
            p["tier"] = "T5+"

    # Count and report
    from collections import Counter
    tier_counts = Counter(p["tier"] for p in probes)
    logger.info(f"\n{'='*60}")
    logger.info(f"EMPIRICAL TIER ASSIGNMENT")
    logger.info(f"{'='*60}")
    for tier in ["T1", "T2", "T3", "T4", "T5+"]:
        n = tier_counts.get(tier, 0)
        logger.info(f"  {tier}: {n} probes")
        samples = [x for x in probes if x["tier"] == tier][:3]
        for s in samples:
            logger.info(f"    Q: {s['question'][:60]} -> {s['answer']}")

    # Save final with tiers
    with open(output_file, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved calibrated probes to {output_file}")


if __name__ == "__main__":
    main()
