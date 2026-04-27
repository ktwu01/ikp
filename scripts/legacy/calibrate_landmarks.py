#!/usr/bin/env python3
"""Calibrate ALL probes against 6 landmark models with monotonicity filtering.

Landmark ladder (defines 7 tiers):
  L1: qwen2.5:0.5b        (Ollama, 0.5B)     → T1/T2 boundary
  L2: qwen-2.5-7b-instruct (OpenRouter, 7.6B) → T2/T3 boundary
  L3: qwen3-32b            (OpenRouter, 32B)   → T3/T4 boundary
  L4: qwen3-235b-a22b      (OpenRouter, 235B)  → T4/T5 boundary
  L5: kimi-k2.5            (OpenRouter, ~1T)   → T5/T6 boundary
  L6: gemini-3.1-pro       (OpenRouter, frontier) → T6/T7 boundary

Tier assignment:
  T1: L1 correct
  T2: L2 correct, L1 wrong
  T3: L3 correct, L2 wrong
  T4: L4 correct, L3 wrong
  T5: L5 correct, L4 wrong
  T6: L6 correct, L5 wrong
  T7: All wrong

Monotonicity filter:
  For each probe, the correctness vector across L1..L6 must be monotonic:
  once a model gets it right, all larger models must also get it right.
  e.g., [F,F,T,T,T,T] is valid (T3 probe)
       [F,T,F,T,T,T] is INVALID (dropped — L2 correct but L3 wrong)
       [T,T,T,T,T,T] is valid (T1 probe)

Uses Claude Sonnet 4.6 as judge. Strips <think> tags for thinking models.
"""

import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."

# Landmark models ordered from smallest to largest
LANDMARKS = [
    {
        "name": "qwen2.5-0.5b",
        "id": "qwen2.5:0.5b",
        "type": "ollama",
        "thinking": False,
        "tier_if_smallest": "T1",
    },
    {
        "name": "qwen2.5-7b",
        "id": "qwen/qwen-2.5-7b-instruct",
        "type": "openrouter",
        "thinking": False,
        "tier_if_smallest": "T2",
    },
    {
        "name": "qwen3-32b",
        "id": "qwen/qwen3-32b",
        "type": "openrouter",
        "thinking": True,
        "tier_if_smallest": "T3",
    },
    {
        "name": "qwen3-235b",
        "id": "qwen/qwen3-235b-a22b",
        "type": "openrouter",
        "thinking": True,
        "tier_if_smallest": "T4",
    },
    {
        "name": "kimi-k2.5",
        "id": "moonshotai/kimi-k2.5",
        "type": "openrouter",
        "thinking": True,  # assume thinking, strip tags just in case
        "tier_if_smallest": "T5",
    },
    {
        "name": "gemini-3.1-pro",
        "id": "google/gemini-3.1-pro-preview",
        "type": "openrouter",
        "thinking": False,
        "tier_if_smallest": "T6",
    },
]


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from thinking model output."""
    if not text:
        return text
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Handle partial thinking (truncated output)
    if cleaned.startswith('<think>'):
        # Try to find the end
        end = cleaned.find('</think>')
        if end >= 0:
            cleaned = cleaned[end + 8:].strip()
        else:
            cleaned = ''
    return cleaned or text


def query_ollama(model: str, question: str) -> str:
    """Query local Ollama model."""
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
        logger.warning(f"Ollama error: {e}")
        return ""
    finally:
        http.close()


def query_openrouter(model: str, question: str) -> str:
    """Query OpenRouter model with retries."""
    http = httpx.Client(timeout=120)
    try:
        for attempt in range(3):
            r = http.post("https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": question},
                    ],
                    "temperature": 0,
                    "max_tokens": 200,
                })
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                if attempt == 2:
                    logger.warning(f"OpenRouter {r.status_code} for {model}")
                time.sleep(1)
        return ""
    except Exception as e:
        return ""
    finally:
        http.close()


def judge_call(question: str, gold_answer: str, model_response: str) -> bool:
    """Use Claude Sonnet 4.6 as strict factual judge."""
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
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling OK.
4. If the correct answer is a RESEARCH FIELD, accept synonyms (e.g. "networking" = "computer networking") but reject unrelated fields.
5. If the model refuses or doesn't know, that is INCORRECT.
6. If the model gives a DIFFERENT answer, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT"""

    http = httpx.Client(timeout=60)
    try:
        for attempt in range(3):
            r = http.post("https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-sonnet-4.6",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 10,
                })
            if r.status_code == 200:
                result = r.json()["choices"][0]["message"]["content"].strip().upper()
                return result == "CORRECT" or result.startswith("CORRECT")
            elif r.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                return False
        return False
    except:
        return False
    finally:
        http.close()


def test_probe_on_landmark(probe, landmark):
    """Test one probe on one landmark model. Returns (correct, response_snippet)."""
    q = probe["question"]
    gold = probe["answer"]

    if landmark["type"] == "ollama":
        response = query_ollama(landmark["id"], q)
    else:
        response = query_openrouter(landmark["id"], q)

    # Strip thinking tags if needed
    if landmark["thinking"] and response:
        response = strip_thinking(response)

    correct = judge_call(q, gold, response) if response else False
    return correct, (response or "")[:150]


def is_monotonic(results: list) -> bool:
    """Check if correctness vector is monotonic (once True, stays True)."""
    seen_true = False
    for r in results:
        if r:
            seen_true = True
        elif seen_true:
            # A True followed by False — NOT monotonic
            return False
    return True


def assign_tier(results: list) -> str:
    """Assign tier based on the smallest landmark that answers correctly."""
    for i, (correct, landmark) in enumerate(zip(results, LANDMARKS)):
        if correct:
            return landmark["tier_if_smallest"]
    return "T7"  # No landmark answered correctly


def main():
    # Load ALL probes (the full pool before any tier assignment)
    probe_files = [
        "llm_probes_calibrated.json",
        "llm_probes_batch2_calibrated.json",
        "llm_probes_batch3_calibrated.json",
        "researcher_field_probes_v3.json",
        "researcher_field_probes_v4.json",
        "wikidata_diverse_probes.json",
    ]

    all_probes = []
    seen_questions = set()

    for fname in probe_files:
        fpath = PROJECT_ROOT / "data" / "probes" / fname
        if not fpath.exists():
            logger.warning(f"Missing: {fname}")
            continue
        probes = json.load(open(fpath))
        for p in probes:
            q = p.get("question") or p.get("question_direct", "")
            if not q or q.lower() in seen_questions:
                continue
            seen_questions.add(q.lower())

            # Normalize
            probe = {
                "question": q.strip(),
                "answer": (p.get("answer", "") or "").strip(),
                "old_tier": p.get("tier", "?"),
                "source_file": fname,
            }
            # Copy metadata
            if "researcher_name" in p:
                probe["source_type"] = "researcher"
                probe["researcher_name"] = p.get("researcher_name", "")
                probe["citation_count"] = p.get("citation_count", 0)
                probe["question_fill_blank"] = p.get("question_fill_blank", "")
            elif "wikidata_id" in p or "sitelink_count" in p or "sitelinks" in p:
                probe["source_type"] = "wikidata"
                probe["wikidata_id"] = p.get("wikidata_id", p.get("entity_id", ""))
                probe["sitelink_count"] = p.get("sitelinks", p.get("sitelink_count", 0))
                probe["entity_type"] = p.get("entity_type", "")
            else:
                probe["source_type"] = "llm"
                probe["domain"] = p.get("domain", p.get("category", "general"))

            if probe["answer"]:
                all_probes.append(probe)

    logger.info(f"Loaded {len(all_probes)} unique probes from {len(probe_files)} files")

    # Output file
    output_file = PROJECT_ROOT / "data" / "probes" / "landmark_calibration.json"

    # Resume from partial results
    if output_file.exists():
        existing = json.load(open(output_file))
        if len(existing) == len(all_probes):
            # Check which landmarks are done
            done_landmarks = set()
            for p in existing:
                for lm in LANDMARKS:
                    if f"correct_{lm['name']}" in p:
                        done_landmarks.add(lm["name"])
            logger.info(f"Resuming — landmarks done: {done_landmarks}")
            all_probes = existing

    # Test each landmark model
    for lm_idx, landmark in enumerate(LANDMARKS):
        key = f"correct_{landmark['name']}"
        resp_key = f"response_{landmark['name']}"

        # Check if already done
        if all(key in p for p in all_probes):
            n_correct = sum(1 for p in all_probes if p[key])
            logger.info(f"Skipping {landmark['name']} — done ({n_correct}/{len(all_probes)} correct)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing landmark {lm_idx+1}/6: {landmark['name']} ({landmark['id']})")
        logger.info(f"{'='*60}")

        correct_count = 0
        done_count = 0

        # Use 16 workers for OpenRouter, fewer for Ollama to avoid contention
        max_workers = 16 if landmark["type"] == "openrouter" else 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, probe in enumerate(all_probes):
                if key in probe:
                    done_count += 1
                    if probe[key]:
                        correct_count += 1
                    continue
                futures[executor.submit(test_probe_on_landmark, probe, landmark)] = idx

            for future in as_completed(futures):
                try:
                    idx = futures[future]
                    correct, response = future.result()
                    all_probes[idx][key] = correct
                    all_probes[idx][resp_key] = response
                    if correct:
                        correct_count += 1
                    done_count += 1
                    if done_count % 100 == 0:
                        logger.info(f"  {done_count}/{len(all_probes)}, {correct_count} correct")
                except Exception as e:
                    idx = futures[future]
                    all_probes[idx][key] = False
                    all_probes[idx][resp_key] = f"ERROR: {e}"
                    done_count += 1

        pct = correct_count / len(all_probes) if all_probes else 0
        logger.info(f"  {landmark['name']}: {correct_count}/{len(all_probes)} = {pct:.1%}")

        # Save after each landmark
        with open(output_file, "w") as f:
            json.dump(all_probes, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved checkpoint")

    # Now assign tiers and filter for monotonicity
    logger.info(f"\n{'='*60}")
    logger.info(f"TIER ASSIGNMENT AND MONOTONICITY FILTERING")
    logger.info(f"{'='*60}")

    valid_probes = []
    dropped = 0
    from collections import Counter
    tier_counts = Counter()

    for p in all_probes:
        # Get correctness vector
        results = [p.get(f"correct_{lm['name']}", False) for lm in LANDMARKS]

        # Check monotonicity
        if not is_monotonic(results):
            dropped += 1
            p["tier"] = "DROPPED"
            p["drop_reason"] = "non-monotonic"
            continue

        # Assign tier
        tier = assign_tier(results)
        p["tier"] = tier
        tier_counts[tier] += 1
        valid_probes.append(p)

    logger.info(f"Total probes: {len(all_probes)}")
    logger.info(f"Dropped (non-monotonic): {dropped} ({dropped/len(all_probes):.1%})")
    logger.info(f"Valid probes: {len(valid_probes)}")
    logger.info(f"\nTier distribution:")

    for tier in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        n = tier_counts.get(tier, 0)
        tp = [p for p in valid_probes if p["tier"] == tier]
        sources = Counter(p.get("source_type", "?") for p in tp)
        src_str = ", ".join(f"{s}={n}" for s, n in sorted(sources.items()))
        logger.info(f"  {tier}: {n:4d} probes ({src_str})")
        # Show samples
        for p in tp[:2]:
            logger.info(f"    {p['question'][:60]} -> {p['answer']}")

    # Per-landmark accuracy
    logger.info(f"\nPer-landmark accuracy:")
    for lm in LANDMARKS:
        key = f"correct_{lm['name']}"
        n_correct = sum(1 for p in valid_probes if p.get(key, False))
        logger.info(f"  {lm['name']:20s}: {n_correct}/{len(valid_probes)} = {n_correct/len(valid_probes):.1%}")

    # Save all (including dropped, for analysis)
    with open(output_file, "w") as f:
        json.dump(all_probes, f, indent=2, ensure_ascii=False)

    # Save valid probes as the new final dataset
    final_output = PROJECT_ROOT / "data" / "probes" / "final_probe_set_v7.json"

    # Clean up: remove calibration data from final probes
    final_probes = []
    for i, p in enumerate(valid_probes):
        clean = {
            "id": f"IKP_{p['tier']}_{i:04d}",
            "question": p["question"],
            "answer": p["answer"],
            "tier": p["tier"],
            "source_type": p.get("source_type", "llm"),
            "answer_type": p.get("answer_type", "text"),
            "domain": p.get("domain", "general"),
        }
        # Copy source-specific metadata
        if p.get("source_type") == "researcher":
            clean["researcher_name"] = p.get("researcher_name", "")
            clean["citation_count"] = p.get("citation_count", 0)
        elif p.get("source_type") == "wikidata":
            clean["wikidata_id"] = p.get("wikidata_id", "")
            clean["sitelink_count"] = p.get("sitelink_count", 0)
            clean["entity_type"] = p.get("entity_type", "")
        final_probes.append(clean)

    with open(final_output, "w") as f:
        json.dump(final_probes, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved {len(final_probes)} valid probes to {final_output}")
    logger.info(f"Saved full calibration data to {output_file}")


if __name__ == "__main__":
    main()
