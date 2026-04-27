#!/usr/bin/env python3
"""Generate LLM probes targeting specific difficulty tiers.

The key insight: T1 probes (0.5B can answer) are trivially easy to generate,
but T3/T4 probes (need 32B/235B to answer) require very specific, obscure facts.

This script generates batches of probes at controlled difficulty levels,
then feeds them into the calibration pipeline. The pipeline determines the
actual tier — we just aim for the right ballpark.

Usage:
  python -m pipeline.generate_llm_probes --difficulty hard --count 200
  python -m pipeline.generate_llm_probes --difficulty medium --count 200
"""

import json
import os
import time
import argparse
import logging
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Use a strong model to generate probes
GENERATOR_MODEL = "anthropic/claude-sonnet-4.6"

PROMPT_MEDIUM = """Generate {count} factual questions that a 7B language model would struggle with but a 32B model could answer. These should be moderately obscure facts.

Good categories:
- Specific historical dates of lesser-known events (treaties, battles, founding of institutions)
- Geographic features that aren't the most famous (second-highest peak in X, longest river in Y)
- Scientific discoveries and their dates or discoverers
- Cultural works: who composed/wrote/designed specific works
- Specific years for inventions or engineering milestones

Rules:
- Each question must have ONE unambiguous, verifiable, time-stable answer
- Answer should be short (1-5 words, a number, or a year)
- No questions about current leaders, populations, GDP, or anything that changes
- No questions where the answer appears in the question text
- Diverse topics — mix geography, history, science, culture, engineering
- These should NOT be common trivia — aim for facts that require real knowledge

Output as a JSON array: [{{"question": "...", "answer": "...", "category": "..."}}]
Only output the JSON array, nothing else."""

PROMPT_HARD = """Generate {count} very hard factual questions that only a very large language model (200B+ parameters) could answer. These should be quite obscure facts.

Good categories:
- Exact founding years of obscure institutions, cities, or organizations
- Second or third ranked geographic features in specific regions
- Specific scientific discoverers of niche effects or phenomena
- Architects of lesser-known buildings
- Dates of obscure historical treaties or battles
- Properties of rare chemical elements (melting points, atomic numbers of uncommon elements)
- Composers of specific classical works that aren't the most famous
- Engineers/inventors of specific technologies

Rules:
- Each question must have ONE unambiguous, verifiable, time-stable answer
- Answer should be short (1-5 words, a number, or a year)
- No questions about current events or anything that changes over time
- No questions where the answer appears in the question text
- These should be facts that even well-educated people wouldn't know offhand
- Verify your answers are correct

Output as a JSON array: [{{"question": "...", "answer": "...", "category": "..."}}]
Only output the JSON array, nothing else."""

PROMPT_VERY_HARD = """Generate {count} extremely obscure factual questions that only frontier language models (1T+ parameters) could possibly answer. These test the absolute limits of memorized knowledge.

Good categories:
- Founding years of very obscure local institutions or small organizations
- Specific minor geographic features (capes, small islands, minor rivers)
- Dates of very minor historical events or local treaties
- Properties of the rarest chemical elements
- Obscure architectural works and their designers
- Minor scientific discoveries and their exact dates
- Very specific cultural works from non-Western traditions

Rules:
- Each question must have ONE unambiguous, verifiable, time-stable answer
- Answer should be short (1-5 words, a number, or a year)
- No questions about current events or anything that changes over time
- No questions where the answer appears in the question text
- These should be facts that only someone with encyclopedic knowledge would know
- Verify your answers are correct — do not make up facts

Output as a JSON array: [{{"question": "...", "answer": "...", "category": "..."}}]
Only output the JSON array, nothing else."""


def generate_batch(difficulty: str, count: int) -> list[dict]:
    """Generate a batch of probes at a given difficulty level."""
    if difficulty == "medium":
        prompt = PROMPT_MEDIUM.format(count=count)
    elif difficulty == "hard":
        prompt = PROMPT_HARD.format(count=count)
    elif difficulty == "very_hard":
        prompt = PROMPT_VERY_HARD.format(count=count)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    with httpx.Client(timeout=120) as http:
        r = http.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": GENERATOR_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7})
        if r.status_code != 200:
            logger.error(f"Generator returned {r.status_code}")
            return []

        content = r.json()["choices"][0]["message"]["content"]

    # Parse JSON from response (handle markdown code blocks)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    try:
        probes = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Content: {content[:200]}")
        return []

    # Validate and normalize
    valid = []
    for p in probes:
        if not isinstance(p, dict):
            continue
        q = p.get("question", "").strip()
        a = p.get("answer", "").strip()
        if not q or not a:
            continue
        # Check answer not in question
        if len(a) > 3 and a.lower() in q.lower():
            continue
        valid.append({
            "question": q,
            "answer": str(a),
            "source_type": "llm",
            "domain": p.get("category", "general"),
            "difficulty_target": difficulty,
        })

    return valid


def generate_all(difficulty: str, total: int, batch_size: int = 50) -> list[dict]:
    """Generate probes in batches until we have enough."""
    all_probes = []
    seen = set()
    batches = (total + batch_size - 1) // batch_size

    for i in range(batches):
        remaining = total - len(all_probes)
        if remaining <= 0:
            break
        n = min(batch_size, remaining)
        logger.info(f"Generating batch {i+1}/{batches} ({n} probes, difficulty={difficulty})...")

        probes = generate_batch(difficulty, n)

        # Deduplicate
        new = 0
        for p in probes:
            key = p["question"].lower()
            if key not in seen:
                seen.add(key)
                all_probes.append(p)
                new += 1

        logger.info(f"  Got {len(probes)} probes, {new} new (total: {len(all_probes)})")
        time.sleep(1)

    logger.info(f"Generated {len(all_probes)} {difficulty} probes")
    return all_probes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["medium", "hard", "very_hard"], default="hard")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    probes = generate_all(args.difficulty, args.count, args.batch_size)

    # Save
    output = PROJECT_ROOT / "data" / "probes" / f"llm_probes_{args.difficulty}.json"
    output.write_text(json.dumps(probes, indent=2, ensure_ascii=False))
    logger.info(f"Saved to {output}")

    # Also ingest into store
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.store import ProbeStore
    store = ProbeStore(PROJECT_ROOT / "data" / "pipeline_store.json")
    added = store.add_batch(probes)
    store.save()
    logger.info(f"Added {added} new probes to pipeline store")
