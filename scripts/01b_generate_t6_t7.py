#!/usr/bin/env python3
"""Generate T6 (long-tail) and T7 (extreme long-tail) probe candidates.

T6: LLM generates very obscure facts, we attempt verification.
T7: Constructed from known-obscure factual data sources (small municipalities,
    niche databases, minor officials, etc.)

For the pilot study, we generate candidates via LLM with explicit obscurity
instructions, since full web-frequency verification would require Common Crawl
access. The calibration filtering step will validate discrimination power
empirically.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from probe_generator import validate_probe_format, save_probes, load_probes, REGIONS, DOMAINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


T6_PROMPT_TEMPLATE = """You are generating extremely obscure factual knowledge probes for a research study.

## Task
Generate exactly {batch_size} factual questions about VERY obscure, real-world facts.

## Requirements — CRITICAL
These must be facts that:
1. ARE REAL and verifiable (not made up)
2. Appear on only 2-10 web pages worldwide
3. Are NOT derivable by reasoning — they must be memorized
4. Have a single, unambiguous, concise answer

## Types of obscure facts to generate (emphasize region: {region}, domain: {domain})
- Population of a very small town (<2000 people) in a specific country
- The mayor or council leader of a small municipality
- The founding year of an obscure local institution (small museum, community center)
- A specific building's height in a non-major city
- An obscure academic: their specific university department and research focus
- A minor diplomatic event between two small countries (exact date)
- The specific area (in km²) of a small administrative district
- A niche product's specific technical specification
- The chairperson of a minor professional association in a small country
- Author(s) of a paper with <20 citations published in a minor workshop/venue

## Output Format
Return a JSON array. Each element:
```json
{{
  "id": "T6_b{batch_num}_001",
  "question_direct": "What is ...?",
  "question_fill_blank": "The ... is ___.",
  "question_contextual": "In ..., the ... is ___.",
  "answer": "...",
  "answer_type": "text|numeric",
  "domain": "people|places|publications|measurements|events|organizations",
  "region": "North America|Europe|East Asia|South Asia|Middle East|Africa|Latin America|Oceania",
  "estimated_web_frequency": "2-10",
  "source_description": "description of where this can be verified"
}}
```

CRITICAL: These must be REAL facts you are confident about. Do NOT hallucinate or guess.
If you cannot think of enough real obscure facts, generate fewer rather than making things up.

Generate exactly {batch_size} probes. Return ONLY the JSON array."""


T7_PROMPT_TEMPLATE = """You are generating ceiling-level factual knowledge probes for a research study.

## Task
Generate exactly {batch_size} factual questions about the MOST obscure real-world facts you can think of.

## Requirements — CRITICAL
These facts should:
1. BE ABSOLUTELY REAL (not invented)
2. Appear on essentially 1-2 web pages worldwide
3. Be the kind of detail that only someone personally familiar with the topic would know
4. Have a single verifiable answer

## Types of extremely obscure facts (emphasize region: {region}, domain: {domain})
- The exact population of a tiny village (<500 people)
- A specific detail about an obscure individual (their exact job title at a small institution)
- The founding year of a very small local organization
- The exact area or elevation of a very minor geographic feature
- A specific resolution number from a minor local government
- The name of the principal of a specific small school
- The exact date a minor local regulation was enacted
- A specific technical detail about a niche industrial product
- The name of the editor of a very minor regional journal
- The exact distance between two minor landmarks

## Output Format
Return a JSON array. Each element:
```json
{{
  "id": "T7_b{batch_num}_001",
  "question_direct": "What is ...?",
  "question_fill_blank": "The ... is ___.",
  "question_contextual": "In ..., the ... is ___.",
  "answer": "...",
  "answer_type": "text|numeric",
  "domain": "people|places|publications|measurements|events|organizations",
  "region": "North America|Europe|East Asia|South Asia|Middle East|Africa|Latin America|Oceania",
  "estimated_web_frequency": "1",
  "source_description": "description of where this can be verified"
}}
```

CRITICAL: Only generate facts you are CONFIDENT are real. Quality over quantity.
If you can only think of {fewer} real facts this obscure, that is fine.

Return ONLY the JSON array."""


def generate_t6_probes(client, model, target=400, batch_size=15):
    """Generate T6 candidate probes."""
    all_probes = []
    num_batches = (target + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        region = REGIONS[batch_num % len(REGIONS)]
        domain = DOMAINS[batch_num % len(DOMAINS)]
        prompt = T6_PROMPT_TEMPLATE.format(
            batch_size=batch_size, batch_num=batch_num,
            region=region, domain=domain,
        )
        messages = [
            {"role": "system", "content": "You are generating obscure factual knowledge probes. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"T6 batch {batch_num + 1}/{num_batches} (region={region}, domain={domain})")
        text = client.get_response_text(model=model, messages=messages, temperature=0.7, max_tokens=16000)

        probes = _parse_json_response(text, f"T6_b{batch_num}")
        for p in probes:
            p["tier"] = "T6"
        all_probes.extend(probes)
        logger.info(f"  Got {len(probes)} probes")

    return all_probes


def generate_t7_probes(client, model, target=400, batch_size=15):
    """Generate T7 candidate probes."""
    all_probes = []
    num_batches = (target + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        region = REGIONS[batch_num % len(REGIONS)]
        domain = DOMAINS[batch_num % len(DOMAINS)]
        prompt = T7_PROMPT_TEMPLATE.format(
            batch_size=batch_size, batch_num=batch_num,
            region=region, domain=domain,
            fewer=batch_size // 2,
        )
        messages = [
            {"role": "system", "content": "You are generating extremely obscure factual knowledge probes. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"T7 batch {batch_num + 1}/{num_batches} (region={region}, domain={domain})")
        text = client.get_response_text(model=model, messages=messages, temperature=0.8, max_tokens=16000)

        probes = _parse_json_response(text, f"T7_b{batch_num}")
        for p in probes:
            p["tier"] = "T7"
        all_probes.extend(probes)
        logger.info(f"  Got {len(probes)} probes")

    return all_probes


def _parse_json_response(text: str, id_prefix: str) -> list:
    """Parse JSON array from LLM response, handling common formatting issues."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines) - 1
        for i, line in enumerate(lines):
            if line.strip().startswith("```") and i > 0:
                end = i
                break
        text = "\n".join(lines[start:end])
        if text.startswith("json"):
            text = text[4:]

    try:
        probes = json.loads(text)
        if isinstance(probes, list):
            for i, p in enumerate(probes):
                p["id"] = f"{id_prefix}_{i:03d}"
            return probes
    except json.JSONDecodeError:
        pass

    # Try to salvage
    try:
        start = text.find("[")
        if start >= 0:
            bracket_count = 0
            for end_idx in range(start, len(text)):
                if text[end_idx] == "[":
                    bracket_count += 1
                elif text[end_idx] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        probes = json.loads(text[start:end_idx + 1])
                        for i, p in enumerate(probes):
                            p["id"] = f"{id_prefix}_{i:03d}"
                        return probes
    except Exception:
        pass

    logger.warning(f"Could not parse JSON for {id_prefix}")
    return []


def main():
    with open(PROJECT_ROOT / "configs" / "experiment.json") as f:
        config = json.load(f)

    client = OpenRouterClient(
        requests_per_minute=config["api"]["requests_per_minute"],
        max_retries=config["api"]["max_retries"],
        timeout=120,
    )

    model = config["probe_generation"]["generator_model"]

    # T6
    existing_t6 = load_probes("T6", "candidates")
    if len(existing_t6) < 320:
        logger.info("=== Generating T6 candidates ===")
        t6_probes = generate_t6_probes(client, model, target=400, batch_size=15)
        valid_t6 = [p for p in t6_probes if validate_probe_format(p)[0]]
        save_probes(valid_t6, "T6", "candidates")
        logger.info(f"T6: {len(valid_t6)} valid candidates saved")
    else:
        logger.info(f"T6: already have {len(existing_t6)} candidates, skipping")

    # T7
    existing_t7 = load_probes("T7", "candidates")
    if len(existing_t7) < 320:
        logger.info("=== Generating T7 candidates ===")
        t7_probes = generate_t7_probes(client, model, target=400, batch_size=15)
        valid_t7 = [p for p in t7_probes if validate_probe_format(p)[0]]
        save_probes(valid_t7, "T7", "candidates")
        logger.info(f"T7: {len(valid_t7)} valid candidates saved")
    else:
        logger.info(f"T7: already have {len(existing_t7)} candidates, skipping")

    logger.info(f"\nAPI stats: {json.dumps(client.get_stats(), indent=2)}")


if __name__ == "__main__":
    main()
