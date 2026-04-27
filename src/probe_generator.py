"""Generate IKP probe candidates using LLMs via OpenRouter.

Generates 400 candidates per tier (T1-T5) with 3 phrasings each.
T6-T7 require separate handling (web search verification / human experts).
"""

import json
import logging
import random
import sys
from pathlib import Path

from api_client import OpenRouterClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

TIER_DEFINITIONS = {
    "T1": {
        "name": "Universal Knowledge",
        "param_range": "0.1B-1B",
        "web_freq": ">100K documents in Common Crawl",
        "description": "Facts any literate person knows; extremely high frequency in web corpora.",
        "examples": [
            "What is the capital of France? (Answer: Paris)",
            "At what temperature does water boil at sea level in Celsius? (Answer: 100)",
            "What planet is closest to the Sun? (Answer: Mercury)",
        ],
        "difficulty": "Even the smallest language models (>1B params) should get these right. Models <0.5B may fail.",
    },
    "T2": {
        "name": "Common Reference Knowledge",
        "param_range": "1B-7B",
        "web_freq": "10K-100K documents",
        "description": "Facts a well-read person knows; high frequency but more specific.",
        "examples": [
            "What is the height of the Eiffel Tower in meters? (Answer: 330)",
            "Who won the Nobel Prize in Physics in 2017? (Answer: Rainer Weiss, Barry Barish, Kip Thorne)",
            "What is the population of Portugal? (Answer: ~10.3 million)",
        ],
        "difficulty": "7B models score well; 1B models start failing.",
    },
    "T3": {
        "name": "Domain-Specific Knowledge",
        "param_range": "7B-70B",
        "web_freq": "1K-10K documents",
        "description": "Facts a domain specialist would know; moderate frequency in corpora.",
        "examples": [
            "What year was the Treaty of Shimonoseki signed? (Answer: 1895)",
            "Who is the author of the paper 'Attention Is All You Need'? (Answer: Ashish Vaswani et al.)",
            "What is the elevation of Bogotá, Colombia in meters? (Answer: ~2,640)",
        ],
        "difficulty": "70B models score well; 7B models struggle.",
    },
    "T4": {
        "name": "Obscure Knowledge",
        "param_range": "70B-300B",
        "web_freq": "100-1K documents",
        "description": "Facts that exist on the web but are rarely mentioned.",
        "examples": [
            "What is the population of Vanuatu? (Answer: ~320,000)",
            "Who was the first president of Botswana? (Answer: Seretse Khama)",
            "What is the height of the Turning Torso building in Malmö, Sweden? (Answer: 190 meters)",
        ],
        "difficulty": "300B models score moderately; 70B models mostly fail.",
    },
    "T5": {
        "name": "Deep Knowledge",
        "param_range": "300B-1T",
        "web_freq": "10-100 documents",
        "description": "Facts that only very large or frontier-scale models can retrieve with confidence.",
        "examples": [
            "What is the population of Nauru? (Answer: ~12,500)",
            "Who wrote the 2019 paper 'Language Models are Few-Shot Learners'? (Answer: Tom Brown et al.)",
            "What is the exact height of the Abraj Al-Bait Clock Tower? (Answer: 601 meters)",
        ],
        "difficulty": "Only the largest frontier models (>300B) score well; 70B models fail almost entirely.",
    },
    "T6": {
        "name": "Long-Tail Knowledge",
        "param_range": "1T-5T",
        "web_freq": "2-10 documents",
        "description": "Facts at the far end of what the largest models can retrieve; mentioned very rarely online.",
        "examples": [
            "A very obscure researcher's specific conference paper from a minor workshop",
            "A local politician's committee membership in a small country",
            "A niche product's exact revision history",
        ],
        "difficulty": "Only the very largest closed-source models show meaningful accuracy; models under 1T score near zero.",
    },
    "T7": {
        "name": "Extreme Long-Tail (ceiling probes)",
        "param_range": ">5T (beyond current models)",
        "web_freq": "~1 document",
        "description": "Facts that exist on essentially a single page — below the memorization threshold of any current model.",
        "examples": [
            "An individual known only from their personal homepage",
            "A data point buried in a single municipal PDF",
            "A specific line item in a single government report",
        ],
        "difficulty": "All current models should score near zero. Used as ceiling/control.",
    },
}

REGIONS = [
    "North America", "Europe", "East Asia", "South Asia",
    "Middle East", "Africa", "Latin America", "Oceania",
]

DOMAINS = [
    "people", "places", "publications",
    "measurements", "events", "organizations",
]


def build_generation_prompt(tier: str, batch_num: int, batch_size: int = 50) -> str:
    """Build a structured prompt to generate probe candidates for a given tier."""
    tier_def = TIER_DEFINITIONS[tier]

    # Rotate region/domain emphasis per batch to ensure balance
    region_idx = batch_num % len(REGIONS)
    domain_idx = batch_num % len(DOMAINS)
    primary_region = REGIONS[region_idx]
    primary_domain = DOMAINS[domain_idx]

    prompt = f"""You are generating factual knowledge probes for a research study on LLM knowledge capacity.

## Task
Generate exactly {batch_size} factual questions for **{tier}: {tier_def['name']}**.

## Tier Definition
- Parameter range this tier discriminates: {tier_def['param_range']}
- Approximate web frequency of these facts: {tier_def['web_freq']}
- Description: {tier_def['description']}
- Difficulty: {tier_def['difficulty']}

## Examples of this tier's difficulty level
{chr(10).join(f"- {ex}" for ex in tier_def['examples'])}

## Requirements
1. Each question must have a **single, objectively verifiable answer** (not opinion, not ambiguous)
2. Facts must be **incompressible** — the answer cannot be derived by reasoning from other facts. It must be memorized. For instance, "What is 15 squared?" is compressible (225 can be computed). "What is the population of Fiji?" is incompressible.
3. Facts must be **real and accurate** — do not invent facts
4. This batch should emphasize (but not be limited to):
   - Region: **{primary_region}** (at least 40% of questions)
   - Domain: **{primary_domain}** (at least 40% of questions)
   - Remaining questions should cover other regions and domains
5. Answers should be concise (a name, a number, a short phrase)

## Output Format
Return a JSON array. Each element must be:
```json
{{
  "id": "{tier}_batch{batch_num}_001",
  "question_direct": "What is ...?",
  "question_fill_blank": "The ... is ___.",
  "question_contextual": "Among ..., the ... is ___.",
  "answer": "...",
  "answer_type": "text|numeric",
  "domain": "people|places|publications|measurements|events|organizations",
  "region": "North America|Europe|East Asia|South Asia|Middle East|Africa|Latin America|Oceania",
  "estimated_web_frequency": "approximate number of web pages containing this fact",
  "source_description": "Brief description of where this fact can be verified"
}}
```

Generate exactly {batch_size} probes. Return ONLY the JSON array, no other text."""

    return prompt


def generate_tier_probes(
    client: OpenRouterClient,
    tier: str,
    generator_model: str,
    candidates_target: int = 400,
    batch_size: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Generate probe candidates for a single tier."""
    random.seed(seed)
    all_probes = []
    num_batches = (candidates_target + batch_size - 1) // batch_size

    logger.info(f"Generating {candidates_target} candidates for {tier} in {num_batches} batches")

    for batch_num in range(num_batches):
        prompt = build_generation_prompt(tier, batch_num, batch_size)
        messages = [
            {"role": "system", "content": "You are a research assistant generating factual knowledge probes. Always return valid JSON."},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"  {tier} batch {batch_num + 1}/{num_batches}...")
        response_text = client.get_response_text(
            model=generator_model,
            messages=messages,
            temperature=0.7,  # Some diversity in generation
            max_tokens=16000,
        )

        # Parse JSON from response
        try:
            # Try to find JSON array in the response
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code fences
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            probes = json.loads(text)
            if isinstance(probes, list):
                # Re-ID probes to ensure uniqueness
                for i, probe in enumerate(probes):
                    probe["id"] = f"{tier}_b{batch_num}_{i:03d}"
                    probe["tier"] = tier
                all_probes.extend(probes)
                logger.info(f"    Got {len(probes)} probes from batch {batch_num + 1}")
            else:
                logger.warning(f"    Batch {batch_num + 1} returned non-list JSON, skipping")
        except json.JSONDecodeError as e:
            logger.warning(f"    Failed to parse JSON from batch {batch_num + 1}: {e}")
            # Try to salvage partial JSON
            try:
                # Find the first [ and try to parse from there
                start = text.find("[")
                if start >= 0:
                    # Find matching ]
                    bracket_count = 0
                    for end_idx in range(start, len(text)):
                        if text[end_idx] == "[":
                            bracket_count += 1
                        elif text[end_idx] == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                probes = json.loads(text[start:end_idx + 1])
                                for i, probe in enumerate(probes):
                                    probe["id"] = f"{tier}_b{batch_num}_{i:03d}"
                                    probe["tier"] = tier
                                all_probes.extend(probes)
                                logger.info(f"    Salvaged {len(probes)} probes from batch {batch_num + 1}")
                                break
            except Exception:
                logger.warning(f"    Could not salvage any probes from batch {batch_num + 1}")

    logger.info(f"Generated {len(all_probes)} total candidates for {tier}")
    return all_probes


def validate_probe_format(probe: dict) -> tuple[bool, str]:
    """Validate that a probe has all required fields."""
    required = ["id", "question_direct", "question_fill_blank", "question_contextual",
                 "answer", "answer_type", "domain", "region", "tier"]
    for field in required:
        if field not in probe:
            return False, f"Missing field: {field}"
    if probe["answer_type"] not in ("text", "numeric"):
        return False, f"Invalid answer_type: {probe['answer_type']}"
    if probe["domain"] not in DOMAINS:
        return False, f"Invalid domain: {probe['domain']}"
    if probe["region"] not in REGIONS:
        return False, f"Invalid region: {probe['region']}"
    return True, "ok"


def save_probes(probes: list[dict], tier: str, stage: str = "candidates"):
    """Save probes to a JSON file."""
    output_dir = PROJECT_ROOT / "data" / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{tier}_{stage}.json"
    with open(output_file, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(probes)} probes to {output_file}")
    return output_file


def load_probes(tier: str, stage: str = "candidates") -> list[dict]:
    """Load probes from a JSON file."""
    input_file = PROJECT_ROOT / "data" / "probes" / f"{tier}_{stage}.json"
    if not input_file.exists():
        return []
    with open(input_file) as f:
        return json.load(f)


def main():
    """Generate probe candidates for all tiers T1-T5."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(PROJECT_ROOT / "configs" / "experiment.json") as f:
        config = json.load(f)

    client = OpenRouterClient(
        requests_per_minute=config["api"]["requests_per_minute"],
        max_retries=config["api"]["max_retries"],
        retry_delay=config["api"]["retry_delay_seconds"],
        timeout=config["api"]["timeout_seconds"],
    )

    generator_model = config["probe_generation"]["generator_model"]
    candidates_per_tier = config["probe_generation"]["candidates_per_tier"]

    # Determine which tiers to generate (skip if already done)
    tiers_to_generate = []
    for tier in ["T1", "T2", "T3", "T4", "T5"]:
        existing = load_probes(tier, "candidates")
        if len(existing) >= candidates_per_tier * 0.8:
            logger.info(f"Skipping {tier}: already have {len(existing)} candidates")
        else:
            tiers_to_generate.append(tier)

    if not tiers_to_generate:
        logger.info("All T1-T5 tiers already have sufficient candidates")
        return

    for tier in tiers_to_generate:
        probes = generate_tier_probes(
            client=client,
            tier=tier,
            generator_model=generator_model,
            candidates_target=candidates_per_tier,
            seed=config.get("random_seed", 42),
        )

        # Validate and filter
        valid_probes = []
        for probe in probes:
            ok, msg = validate_probe_format(probe)
            if ok:
                valid_probes.append(probe)
            else:
                logger.debug(f"Dropping invalid probe {probe.get('id', '?')}: {msg}")

        save_probes(valid_probes, tier, "candidates")
        logger.info(f"{tier}: {len(valid_probes)}/{len(probes)} valid candidates saved")

    stats = client.get_stats()
    logger.info(f"API stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
