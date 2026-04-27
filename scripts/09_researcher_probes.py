#!/usr/bin/env python3
"""Researcher Knowledge Probes: the motivating case study.

Tests whether frontier LLMs have memorized individual researchers
"inside the weights." This is the observation that inspired the IKP paper:

  Bojie Li noticed that frontier models know him, his peers, and their
  publications — but not less prominent researchers. The depth of this
  knowledge correlates with model capacity: GPT-5.4 < Claude Sonnet
  < Claude Opus < Gemini 3.1 Pro.

This script generates probes about real CS systems researchers at varying
prominence levels and tests how model knowledge depth tracks with model size.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from scorer import score_response, is_refusal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "researcher_probes.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def generate_researcher_probes(client, model):
    """Generate probes about CS systems researchers at varying prominence levels."""

    prompt = """You are helping create a research probe dataset about computer science researchers,
specifically in systems, networking, and data center research.

Generate probes about REAL researchers at three prominence levels:

## Level 1: Well-known researchers (expect >70B models to know them)
These are senior faculty at top universities or research labs, with >5000 citations,
multiple best paper awards, and well-known systems contributions.
Generate 20 probes asking about: their affiliation, their most cited paper,
their specific research area, their PhD advisor, awards they've received.

## Level 2: Established researchers (expect >200B models to know them)
These are mid-career researchers or senior PhD students with 500-5000 citations,
publications at top venues (SIGCOMM, OSDI, SOSP, NSDI, SIGMOD, etc.),
known within their subfield but not household names.
Generate 20 probes. Include researchers from varied backgrounds and institutions.

## Level 3: Emerging researchers (expect only the largest models to know them)
These are junior faculty, postdocs, or recent PhD graduates with <500 citations,
published at top venues but not yet widely known.
Generate 10 probes.

For each probe, provide:
```json
{
  "id": "researcher_L1_001",
  "level": 1,
  "researcher_name": "Full Name",
  "question_direct": "What university is [Name] affiliated with?",
  "question_fill_blank": "[Name] is a professor at ___.",
  "question_contextual": "In the field of [area], [Name] works at ___.",
  "answer": "MIT",
  "answer_type": "text",
  "fact_type": "affiliation|paper|research_area|advisor|award",
  "subfield": "systems|networking|databases|architecture|security"
}
```

CRITICAL: Only include REAL researchers with verifiable facts.
Include researchers from diverse backgrounds, institutions, and countries.
Include both English and Chinese-name researchers working in these fields.

Return ONLY a JSON array."""

    messages = [
        {"role": "system", "content": "You are a research assistant generating factual probes about real computer science researchers. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    logger.info("Generating researcher probes...")
    text = client.get_response_text(model=model, messages=messages, temperature=0.3, max_tokens=16000)

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
        if text.startswith("json"):
            text = text[4:]

    try:
        probes = json.loads(text)
        logger.info(f"Generated {len(probes)} researcher probes")
        return probes
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse: {e}")
        # Try to salvage
        start = text.find("[")
        if start >= 0:
            bracket_count = 0
            for end_idx in range(start, len(text)):
                if text[end_idx] == "[": bracket_count += 1
                elif text[end_idx] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            probes = json.loads(text[start:end_idx+1])
                            logger.info(f"Salvaged {len(probes)} probes")
                            return probes
                        except:
                            pass
        return []


def run_researcher_probes(client, model_name, model_id, probes):
    """Run researcher probes and collect detailed response analysis."""
    results = []

    for probe in probes:
        # Ask the direct question
        messages = [
            {"role": "system", "content": "Answer factual questions about computer science researchers. Be direct and concise. If you don't know, say 'I don't know'."},
            {"role": "user", "content": probe.get("question_direct", "")},
        ]

        try:
            response = client.get_response_text(model=model_id, messages=messages, temperature=0, max_tokens=200)
        except Exception as e:
            response = f"[ERROR: {e}]"

        correct = score_response(response, probe.get("answer", ""), probe.get("answer_type", "auto"))
        refusal = is_refusal(response)

        results.append({
            "probe_id": probe.get("id", "?"),
            "level": probe.get("level", 0),
            "researcher": probe.get("researcher_name", "?"),
            "fact_type": probe.get("fact_type", "?"),
            "question": probe.get("question_direct", ""),
            "gold_answer": probe.get("answer", ""),
            "response": response[:300],
            "correct": correct,
            "refusal": refusal,
        })

    return results


def analyze_results(all_model_results):
    """Analyze researcher probe results across models."""
    print("\n" + "=" * 100)
    print("  RESEARCHER KNOWLEDGE PROBE RESULTS")
    print("  (The observation that inspired the IKP paper)")
    print("=" * 100)

    # Per-model, per-level accuracy
    print(f"\n  {'Model':30s} {'Level 1':>10s} {'Level 2':>10s} {'Level 3':>10s} {'Overall':>10s}")
    print(f"  {'':30s} {'(well-known)':>10s} {'(established)':>10s} {'(emerging)':>10s}")
    print(f"  {'─'*75}")

    for model_name, results in sorted(all_model_results.items()):
        level_stats = {}
        for r in results:
            lev = r["level"]
            if lev not in level_stats:
                level_stats[lev] = {"correct": 0, "total": 0, "refusal": 0}
            if r["refusal"]:
                level_stats[lev]["refusal"] += 1
            else:
                level_stats[lev]["total"] += 1
                if r["correct"]:
                    level_stats[lev]["correct"] += 1

        parts = []
        for lev in [1, 2, 3]:
            s = level_stats.get(lev, {"correct": 0, "total": 0})
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            parts.append(f"{acc:10.1%}")

        total_c = sum(s["correct"] for s in level_stats.values())
        total_t = sum(s["total"] for s in level_stats.values())
        overall = total_c / total_t if total_t > 0 else 0

        print(f"  {model_name:30s} {parts[0]} {parts[1]} {parts[2]} {overall:10.1%}")

    # Show which specific researchers are known by which models
    print(f"\n  KNOWLEDGE DEPTH SPECTRUM:")
    print(f"  (Which researchers does each model recognize?)")
    print(f"  {'─'*80}")

    # Collect per-researcher recognition across models
    researcher_known_by = {}
    for model_name, results in all_model_results.items():
        for r in results:
            rname = r["researcher"]
            if rname not in researcher_known_by:
                researcher_known_by[rname] = {"level": r["level"], "known_by": [], "unknown_to": []}
            if r["correct"]:
                researcher_known_by[rname]["known_by"].append(model_name)
            elif not r["refusal"]:
                researcher_known_by[rname]["unknown_to"].append(model_name)

    for level in [1, 2, 3]:
        researchers = [(name, info) for name, info in researcher_known_by.items() if info["level"] == level]
        if researchers:
            print(f"\n  Level {level} researchers:")
            for name, info in sorted(researchers, key=lambda x: -len(x[1]["known_by"])):
                n_known = len(info["known_by"])
                n_total = n_known + len(info["unknown_to"])
                models_str = ", ".join(info["known_by"][:5])
                if len(info["known_by"]) > 5:
                    models_str += f" +{len(info['known_by'])-5}"
                print(f"    {name:30s}: {n_known}/{n_total} models know this ({models_str})")


def main():
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))["models"]
    client = OpenRouterClient(requests_per_minute=50, max_retries=5, timeout=120)

    # Step 1: Generate probes
    probes_file = PROJECT_ROOT / "data" / "probes" / "researcher_probes.json"
    if probes_file.exists():
        probes = json.load(open(probes_file))
        logger.info(f"Loaded {len(probes)} existing researcher probes")
    else:
        probes = generate_researcher_probes(client, "anthropic/claude-sonnet-4")
        with open(probes_file, "w") as f:
            json.dump(probes, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(probes)} researcher probes")

    # Step 2: Test models — focus on the spectrum Bojie observed
    test_models = [
        # Small open-source (should know very few researchers)
        "llama-3.2-3b",
        "qwen-2.5-7b",
        # Medium (should know well-known researchers)
        "phi-4",
        "qwen-2.5-72b",
        "mistral-large",
        # Large open (should know established researchers)
        "hermes-3-405b",
        "deepseek-v3",
        # Frontier proprietary (should know most researchers)
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4o",
        "claude-sonnet-4",
        "claude-opus-4",
        "gemini-2.5-pro",
        "gemini-3.1-pro",
    ]

    output_dir = PROJECT_ROOT / "data" / "researcher_responses"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in test_models:
        if model_name not in config:
            continue

        results_file = output_dir / f"{model_name}_researcher.json"
        if results_file.exists():
            existing = json.load(open(results_file))
            all_results[model_name] = existing
            logger.info(f"Loaded existing results for {model_name}")
            continue

        model_info = config[model_name]
        logger.info(f"\n=== {model_name} ({model_info['vendor']}) — researcher probes ===")

        try:
            results = run_researcher_probes(client, model_name, model_info["id"], probes)
            all_results[model_name] = results

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Quick summary
            correct = sum(1 for r in results if r["correct"])
            total = sum(1 for r in results if not r["refusal"])
            logger.info(f"  {model_name}: {correct}/{total} correct ({correct/total:.1%})" if total > 0 else f"  {model_name}: all refusals")

        except Exception as e:
            logger.error(f"  FAILED {model_name}: {e}")

    # Step 3: Analysis
    if all_results:
        analyze_results(all_results)


if __name__ == "__main__":
    main()
