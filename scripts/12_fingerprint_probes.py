#!/usr/bin/env python3
"""Phase 6: Generate extended-phrasing fingerprint probes for distillation detection.

This script selects a subset of T5-T7 probes that are most diagnostic for
knowledge fingerprinting (high inter-model variance), then generates 7
additional phrasings per probe (10 total) to increase fingerprint resolution.

The extended phrasings are:
  1-3. Existing: direct, fill_blank, contextual
  4. Reversed question (ask about entity given the attribute value)
  5. Multiple-choice (correct answer + 3 distractors)
  6. True/false statement
  7. Negated question
  8. Embedded in longer context
  9. Different formality register
  10. Paraphrased with synonyms

Usage:
  python scripts/12_fingerprint_probes.py
  python scripts/12_fingerprint_probes.py --count 100
  python scripts/12_fingerprint_probes.py --select-from-results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api_client import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "fingerprint.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_probe_results():
    """Load all model response files and compute per-probe correctness matrix."""
    responses_dir = PROJECT_ROOT / "data" / "raw_responses"
    model_results = {}

    for f in sorted(responses_dir.glob("*_responses.json")):
        data = json.load(open(f))
        model_name = data["model_name"]
        probe_map = {}
        for r in data["probe_results"]:
            probe_map[r["probe_id"]] = {
                "correct": r["correct"],
                "excluded": r.get("excluded", False),
                "responses": r.get("responses", {}),
            }
        model_results[model_name] = probe_map

    return model_results


def compute_probe_variance(model_results, tiers=("T5", "T6", "T7")):
    """Compute inter-model variance for each probe.

    Returns probes sorted by variance (highest first). High-variance probes
    are the most diagnostic: some models know them, others don't.
    """
    # Collect all probe IDs from target tiers
    probe_ids = set()
    for model_name, probes in model_results.items():
        for pid, result in probes.items():
            if any(pid.startswith(t) for t in tiers) and not result["excluded"]:
                probe_ids.add(pid)

    probe_stats = []
    for pid in sorted(probe_ids):
        correct_count = 0
        total_count = 0
        for model_name, probes in model_results.items():
            if pid in probes and not probes[pid]["excluded"]:
                total_count += 1
                if probes[pid]["correct"]:
                    correct_count += 1

        if total_count < 3:
            continue

        accuracy = correct_count / total_count
        # Bernoulli variance: p(1-p), maximized at p=0.5
        variance = accuracy * (1 - accuracy)

        probe_stats.append({
            "probe_id": pid,
            "tier": pid.split("_")[0],
            "accuracy_across_models": accuracy,
            "variance": variance,
            "models_correct": correct_count,
            "models_total": total_count,
        })

    probe_stats.sort(key=lambda x: x["variance"], reverse=True)
    return probe_stats


def select_fingerprint_probes(probe_stats, count=100):
    """Select the top-N most diagnostic probes, balanced across T5-T7."""
    per_tier_target = count // 3
    selected = []

    for tier in ("T5", "T6", "T7"):
        tier_probes = [p for p in probe_stats if p["tier"] == tier]
        # Take the highest-variance probes, but exclude those at 0% or 100%
        # (no information content)
        tier_probes = [p for p in tier_probes if 0 < p["variance"]]
        selected.extend(tier_probes[:per_tier_target])

    # Fill remaining slots from any tier
    remaining = count - len(selected)
    if remaining > 0:
        selected_ids = {p["probe_id"] for p in selected}
        extras = [p for p in probe_stats if p["probe_id"] not in selected_ids and p["variance"] > 0]
        selected.extend(extras[:remaining])

    logger.info(f"Selected {len(selected)} fingerprint probes:")
    for tier in ("T5", "T6", "T7"):
        n = sum(1 for p in selected if p["tier"] == tier)
        logger.info(f"  {tier}: {n} probes")

    return selected


def load_original_probes(selected_ids):
    """Load the original probe data for selected probe IDs."""
    probes_dir = PROJECT_ROOT / "data" / "probes"
    original_probes = {}

    for f in sorted(probes_dir.glob("*.json")):
        try:
            data = json.load(open(f))
            if isinstance(data, list):
                for probe in data:
                    pid = probe.get("id", "")
                    if pid in selected_ids:
                        original_probes[pid] = probe
        except (json.JSONDecodeError, KeyError):
            continue

    return original_probes


EXTENDED_PHRASING_PROMPT = """You are generating additional phrasings for a factual knowledge probe.
The probe tests whether a language model has memorized a specific fact.

Original probe:
- Direct question: {question_direct}
- Fill-in-the-blank: {question_fill_blank}
- Contextual: {question_contextual}
- Correct answer: {answer}

Generate exactly 7 additional phrasings of this same factual question.
Each phrasing must test the SAME fact but in a different format:

4. REVERSED: Ask about the entity given the attribute value.
   Example: If original asks "What is X's affiliation?", reversed asks "Who at MIT works on distributed systems?" (where MIT is the answer)

5. MULTIPLE_CHOICE: Present the correct answer among 3 plausible but wrong alternatives. Format: "Q? (A) wrong1 (B) correct (C) wrong2 (D) wrong3" — randomize the position of the correct answer.

6. TRUE_FALSE: A statement that is either true or false about this fact. Make it true 50% of the time. Format: "True or false: [statement]"

7. NEGATED: A question that includes a negation. Example: "Which of the following is NOT the capital of France?"

8. EMBEDDED: Embed the question naturally in a longer passage or scenario that provides additional context.

9. CASUAL: Rephrase in a very casual, conversational tone. Example: "Hey, do you happen to know where [person] works?"

10. SYNONYM_PARAPHRASE: Rephrase using different vocabulary and sentence structure while asking the same thing.

Respond in JSON format:
{{
  "reversed": "...",
  "multiple_choice": "...",
  "multiple_choice_answer_letter": "A|B|C|D",
  "true_false": "...",
  "true_false_answer": true|false,
  "negated": "...",
  "embedded": "...",
  "casual": "...",
  "synonym_paraphrase": "..."
}}
"""


def generate_extended_phrasings(client, probe, model="anthropic/claude-sonnet-4"):
    """Generate 7 additional phrasings for a single probe using an LLM."""
    prompt = EXTENDED_PHRASING_PROMPT.format(
        question_direct=probe.get("question_direct", ""),
        question_fill_blank=probe.get("question_fill_blank", ""),
        question_contextual=probe.get("question_contextual", ""),
        answer=probe.get("answer", ""),
    )

    try:
        response = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.error(f"Failed to generate extended phrasings for {probe.get('id')}: {e}")

    return None


def build_fingerprint_dataset(selected_stats, original_probes, extended_phrasings):
    """Combine original probes with extended phrasings into fingerprint dataset."""
    dataset = []

    for stat in selected_stats:
        pid = stat["probe_id"]
        if pid not in original_probes:
            logger.warning(f"Original probe not found: {pid}")
            continue

        orig = original_probes[pid]
        extended = extended_phrasings.get(pid, {})

        entry = {
            "id": pid,
            "fingerprint_id": f"FP_{pid}",
            "tier": stat["tier"],
            "answer": orig.get("answer", ""),
            "answer_type": orig.get("answer_type", "text"),
            "domain": orig.get("domain", ""),
            "region": orig.get("region", ""),
            "source": orig.get("source", ""),
            # Original 3 phrasings
            "question_direct": orig.get("question_direct", ""),
            "question_fill_blank": orig.get("question_fill_blank", ""),
            "question_contextual": orig.get("question_contextual", ""),
            # Extended 7 phrasings
            "question_reversed": extended.get("reversed", ""),
            "question_multiple_choice": extended.get("multiple_choice", ""),
            "multiple_choice_answer_letter": extended.get("multiple_choice_answer_letter", ""),
            "question_true_false": extended.get("true_false", ""),
            "true_false_answer": extended.get("true_false_answer", None),
            "question_negated": extended.get("negated", ""),
            "question_embedded": extended.get("embedded", ""),
            "question_casual": extended.get("casual", ""),
            "question_synonym_paraphrase": extended.get("synonym_paraphrase", ""),
            # Selection metadata
            "inter_model_variance": stat["variance"],
            "accuracy_across_models": stat["accuracy_across_models"],
            "models_correct": stat["models_correct"],
            "models_total": stat["models_total"],
        }
        dataset.append(entry)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate fingerprint probe set for distillation detection"
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of fingerprint probes to select (default: 100)"
    )
    parser.add_argument(
        "--select-from-results", action="store_true",
        help="Select probes based on inter-model variance from existing results"
    )
    parser.add_argument(
        "--generate-phrasings", action="store_true",
        help="Generate extended phrasings using LLM (requires API key)"
    )
    parser.add_argument(
        "--model", type=str, default="anthropic/claude-sonnet-4",
        help="Model to use for phrasing generation"
    )
    args = parser.parse_args()

    logger.info("=== Phase 6: Fingerprint Probe Generation ===")

    output_dir = PROJECT_ROOT / "data" / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Select most diagnostic probes
    if args.select_from_results:
        logger.info("Loading existing model results for probe selection...")
        model_results = load_probe_results()

        if not model_results:
            logger.error("No model results found. Run calibration (02_run_calibration.py) first.")
            sys.exit(1)

        logger.info(f"Loaded results for {len(model_results)} models")

        probe_stats = compute_probe_variance(model_results)
        logger.info(f"Computed variance for {len(probe_stats)} T5-T7 probes")

        selected = select_fingerprint_probes(probe_stats, count=args.count)

        # Save selection metadata
        selection_file = output_dir / "fingerprint_selection.json"
        json.dump(selected, open(selection_file, "w"), indent=2)
        logger.info(f"Saved probe selection to {selection_file}")
    else:
        # Load existing selection
        selection_file = output_dir / "fingerprint_selection.json"
        if selection_file.exists():
            selected = json.load(open(selection_file))
            logger.info(f"Loaded existing selection: {len(selected)} probes")
        else:
            logger.error("No selection found. Run with --select-from-results first.")
            sys.exit(1)

    # Step 2: Load original probe data
    selected_ids = {s["probe_id"] for s in selected}
    original_probes = load_original_probes(selected_ids)
    logger.info(f"Loaded {len(original_probes)}/{len(selected_ids)} original probes")

    # Step 3: Generate extended phrasings
    extended_phrasings = {}
    extended_cache_file = output_dir / "fingerprint_extended_phrasings.json"

    if args.generate_phrasings:
        logger.info("Generating extended phrasings via LLM...")
        client = OpenRouterClient()

        # Load cache if exists
        if extended_cache_file.exists():
            extended_phrasings = json.load(open(extended_cache_file))
            logger.info(f"Loaded {len(extended_phrasings)} cached phrasings")

        for i, stat in enumerate(selected):
            pid = stat["probe_id"]
            if pid in extended_phrasings:
                continue
            if pid not in original_probes:
                continue

            logger.info(f"  [{i+1}/{len(selected)}] Generating phrasings for {pid}")
            result = generate_extended_phrasings(
                client, original_probes[pid], model=args.model
            )
            if result:
                extended_phrasings[pid] = result
                # Save incrementally
                json.dump(extended_phrasings, open(extended_cache_file, "w"), indent=2)

        logger.info(f"Generated phrasings for {len(extended_phrasings)} probes")
    else:
        if extended_cache_file.exists():
            extended_phrasings = json.load(open(extended_cache_file))

    # Step 4: Build fingerprint dataset
    dataset = build_fingerprint_dataset(selected, original_probes, extended_phrasings)

    output_file = output_dir / "fingerprint_probes.json"
    json.dump(dataset, open(output_file, "w"), indent=2)
    logger.info(f"Saved fingerprint dataset: {len(dataset)} probes to {output_file}")

    # Summary
    logger.info("\n=== Fingerprint Dataset Summary ===")
    for tier in ("T5", "T6", "T7"):
        tier_probes = [p for p in dataset if p["tier"] == tier]
        has_extended = sum(1 for p in tier_probes if p.get("question_reversed"))
        logger.info(f"  {tier}: {len(tier_probes)} probes ({has_extended} with extended phrasings)")


if __name__ == "__main__":
    main()
