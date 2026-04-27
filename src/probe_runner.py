"""Run IKP probes against models and record raw responses.

This is the core experiment runner. It sends each probe's 3 phrasings to a model,
records the raw responses, and saves structured results for later scoring.

Designed for reproducibility:
- Deterministic (temperature=0)
- Cached (same query won't hit API twice)
- Resumable (checks for existing results before running)
- Logged (full audit trail)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from api_client import OpenRouterClient
from scorer import score_probe, is_refusal

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def build_probe_message(phrasing: str, phrasing_type: str) -> list[dict]:
    """Build the chat messages for a single probe phrasing."""
    system_msg = (
        "You are answering factual knowledge questions. "
        "Give a direct, concise answer. "
        "If the question asks for a number, give just the number. "
        "If it asks for a name, give just the name. "
        "Do not explain or hedge — just answer directly. "
        "If you genuinely do not know, say 'I don't know'."
    )

    if phrasing_type == "fill_blank":
        user_msg = f"Complete the following with the correct answer:\n\n{phrasing}"
    elif phrasing_type == "contextual":
        user_msg = f"Fill in the blank with the correct factual answer:\n\n{phrasing}"
    else:
        user_msg = phrasing

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def run_single_probe(
    client: OpenRouterClient,
    model_id: str,
    probe: dict,
    max_tokens: int = 150,
) -> dict:
    """Run a single probe (all 3 phrasings) against a model.

    Returns a result dict with raw responses and scoring.
    """
    phrasings = [
        ("direct", probe["question_direct"]),
        ("fill_blank", probe["question_fill_blank"]),
        ("contextual", probe["question_contextual"]),
    ]

    responses = {}
    for phrasing_type, phrasing_text in phrasings:
        messages = build_probe_message(phrasing_text, phrasing_type)
        try:
            response_text = client.get_response_text(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            responses[phrasing_type] = {
                "text": response_text,
                "is_refusal": is_refusal(response_text),
            }
        except Exception as e:
            logger.warning(f"Error on probe {probe['id']}, phrasing {phrasing_type}: {e}")
            responses[phrasing_type] = {
                "text": None,
                "error": str(e),
                "is_refusal": True,
            }

    # Score: best of 3 phrasings
    response_texts = [
        responses[pt]["text"]
        for pt in ["direct", "fill_blank", "contextual"]
        if responses[pt]["text"] is not None
    ]
    correct = score_probe(response_texts, probe["answer"], probe.get("answer_type", "auto"))

    # Check if all phrasings were refusals
    all_refusal = all(responses[pt].get("is_refusal", True) for pt in responses)

    return {
        "probe_id": probe["id"],
        "tier": probe["tier"],
        "gold_answer": probe["answer"],
        "answer_type": probe.get("answer_type", "auto"),
        "responses": responses,
        "correct": correct,
        "all_refusal": all_refusal,
        "excluded": all_refusal,  # Excluded from scoring if all refusals
    }


def run_model_probes(
    client: OpenRouterClient,
    model_name: str,
    model_id: str,
    probes: list[dict],
    output_dir: Path,
    max_tokens: int = 150,
) -> dict:
    """Run all probes against a single model. Resumable.

    Returns a summary dict with per-tier accuracies.
    """
    output_file = output_dir / f"{model_name}_responses.json"
    results_file = output_dir / f"{model_name}_results.json"

    # Load existing results for resumability
    existing_results = {}
    if output_file.exists():
        with open(output_file) as f:
            existing_data = json.load(f)
            for r in existing_data.get("probe_results", []):
                existing_results[r["probe_id"]] = r
        logger.info(f"Loaded {len(existing_results)} existing results for {model_name}")

    probe_results = []
    new_count = 0

    for i, probe in enumerate(probes):
        if probe["id"] in existing_results:
            probe_results.append(existing_results[probe["id"]])
            continue

        result = run_single_probe(client, model_id, probe, max_tokens)
        probe_results.append(result)
        new_count += 1

        # Save incrementally every 50 probes
        if new_count % 50 == 0:
            _save_raw_results(output_file, model_name, model_id, probe_results)
            logger.info(f"  {model_name}: {i + 1}/{len(probes)} probes done ({new_count} new)")

    # Final save
    _save_raw_results(output_file, model_name, model_id, probe_results)

    # Compute summary
    summary = compute_summary(model_name, model_id, probe_results)
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"  {model_name}: complete. "
        f"Aggregate={summary['aggregate_accuracy']:.3f}, "
        f"Per-tier={[f'{t}={v:.3f}' for t, v in sorted(summary['per_tier_accuracy'].items())]}"
    )

    return summary


def _save_raw_results(output_file: Path, model_name: str, model_id: str, probe_results: list):
    data = {
        "model_name": model_name,
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe_results": probe_results,
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_summary(model_name: str, model_id: str, probe_results: list[dict]) -> dict:
    """Compute per-tier and aggregate accuracy from probe results."""
    tier_correct = {}
    tier_total = {}
    tier_excluded = {}

    for r in probe_results:
        tier = r["tier"]
        if tier not in tier_correct:
            tier_correct[tier] = 0
            tier_total[tier] = 0
            tier_excluded[tier] = 0

        if r.get("excluded", False):
            tier_excluded[tier] += 1
            continue

        tier_total[tier] += 1
        if r["correct"]:
            tier_correct[tier] += 1

    per_tier_accuracy = {}
    for tier in sorted(tier_correct.keys()):
        total = tier_total[tier]
        if total > 0:
            per_tier_accuracy[tier] = tier_correct[tier] / total
        else:
            per_tier_accuracy[tier] = 0.0

    # Aggregate = mean of per-tier accuracies
    if per_tier_accuracy:
        aggregate = sum(per_tier_accuracy.values()) / len(per_tier_accuracy)
    else:
        aggregate = 0.0

    return {
        "model_name": model_name,
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_tier_accuracy": per_tier_accuracy,
        "per_tier_correct": tier_correct,
        "per_tier_total": tier_total,
        "per_tier_excluded": tier_excluded,
        "aggregate_accuracy": aggregate,
        "total_probes": len(probe_results),
        "total_excluded": sum(tier_excluded.values()),
    }


def load_all_probes(stage: str = "calibrated") -> list[dict]:
    """Load probes from all tiers."""
    probes_dir = PROJECT_ROOT / "data" / "probes"
    all_probes = []
    for tier in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        tier_file = probes_dir / f"{tier}_{stage}.json"
        if tier_file.exists():
            with open(tier_file) as f:
                tier_probes = json.load(f)
                all_probes.extend(tier_probes)
                logger.info(f"Loaded {len(tier_probes)} probes from {tier}")
        else:
            logger.warning(f"No probes found for {tier} at stage '{stage}'")
    return all_probes
