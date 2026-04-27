#!/usr/bin/env python3
"""Phase 2: Run probes against calibration (anchor) models.

Usage:
  python scripts/02_run_calibration.py [--models model1,model2,...] [--probe-stage candidates]
  python scripts/02_run_calibration.py --pilot   # Run quick pilot with subset
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from probe_runner import run_model_probes, load_all_probes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "calibration_run.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Models selected for pilot: span the full range with minimal cost
PILOT_MODELS = [
    "llama-3.2-1b",       # ~1B
    "llama-3.2-3b",       # ~3B
    "qwen-2.5-7b",        # ~7B
    "gemma-2-9b",          # ~9B
    "phi-4",               # ~14B
    "gemma-2-27b",         # ~27B
    "qwen-2.5-72b",        # ~72B
    "mistral-large",       # ~123B
    "hermes-3-405b",       # ~405B
    "deepseek-v3",         # ~671B MoE
]


def subsample_probes(probes, per_tier=50, seed=42):
    """Subsample probes for pilot, maintaining tier balance."""
    import random
    random.seed(seed)
    by_tier = {}
    for p in probes:
        tier = p["tier"]
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(p)

    result = []
    for tier in sorted(by_tier.keys()):
        tier_probes = by_tier[tier]
        n = min(per_tier, len(tier_probes))
        selected = random.sample(tier_probes, n)
        result.extend(selected)
        logger.info(f"  {tier}: selected {n}/{len(tier_probes)} probes")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run IKP probes against calibration models")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all calibration models)")
    parser.add_argument("--probe-stage", type=str, default="candidates",
                        help="Probe stage to use: candidates, calibrated, final")
    parser.add_argument("--max-probes-per-tier", type=int, default=None,
                        help="Limit probes per tier (for pilot)")
    parser.add_argument("--pilot", action="store_true",
                        help="Run quick pilot: 50 probes/tier × 10 key models")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "configs" / "models.json") as f:
        models_config = json.load(f)
    with open(PROJECT_ROOT / "configs" / "experiment.json") as f:
        exp_config = json.load(f)

    # Load probes
    probes = load_all_probes(stage=args.probe_stage)
    if not probes:
        logger.error(f"No probes found at stage '{args.probe_stage}'. Run 01_generate_probes.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(probes)} total probes (stage: {args.probe_stage})")

    # Subsample for pilot
    if args.pilot:
        probes = subsample_probes(probes, per_tier=50)
        logger.info(f"Pilot mode: {len(probes)} probes selected")

    elif args.max_probes_per_tier:
        probes = subsample_probes(probes, per_tier=args.max_probes_per_tier)
        logger.info(f"Subsampled to {len(probes)} probes")

    # Select models
    cal_models = models_config["calibration_models"]
    if args.pilot:
        selected = {k: v for k, v in cal_models.items() if k in PILOT_MODELS}
        logger.info(f"Pilot mode: {len(selected)} models selected")
    elif args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        selected = {k: v for k, v in cal_models.items() if k in model_names}
    else:
        selected = cal_models

    logger.info(f"Running {len(probes)} probes against {len(selected)} models")
    total_api_calls = len(probes) * 3 * len(selected)
    logger.info(f"Estimated API calls: {total_api_calls} ({total_api_calls/60:.0f} min at 60 RPM)")

    # Setup client with higher rate for calibration
    client = OpenRouterClient(
        requests_per_minute=50,  # Increase for batch runs
        max_retries=exp_config["api"]["max_retries"],
        retry_delay=exp_config["api"]["retry_delay_seconds"],
        timeout=exp_config["api"]["timeout_seconds"],
    )

    output_dir = PROJECT_ROOT / "data" / "raw_responses"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    # Sort models by size for readable output
    sorted_models = sorted(selected.items(), key=lambda x: x[1]["params_billion"])

    for model_name, model_info in sorted_models:
        logger.info(f"\n=== {model_name} ({model_info['params_billion']}B, {model_info['family']}) ===")
        try:
            summary = run_model_probes(
                client=client,
                model_name=model_name,
                model_id=model_info["openrouter_id"],
                probes=probes,
                output_dir=output_dir,
            )
            summary["params_billion"] = model_info["params_billion"]
            summary["family"] = model_info["family"]
            summary["architecture"] = model_info["architecture"]
            summary["active_params_billion"] = model_info.get("active_params_billion", model_info["params_billion"])
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            continue

    # Save combined results
    results_file = PROJECT_ROOT / "data" / "calibration" / "all_calibration_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"\n{'='*100}")
    logger.info(f"{'Model':25s} {'Params':>8s} {'Family':>10s} {'T1':>6s} {'T2':>6s} {'T3':>6s} {'T4':>6s} {'T5':>6s} {'T6':>6s} {'T7':>6s} {'Agg':>6s}")
    logger.info(f"{'='*100}")
    for s in sorted(all_summaries, key=lambda x: x["params_billion"]):
        ta = s["per_tier_accuracy"]
        tiers_str = " ".join(f"{ta.get(t, 0):6.3f}" for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])
        logger.info(
            f"{s['model_name']:25s} {s['params_billion']:8.1f} {s.get('family','?'):>10s} "
            f"{tiers_str} {s['aggregate_accuracy']:6.3f}"
        )

    stats = client.get_stats()
    logger.info(f"\nAPI stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
