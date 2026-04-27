#!/usr/bin/env python3
"""Phase 4: Run probes against target (unknown size) models.

Usage: python scripts/04_run_targets.py [--models model1,model2,...] [--probe-stage calibrated]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from probe_runner import run_model_probes, load_all_probes
from calibration import estimate_target_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "target_run.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run IKP probes against target models")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all target models)")
    parser.add_argument("--probe-stage", type=str, default="candidates",
                        help="Probe stage to use")
    parser.add_argument("--max-probes-per-tier", type=int, default=50,
                        help="Max probes per tier (default: 50 for pilot)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "configs" / "models.json") as f:
        models_config = json.load(f)
    with open(PROJECT_ROOT / "configs" / "experiment.json") as f:
        exp_config = json.load(f)

    # Load calibration
    cal_file = PROJECT_ROOT / "data" / "calibration" / "calibration_fit.json"
    if not cal_file.exists():
        logger.error("No calibration fit found. Run 03_fit_calibration.py first.")
        sys.exit(1)
    with open(cal_file) as f:
        calibration = json.load(f)

    # Load probes
    probes = load_all_probes(stage=args.probe_stage)
    if not probes:
        logger.error("No probes found.")
        sys.exit(1)

    # Subsample to max_probes_per_tier for pilot runs
    import random
    random.seed(42)
    by_tier = {}
    for p in probes:
        tier = p["tier"]
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(p)
    probes = []
    for tier in sorted(by_tier.keys()):
        tier_probes = by_tier[tier]
        n = min(args.max_probes_per_tier, len(tier_probes))
        selected_probes = random.sample(tier_probes, n)
        probes.extend(selected_probes)
        logger.info(f"  {tier}: selected {n}/{len(tier_probes)} probes")

    logger.info(f"Loaded {len(probes)} probes and calibration fit")

    # Select target models
    target_models = models_config["target_models"]
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        selected = {k: v for k, v in target_models.items() if k in model_names}
    else:
        selected = target_models

    client = OpenRouterClient(
        requests_per_minute=50,
        max_retries=exp_config["api"]["max_retries"],
        retry_delay=exp_config["api"]["retry_delay_seconds"],
        timeout=120,  # Longer timeout for reasoning models (o1, o3)
    )

    output_dir = PROJECT_ROOT / "data" / "raw_responses"
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates = []

    for model_name, model_info in selected.items():
        logger.info(f"\n=== {model_name} ({model_info['vendor']}) ===")
        summary = run_model_probes(
            client=client,
            model_name=model_name,
            model_id=model_info["openrouter_id"],
            probes=probes,
            output_dir=output_dir,
        )

        # Estimate parameter count
        estimate = estimate_target_model(summary, calibration)
        estimate["vendor"] = model_info["vendor"]
        estimate["reference_estimate_billion"] = model_info.get("reference_estimate_billion")
        estimate["reference_source"] = model_info.get("reference_source")
        estimates.append(estimate)

        logger.info(
            f"  Estimate: {estimate['estimated_params_aggregate_B']:.1f}B (aggregate), "
            f"CI: [{estimate['ci_95_aggregate_B'][0]:.1f}B, {estimate['ci_95_aggregate_B'][1]:.1f}B]"
        )
        if estimate.get("estimated_params_tier_B"):
            logger.info(f"  Estimate (tier): {estimate['estimated_params_tier_B']:.1f}B")

    # Save estimates
    output_file = PROJECT_ROOT / "results" / "target_estimates.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(estimates, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"{'Model':25s} {'Vendor':10s} {'Agg Acc':>8s} {'Est (B)':>10s} {'CI low':>10s} {'CI high':>10s} {'Ref (B)':>10s}")
    logger.info(f"{'='*80}")
    for e in sorted(estimates, key=lambda x: x["estimated_params_aggregate_B"]):
        ref = f"{e['reference_estimate_billion']:.0f}" if e.get("reference_estimate_billion") else "?"
        logger.info(
            f"{e['model_name']:25s} {e.get('vendor','?'):10s} "
            f"{e['aggregate_accuracy']:8.3f} "
            f"{e['estimated_params_aggregate_B']:10.1f} "
            f"{e['ci_95_aggregate_B'][0]:10.1f} "
            f"{e['ci_95_aggregate_B'][1]:10.1f} "
            f"{ref:>10s}"
        )

    logger.info(f"\nEstimates saved to {output_file}")


if __name__ == "__main__":
    main()
