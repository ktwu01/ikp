#!/usr/bin/env python3
"""Phase 2b: Fit calibration curves from calibration run results.

Usage: python scripts/03_fit_calibration.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration import run_full_calibration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    results_file = PROJECT_ROOT / "data" / "calibration" / "all_calibration_results.json"
    if not results_file.exists():
        logger.error("No calibration results found. Run 02_run_calibration.py first.")
        sys.exit(1)

    with open(results_file) as f:
        cal_results = json.load(f)

    logger.info(f"Loaded results for {len(cal_results)} calibration models")

    # Prepare calibration data
    calibration_data = []
    for r in cal_results:
        calibration_data.append({
            "model_name": r["model_name"],
            "params_billion": r["params_billion"],
            "per_tier_accuracy": r["per_tier_accuracy"],
            "aggregate_accuracy": r["aggregate_accuracy"],
            "family": r.get("family", "unknown"),
            "architecture": r.get("architecture", "dense"),
            "active_params_billion": r.get("active_params_billion", r["params_billion"]),
        })

    # Run calibration
    calibration = run_full_calibration(calibration_data)

    # Save
    output_file = PROJECT_ROOT / "data" / "calibration" / "calibration_fit.json"
    with open(output_file, "w") as f:
        json.dump(calibration, f, indent=2)

    # Print summary
    agg = calibration["aggregate_fit"]
    logger.info(f"\n=== Calibration Fit Summary ===")
    logger.info(f"Aggregate log-linear: R²={agg['r_squared']:.4f}, α={agg['alpha']:.4f}, β={agg['beta']:.4f}")
    logger.info(f"Pearson r={agg['pearson_r']:.4f}, p={agg['pearson_p']:.2e}")

    logger.info(f"\nPer-tier logistic fits:")
    for tier, fit in sorted(calibration["tier_fits"].items()):
        if fit["converged"]:
            logger.info(f"  {tier}: R²={fit['r_squared']:.4f}, L={fit['L']:.3f}, k={fit['k']:.3f}, m={fit['m']:.3f}")
        else:
            logger.info(f"  {tier}: FAILED TO CONVERGE")

    loocv = calibration["loocv"]
    logger.info(f"\nLOO-CV (aggregate estimator):")
    logger.info(f"  Median mult error: {loocv['aggregate_estimator']['median_mult_error']:.2f}x")
    logger.info(f"  Mean mult error: {loocv['aggregate_estimator']['mean_mult_error']:.2f}x")
    logger.info(f"  % within 2x: {loocv['aggregate_estimator']['pct_within_2x']:.1%}")

    if "tier_estimator" in loocv:
        logger.info(f"\nLOO-CV (per-tier estimator):")
        logger.info(f"  Median mult error: {loocv['tier_estimator']['median_mult_error']:.2f}x")
        logger.info(f"  % within 2x: {loocv['tier_estimator']['pct_within_2x']:.1%}")

    logger.info(f"\nCalibration saved to {output_file}")


if __name__ == "__main__":
    main()
