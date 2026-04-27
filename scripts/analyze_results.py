#!/usr/bin/env python3
"""Analyze IKP evaluation results.

Produces:
1. Scaling curves: accuracy vs log(params) per tier
2. Per-tier logistic sigmoid fits
3. Aggregate accuracy table (all models)
4. Parameter estimation for proprietary models
5. Distillation detection analysis
6. Per-source breakdown (LLM, researcher, Wikidata)
"""

import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_results():
    """Load all model result files."""
    results_dir = PROJECT_ROOT / "data" / "results"
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "evaluation_summary.json":
            continue
        try:
            data = json.load(open(f))
            results[data["model_name"]] = data
        except:
            pass
    return results


def fit_logistic(params_list, acc_list):
    """Fit logistic sigmoid: accuracy = 1 / (1 + exp(-k*(log(params) - x0)))
    Returns (k, x0, r_squared).
    """
    if len(params_list) < 3:
        return None, None, 0

    log_params = np.array([math.log10(p) for p in params_list])
    accs = np.array(acc_list)

    # Clamp accs to (0.01, 0.99) to avoid log(0)
    accs = np.clip(accs, 0.01, 0.99)

    # Linearize: log(acc/(1-acc)) = k * log(params) - k * x0
    logits = np.log(accs / (1 - accs))

    # Linear regression
    try:
        A = np.vstack([log_params, np.ones(len(log_params))]).T
        result = np.linalg.lstsq(A, logits, rcond=None)
        k = result[0][0]
        b = result[0][1]
        x0 = -b / k if k != 0 else 0

        # R-squared
        predicted = 1 / (1 + np.exp(-(k * log_params + b)))
        ss_res = np.sum((accs - predicted) ** 2)
        ss_tot = np.sum((accs - np.mean(accs)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return k, x0, r_squared
    except:
        return None, None, 0


def estimate_params(tier_accuracies, fits):
    """Estimate parameter count from tier accuracies using fitted curves.
    Returns geometric mean of per-tier estimates.
    """
    estimates = []
    for tier, acc in tier_accuracies.items():
        if tier not in fits or fits[tier][0] is None:
            continue
        k, x0, r2 = fits[tier]
        if r2 < 0.5 or k == 0:
            continue
        if acc <= 0.01 or acc >= 0.99:
            continue
        # Inverse logistic: log10(params) = logit(acc)/k + x0
        logit = math.log(acc / (1 - acc))
        log_params = logit / k + x0
        if 0 < log_params < 4:  # 1B to 10000B range
            estimates.append(10 ** log_params)

    if estimates:
        # Geometric mean
        return math.exp(sum(math.log(e) for e in estimates) / len(estimates))
    return None


def main():
    results = load_results()
    logger.info(f"Loaded results for {len(results)} models")

    if not results:
        logger.warning("No results found. Run evaluation first.")
        return

    # Load model config for ground truth params
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))
    model_config = config["models"]

    # Calibration exclusions (keep in sync with scripts/loo_cv_analysis.py).
    CALIBRATION_EXCLUDE = {
        'minimax-m1-think',      # broken API: most responses empty
        'hunyuan-a13b',          # extreme outlier: 80B scores below 12B dense models
        'hunyuan-a13b-think',    # same issue as hunyuan-a13b
        'hermes-3-405b',         # superseded by hermes-4-405b (3.5σ outlier)
        'ling-2.6-flash',        # extreme outlier: 104B MoE scores like a 1B dense model
        'nemotron-ultra-253b',
        'deepseek-v3.1-nex-n1',  # post-training: -5.58σ below-trend
        'intellect-3-think',     # post-training: GLM-4.5-Air-Base SFT+RL, separate regime
    }

    # Separate known-size and unknown-size models
    known = []
    unknown = []
    for name, data in results.items():
        params = data.get("params_B")
        if params and params > 0 and name not in CALIBRATION_EXCLUDE:
            known.append(data)
        else:
            unknown.append(data)

    logger.info(f"Known-size models: {len(known)}")
    logger.info(f"Unknown-size models: {len(unknown)}")

    # 1. Print accuracy table
    print("\n" + "=" * 100)
    print("MODEL ACCURACY TABLE")
    print("=" * 100)
    header = f"{'Model':30s} {'Params':>8s} {'Family':>12s} {'Agg':>5s}"
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        header += f" {t:>5s}"
    print(header)
    print("-" * 100)

    # Sort by params (known first, then alphabetical for unknown)
    all_sorted = sorted(results.values(), key=lambda d: (0 if d.get("params_B") else 1, d.get("params_B") or 0))

    for data in all_sorted:
        params = data.get("params_B")
        pstr = f"{params:.0f}B" if params else "?"
        family = (data.get("family") or "?")[:12]
        acc = data.get("accuracy", 0)
        row = f"{data['model_name']:30s} {pstr:>8s} {family:>12s} {acc:5.1%}"
        tier_accs = data.get("tier_accuracy", {})
        for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
            ta = tier_accs.get(t, 0)
            row += f" {ta:5.0%}"
        print(row)

    # 2. Fit scaling curves per tier (using known-size models)
    print("\n" + "=" * 100)
    print("SCALING CURVE FITS (known-size models)")
    print("=" * 100)

    fits = {}
    for tier in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        params_list = []
        acc_list = []
        for data in known:
            params = data.get("params_B", 0)
            acc = data.get("tier_accuracy", {}).get(tier, 0)
            if params > 0:
                params_list.append(params)
                acc_list.append(acc)

        k, x0, r2 = fit_logistic(params_list, acc_list)
        fits[tier] = (k, x0, r2)
        if k is not None:
            print(f"  {tier}: k={k:.2f}, x0={x0:.2f} (10^x0={10**x0:.1f}B), R²={r2:.3f}")
        else:
            print(f"  {tier}: insufficient data")

    # 3. Parameter estimation for proprietary models
    print("\n" + "=" * 100)
    print("PARAMETER ESTIMATION (proprietary/unknown-size models)")
    print("=" * 100)

    estimations = []
    for data in unknown:
        tier_accs = data.get("tier_accuracy", {})
        est = estimate_params(tier_accs, fits)
        if est:
            estimations.append((data["model_name"], est, data.get("family"), data.get("vendor")))
            print(f"  {data['model_name']:30s} estimated: {est:>8.1f}B  (family: {data.get('family')}, vendor: {data.get('vendor')})")

    # 4. Distillation detection
    print("\n" + "=" * 100)
    print("DISTILLATION DETECTION")
    print("=" * 100)

    # For distilled models (known), compare estimated vs actual params
    distilled_models = {name: info for name, info in model_config.items()
                       if info.get("note") and "distill" in info.get("note", "").lower()}

    for name, info in distilled_models.items():
        if name in results:
            data = results[name]
            actual = data.get("params_B", 0)
            tier_accs = data.get("tier_accuracy", {})
            est = estimate_params(tier_accs, fits)
            if est and actual:
                ratio = est / actual
                print(f"  {name:35s} actual={actual:.0f}B, estimated={est:.1f}B, ratio={ratio:.2f}x")
                if ratio > 1.5:
                    print(f"    ⚑ Possible knowledge distillation detected (ratio > 1.5x)")

    # 5. Save analysis results
    analysis = {
        "n_models": len(results),
        "n_known": len(known),
        "n_unknown": len(unknown),
        "fits": {t: {"k": k, "x0": x0, "r_squared": r2} for t, (k, x0, r2) in fits.items()},
        "estimations": [{"model": m, "estimated_B": e, "family": f, "vendor": v} for m, e, f, v in estimations],
        "model_accuracies": [
            {
                "model": d["model_name"],
                "params_B": d.get("params_B"),
                "family": d.get("family"),
                "vendor": d.get("vendor"),
                "accuracy": d.get("accuracy"),
                "tier_accuracy": d.get("tier_accuracy"),
            }
            for d in all_sorted
        ],
    }

    output = PROJECT_ROOT / "data" / "results" / "analysis.json"
    with open(output, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved analysis to {output}")


if __name__ == "__main__":
    main()
