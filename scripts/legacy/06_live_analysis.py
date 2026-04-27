#!/usr/bin/env python3
"""Live analysis: process results as they come in from background runs.

Run this periodically to see the latest state of all experiments.
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
config = json.load(open(PROJECT_ROOT / "configs" / "models.json"))
cal_models = config["calibration_models"]
target_models = config["target_models"]

# Load calibration fit
cal_fit_file = PROJECT_ROOT / "data" / "calibration" / "calibration_fit.json"
cal_fit = json.load(open(cal_fit_file)) if cal_fit_file.exists() else None

def load_all_results():
    """Load all available results from raw_responses."""
    results = {}
    for f in sorted(Path(PROJECT_ROOT / "data" / "raw_responses").glob("*_results.json")):
        r = json.load(open(f))
        results[r['model_name']] = r
    return results

def estimate_params(agg_accuracy, cal_fit):
    """Estimate params from aggregate accuracy using calibration fit."""
    if cal_fit is None:
        return None
    alpha = cal_fit['aggregate_fit']['alpha']
    beta = cal_fit['aggregate_fit']['beta']
    log_n = alpha * agg_accuracy + beta
    return np.exp(log_n)

def main():
    results = load_all_results()

    # === Calibration Models ===
    print("=" * 100)
    print("  CALIBRATION MODELS")
    print("=" * 100)
    print(f"{'Model':25s} {'Params':>7s} {'Arch':>6s} {'T1':>6s} {'T2':>6s} {'T3':>6s} {'T4':>6s} {'T5':>6s} {'T6':>6s} {'T7':>6s} {'Agg':>6s}")
    print("-" * 100)

    cal_data = []
    for name in sorted(cal_models.keys(), key=lambda x: cal_models[x]['params_billion']):
        if name not in results:
            continue
        r = results[name]
        info = cal_models[name]
        ta = r['per_tier_accuracy']
        tiers = " ".join(f"{ta.get(t, 0):6.3f}" for t in ['T1','T2','T3','T4','T5','T6','T7'])
        arch = 'MoE' if info['architecture'] == 'moe' else 'Dense'
        print(f"{name:25s} {info['params_billion']:7.1f} {arch:>6s} {tiers} {r['aggregate_accuracy']:6.3f}")
        cal_data.append((info['params_billion'], r['aggregate_accuracy'], name))

    print(f"\n  Total calibration models with results: {len(cal_data)}")

    # === Target Models ===
    print("\n" + "=" * 100)
    print("  TARGET MODEL ESTIMATES")
    print("=" * 100)
    print(f"{'Model':25s} {'Vendor':>10s} {'T1':>6s} {'T2':>6s} {'T3':>6s} {'T4':>6s} {'T5':>6s} {'T6':>6s} {'T7':>6s} {'Agg':>6s} {'Est(B)':>8s} {'Ref':>8s}")
    print("-" * 100)

    target_data = []
    for name in sorted(target_models.keys()):
        if name not in results:
            continue
        r = results[name]
        info = target_models[name]
        ta = r['per_tier_accuracy']
        tiers = " ".join(f"{ta.get(t, 0):6.3f}" for t in ['T1','T2','T3','T4','T5','T6','T7'])
        est = estimate_params(r['aggregate_accuracy'], cal_fit)
        est_str = f"{est:8.0f}" if est else "   ---  "
        ref = info.get('reference_estimate_billion')
        ref_str = f"{ref:8.0f}" if ref else "   ---  "
        print(f"{name:25s} {info.get('vendor','?'):>10s} {tiers} {r['aggregate_accuracy']:6.3f} {est_str} {ref_str}")
        target_data.append((name, r['aggregate_accuracy'], est, ref, info.get('vendor')))

    if not target_data:
        print("  (no target results yet)")

    print(f"\n  Total target models with results: {len(target_data)}")

    # === Key comparisons ===
    if len(target_data) > 0 and cal_fit:
        print("\n" + "=" * 80)
        print("  KEY FINDINGS")
        print("=" * 80)

        # Reasoning model comparison
        reasoning_pairs = [
            ("gpt-4o", "o3", "GPT-4o vs o3: same base, different reasoning?"),
            ("deepseek-v3", "deepseek-r1", "DeepSeek V3 vs R1: same weights, different reasoning?"),
        ]
        for base, reasoning, desc in reasoning_pairs:
            if base in results and reasoning in results:
                base_agg = results[base]['aggregate_accuracy']
                reas_agg = results[reasoning]['aggregate_accuracy']
                diff = reas_agg - base_agg
                print(f"\n  {desc}")
                print(f"    {base}: {base_agg:.3f}")
                print(f"    {reasoning}: {reas_agg:.3f}")
                print(f"    Delta: {diff:+.3f} ({'consistent with same base' if abs(diff) < 0.05 else 'DIFFERENT base weights'})")


if __name__ == "__main__":
    main()
