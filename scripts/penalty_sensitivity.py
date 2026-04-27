"""Penalty sensitivity analysis for the IKP calibration.

For penalty in {0, -0.25, -0.5, -1, -1.5, -2, -3}, recompute each open model's
penalized accuracy from stored tier_stats (correct/wrong/total per tier),
re-fit the log-linear calibration on the same calibration set used by the
paper, and report:

  - R² (penalized) and R² (raw, no penalty)
  - LOO-CV median fold error
  - Calibration slope (pp per decade) and 90% PI factor
  - Effective-capacity ratio (ECR) for V4 Flash, V4 Pro, Gem3 Flash, Gem3.1 Pro

Outputs to data/results/penalty_sensitivity.csv and prints a table.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/Users/boj/ikp-paper")
SUMMARY = ROOT / "data" / "results" / "evaluation_summary.json"
CONFIGS = ROOT / "configs" / "all_models.json"
OUT_CSV = ROOT / "data" / "results" / "penalty_sensitivity.csv"

# Same exclusion list used by loo_cv_analysis.py
CALIBRATION_EXCLUDE = {
    "minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think",
    "hermes-3-405b", "ling-2.6-flash", "deepseek-v3.1-nex-n1",
    "intellect-3-think",
}

TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
PENALTIES = [0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0]
SPOTLIGHT = ["deepseek-v4-flash", "deepseek-v4-pro",
             "deepseek-v4-flash-think", "deepseek-v4-pro-think",
             "gemini-3-flash", "gemini-3-flash-think",
             "gemini-3.1-pro", "gemini-3.1-flash-lite",
             "gemini-2.5-flash", "gemini-2.5-pro"]


def penalized_accuracy(tier_stats, penalty):
    """Recompute mean per-tier penalized accuracy under a given penalty."""
    accs = []
    for t in TIERS:
        s = tier_stats.get(t)
        if not s or not s.get("total"):
            accs.append(0.0)
            continue
        score = (s["correct"] + penalty * s["wrong"]) / s["total"]
        accs.append(max(score, 0.0))
    return float(np.mean(accs))


def fit_log_linear(log_params, accs):
    slope, intercept, r, _, _ = stats.linregress(log_params, accs)
    pred = slope * log_params + intercept
    rmse = float(np.sqrt(np.mean((accs - pred) ** 2)))
    return float(slope), float(intercept), float(r ** 2), rmse


def loo_cv(log_params, accs):
    """Leave-one-out median multiplicative error in predicted params (10**slope*acc+intercept)."""
    n = len(log_params)
    fold_errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        sl, intc, _, _, _ = stats.linregress(log_params[mask], accs[mask])
        # Invert: predicted log_params for held-out point given its accuracy
        # log_params = slope * acc + intercept (in our regression), so given acc -> log_param
        pred_log = sl * accs[i] + intc
        fold_errs.append(abs(pred_log - log_params[i]))
    fold_errs = np.array(fold_errs)
    mult = 10 ** fold_errs
    return float(np.median(mult)), float(np.mean(mult > math.log10(2) and 0 or mult > 2))


def ecr(model_acc, model_log_params, slope, intercept):
    """Effective capacity ratio: 10**(predicted - actual log_params)."""
    pred_log = slope * model_acc + intercept
    return float(10 ** (pred_log - model_log_params))


def main():
    with open(SUMMARY) as f:
        summary = json.load(f)
    with open(CONFIGS) as f:
        configs = json.load(f)["models"]

    summary = [m for m in summary if m["model"] != "nemotron-ultra-253b"]
    for m in summary:
        cfg = configs.get(m["model"], {})
        m["arch"] = cfg.get("arch", "unknown")
        m["type"] = cfg.get("type", "unknown")
        m["thinking"] = cfg.get("thinking", False)
        if m.get("params_B") is None and cfg.get("params_B") is not None:
            m["params_B"] = cfg["params_B"]

    cal_models = [m for m in summary
                  if m["type"] == "open"
                  and m.get("params_B")
                  and m["params_B"] > 0
                  and m["model"] not in CALIBRATION_EXCLUDE]

    # Build per-model accuracy under each penalty
    rows = []
    spotlight_rows = {name: {} for name in SPOTLIGHT}

    for penalty in PENALTIES:
        accs_cal = []
        log_params_cal = []
        all_accs = {}  # model -> acc
        for m in summary:
            acc_p = penalized_accuracy(m.get("tier_stats", {}), penalty)
            all_accs[m["model"]] = acc_p

        for m in cal_models:
            log_params_cal.append(math.log10(m["params_B"]))
            accs_cal.append(all_accs[m["model"]])

        log_params_cal = np.array(log_params_cal)
        accs_cal = np.array(accs_cal)

        slope, intercept, r2, rmse = fit_log_linear(log_params_cal, accs_cal)

        # LOO-CV median multiplicative fold error
        fold_errs_log = []
        for i in range(len(accs_cal)):
            mask = np.ones(len(accs_cal), dtype=bool); mask[i] = False
            sl, intc, _, _, _ = stats.linregress(log_params_cal[mask], accs_cal[mask])
            # Invert regression: given acc, predict log_params via inverse
            # accuracy = sl * log_params + intc  => log_params = (acc - intc)/sl
            if sl != 0:
                pred_log = (accs_cal[i] - intc) / sl
                fold_errs_log.append(abs(pred_log - log_params_cal[i]))
        fold_errs_log = np.array(fold_errs_log)
        fold_mult = 10 ** fold_errs_log
        median_fold = float(np.median(fold_mult))
        within_2x = float(np.mean(fold_mult <= 2))
        within_3x = float(np.mean(fold_mult <= 3))

        # 90% PI factor
        residuals = accs_cal - (slope * log_params_cal + intercept)
        n = len(accs_cal)
        residual_se = float(np.sqrt(np.sum(residuals ** 2) / (n - 2)))
        pi_half_log10 = 1.645 * residual_se / abs(slope) if slope else float("inf")
        pi_factor = 10 ** pi_half_log10

        # Note: regression in loo_cv_analysis.py uses log_params -> accuracy;
        # but ECR is computed by inverting acc -> log_params:
        # observed_log_params - predicted_log_params, where predicted_log_params = (acc - intc)/sl
        # ECR = 10**(predicted - actual) > 1 means above-trend (overshoots params)

        row = {
            "penalty": penalty,
            "n": n,
            "slope_pp_per_decade": slope * 100 if penalty == 0 or True else slope * 100,
            # we report slope with units = accuracy per log10(N)
            "slope": slope,
            "intercept": intercept,
            "R2_penalized": r2,
            "rmse": rmse,
            "loo_median_fold_x": median_fold,
            "loo_within_2x": within_2x,
            "loo_within_3x": within_3x,
            "PI90_factor_x": pi_factor,
        }

        # Effective capacity (predicted params in B) for spotlight models
        for name in SPOTLIGHT:
            m = next((mm for mm in summary if mm["model"] == name), None)
            if not m:
                continue
            acc = all_accs.get(name)
            if acc is None or slope == 0:
                continue
            pred_log = (acc - intercept) / slope
            pred_B = 10 ** pred_log
            ecr_val = None
            if m.get("params_B"):
                actual_log = math.log10(m["params_B"])
                ecr_val = 10 ** (pred_log - actual_log)
            row[f"predB_{name}"] = pred_B
            if ecr_val is not None:
                row[f"ECR_{name}"] = ecr_val
            spotlight_rows[name][penalty] = (acc, ecr_val, pred_B)

        rows.append(row)

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print results
    print(f"\n=== Penalty sensitivity (n_calibration={rows[0]['n']}) ===")
    print(f"{'penalty':>8s} {'R²':>7s} {'RMSE':>7s} {'slope':>7s} {'LOO×':>6s} {'≤2×%':>6s} {'≤3×%':>6s} {'PI90×':>7s}")
    for r in rows:
        print(f"  {r['penalty']:+6.2f}  {r['R2_penalized']:6.3f} {r['rmse']:7.4f} "
              f"{r['slope']:+7.3f} {r['loo_median_fold_x']:6.2f} "
              f"{r['loo_within_2x']*100:5.1f} {r['loo_within_3x']*100:5.1f} {r['PI90_factor_x']:6.2f}")

    print(f"\n=== Spotlight: predicted effective capacity (B) ===")
    hdr = "  " + f"{'model':30s} " + " ".join(f"{p:>8.2f}" for p in PENALTIES)
    print(hdr)
    for name in SPOTLIGHT:
        m = next((mm for mm in summary if mm["model"] == name), None)
        if not m: continue
        params = m.get("params_B")
        ptag = f"({params}B)" if params else "(?)"
        line = f"  {name+ptag:30s} "
        for p in PENALTIES:
            cell = spotlight_rows[name].get(p)
            v = cell[2] if cell else None
            line += f" {v:8.1f}" if v is not None else "      n/a"
        print(line)

    print(f"\n=== Spotlight: ECR (effective capacity / actual params) ===")
    print(hdr)
    for name in SPOTLIGHT:
        m = next((mm for mm in summary if mm["model"] == name), None)
        if not m or not m.get("params_B"): continue
        params = m["params_B"]
        line = f"  {name+'('+str(params)+'B)':30s} "
        for p in PENALTIES:
            cell = spotlight_rows[name].get(p)
            v = cell[1] if cell else None
            line += f" {v:8.2f}" if v is not None else "      n/a"
        print(line)

    print(f"\n=== Spotlight: penalized accuracy ===")
    print(hdr)
    for name in SPOTLIGHT:
        line = f"  {name:30s} "
        for p in PENALTIES:
            cell = spotlight_rows[name].get(p)
            v = cell[0] if cell else None
            line += f" {v:8.4f}" if v is not None else "      n/a"
        print(line)

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
