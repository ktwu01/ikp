"""Refit log-linear calibration on the final-assembled dataset.

Uses data/results/final_assembly.json (researcher v2 + non-researcher v1
verdicts merged, lambda=-1, per-tier floor at 0, mean of 7 floored tier scores).

Outputs:
  - Calibration R², slope, intercept, RMSE, LOO median fold error,
    90% PI factor.
  - Spotlight ECRs for V4 Pro/Flash variants, Gemini 3 Flash variants,
    GPT-5 variants, etc.
  - Penalty sensitivity sweep over lambda in {0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0}
    using the same merged dataset, with per-tier floor at 0 each time.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/Users/boj/ikp-paper")
ASSEMBLED = ROOT / "data" / "results" / "final_assembly.json"
RESULTS_V1 = ROOT / "data" / "results"
RESULTS_V2 = ROOT / "data" / "results_v2"
OUT_REFIT = ROOT / "data" / "results" / "calibration_refit_v2.json"
OUT_SENS = ROOT / "data" / "results" / "penalty_sensitivity_v2.csv"

CALIBRATION_EXCLUDE = {"minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think",
                       "hermes-3-405b", "ling-2.6-flash", "deepseek-v3.1-nex-n1",
                       "intellect-3-think"}
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def score_verdict(verdict: str, lam: float) -> float:
    if verdict == "CORRECT_STRONG" or verdict == "CORRECT":
        return 1.0
    if verdict == "CORRECT_WEAK":
        return 0.5
    if verdict == "REFUSAL":
        return 0.0
    if verdict == "WRONG":
        return lam
    return 0.0


def assemble_one(model_name: str, lam: float):
    """Recompute final-assembled accuracy at a custom lambda."""
    v1_file = RESULTS_V1 / f"{model_name}.json"
    if not v1_file.exists():
        return None
    with open(v1_file) as f:
        v1 = json.load(f)

    v2_file = RESULTS_V2 / f"{model_name}.json"
    v2_results_by_pid = {}
    if v2_file.exists():
        try:
            with open(v2_file) as f:
                v2 = json.load(f)
            results = v2.get("results", [])
            err_count = sum(1 for r in results if r.get("model_query_error"))
            if err_count < len(results) * 0.95:
                for r in results:
                    v2_results_by_pid[r["probe_id"]] = r
        except Exception:
            pass

    by_tier_sum = {t: 0.0 for t in TIERS}
    by_tier_count = {t: 0 for t in TIERS}

    for r in v1.get("results", []):
        pid = r.get("probe_id", "")
        tier = r.get("tier")
        if not tier:
            continue
        is_researcher = (r.get("source_type") == "researcher")
        if is_researcher and pid in v2_results_by_pid:
            v2r = v2_results_by_pid[pid]
            verdict = v2r.get("verdict")
        else:
            verdict = r.get("verdict")
        score = score_verdict(verdict, lam)
        by_tier_sum[tier] += score
        by_tier_count[tier] += 1

    tier_acc = {}
    for t in TIERS:
        if by_tier_count[t] == 0:
            tier_acc[t] = 0.0
        else:
            tier_acc[t] = max(by_tier_sum[t] / by_tier_count[t], 0.0)
    overall = sum(tier_acc[t] for t in TIERS) / 7.0
    return overall, tier_acc


def refit_at_lambda(rows, lam):
    """Refit calibration on open models with known params at the given lambda."""
    open_models = [r for r in rows
                    if r["type"] == "open"
                    and r.get("params_B") and r["params_B"] > 0
                    and r["model"] not in CALIBRATION_EXCLUDE]
    log_params = []
    accs = []
    names = []
    for m in open_models:
        result = assemble_one(m["model"], lam)
        if result is None:
            continue
        overall, _ = result
        log_params.append(math.log10(m["params_B"]))
        accs.append(overall)
        names.append(m["model"])
    log_params = np.array(log_params)
    accs = np.array(accs)
    if len(log_params) < 10:
        return None

    slope, intercept, r, _, _ = stats.linregress(log_params, accs)
    r2 = r ** 2
    pred = slope * log_params + intercept
    rmse = float(np.sqrt(np.mean((accs - pred) ** 2)))

    # LOO median fold error
    fold_errs = []
    for i in range(len(log_params)):
        mask = np.ones(len(log_params), dtype=bool)
        mask[i] = False
        sl, intc, _, _, _ = stats.linregress(log_params[mask], accs[mask])
        if sl == 0:
            continue
        pred_log = (accs[i] - intc) / sl
        fold_errs.append(abs(pred_log - log_params[i]))
    fold_mult = 10 ** np.array(fold_errs)
    median_fold = float(np.median(fold_mult))
    within_2x = float(np.mean(fold_mult <= 2))
    within_3x = float(np.mean(fold_mult <= 3))

    # 90% PI factor
    residuals = accs - (slope * log_params + intercept)
    n = len(accs)
    residual_se = float(np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1)))
    pi_half_log10 = 1.645 * residual_se / abs(slope) if slope else float("inf")
    pi_factor = 10 ** pi_half_log10

    return {
        "lambda": lam,
        "n": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "rmse": rmse,
        "loo_median_fold": median_fold,
        "loo_within_2x": within_2x,
        "loo_within_3x": within_3x,
        "pi90_factor": float(pi_factor),
    }


def main():
    with open(ASSEMBLED) as f:
        rows = json.load(f)

    # Penalty sensitivity sweep
    print("=== Penalty sensitivity sweep on final-assembled dataset ===")
    print(f"{'lambda':>8s} {'n':>4s} {'R²':>7s} {'RMSE':>7s} {'slope':>7s} "
          f"{'LOO×':>6s} {'≤2×':>6s} {'≤3×':>6s} {'PI90×':>7s}")
    sens_rows = []
    for lam in [0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0]:
        fit = refit_at_lambda(rows, lam)
        if fit is None: continue
        sens_rows.append(fit)
        print(f"  {lam:+6.2f}  {fit['n']:3d}  {fit['r_squared']:6.3f} "
              f"{fit['rmse']:7.4f} {fit['slope']:+7.3f} "
              f"{fit['loo_median_fold']:6.2f} "
              f"{fit['loo_within_2x']*100:5.1f} {fit['loo_within_3x']*100:5.1f} "
              f"{fit['pi90_factor']:6.2f}")

    # Save sensitivity table
    import csv
    with open(OUT_SENS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)
    print(f"\nWrote {OUT_SENS}")

    # Headline calibration at lambda=-1.0
    primary = next(r for r in sens_rows if r["lambda"] == -1.0)
    print(f"\n=== Primary calibration at lambda=-1.0 ===")
    print(f"  n = {primary['n']}")
    print(f"  R² = {primary['r_squared']:.4f}")
    print(f"  slope = {primary['slope']:.4f}")
    print(f"  intercept = {primary['intercept']:.4f}")
    print(f"  RMSE = {primary['rmse']:.4f}")
    print(f"  LOO median fold = {primary['loo_median_fold']:.2f}×")
    print(f"  Within 2× = {primary['loo_within_2x']*100:.1f}%")
    print(f"  Within 3× = {primary['loo_within_3x']*100:.1f}%")
    print(f"  90% PI factor = {primary['pi90_factor']:.2f}×")

    # Spotlight ECRs at lambda=-1.0
    slope, intercept = primary["slope"], primary["intercept"]
    print(f"\n=== Spotlight ECRs at lambda=-1.0 ===")
    spotlight = ["deepseek-v4-flash", "deepseek-v4-pro",
                 "deepseek-v4-flash-think", "deepseek-v4-pro-think",
                 "gemini-3-flash", "gemini-3-flash-think", "gemini-3.1-pro",
                 "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
                 "claude-opus-4.6-think", "claude-opus-4.7-think",
                 "claude-sonnet-4.6-think", "gpt-5-think", "gpt-5", "o3", "o1",
                 "kimi-k2.5-think", "kimi-k2.6-think", "deepseek-v3",
                 "grok-3", "grok-4"]
    by_name = {r["model"]: r for r in rows}
    for name in spotlight:
        if name not in by_name: continue
        m = by_name[name]
        result = assemble_one(name, -1.0)
        if result is None: continue
        acc, _ = result
        if slope == 0: continue
        pred_log = (acc - intercept) / slope
        pred_B = 10 ** pred_log
        if m.get("params_B"):
            actual_log = math.log10(m["params_B"])
            ecr = 10 ** (pred_log - actual_log)
            print(f"  {name:30s} acc={acc:.4f}  pred={pred_B:8.1f}B  actual={m['params_B']}B  ECR={ecr:.2f}×")
        else:
            print(f"  {name:30s} acc={acc:.4f}  pred={pred_B:8.1f}B  actual=?")

    # Save full refit
    out = {
        "lambda": -1.0,
        "calibration": primary,
        "sensitivity_sweep": sens_rows,
    }
    with open(OUT_REFIT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_REFIT}")


if __name__ == "__main__":
    main()
