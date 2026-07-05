"""Refit calibration after the evidence-aware judge rerun.

Combines the new researcher scores (data/results_v2/<model>.json) with the
existing non-researcher results (data/results/<model>.json) to compute a
unified per-model penalized accuracy under λ=-1.0, then refit the log-linear
calibration. Compare R², LOO median fold error, and ECRs to the prior fit.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_V1 = ROOT / "data" / "results"
RESULTS_V2 = ROOT / "data" / "results_v2"
CONFIG = ROOT / "configs" / "all_models.json"

CALIBRATION_EXCLUDE = {"minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think",
                       "hermes-3-405b", "ling-2.6-flash", "deepseek-v3.1-nex-n1",
                       "intellect-3-think"}
LAMBDA = -1.0
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def model_score(model_name):
    """Compute mean per-tier penalized score for a model.

    For researcher probes (T3-T7): use data/results_v2/<model>.json scores
        (CORRECT_STRONG=+1, CORRECT_WEAK=+0.5, REFUSAL=0, WRONG=lambda).
    For non-researcher probes: use data/results/<model>.json with verdict→score
        (CORRECT=+1, REFUSAL=0, WRONG=lambda).
    Per-tier score = mean of all probe scores in that tier.
    Overall = mean of per-tier scores.
    """
    v1_file = RESULTS_V1 / f"{model_name}.json"
    v2_file = RESULTS_V2 / f"{model_name}.json"
    if not v1_file.exists():
        return None
    with open(v1_file) as f:
        v1 = json.load(f)
    v2_results_by_pid = {}
    if v2_file.exists():
        with open(v2_file) as f:
            v2 = json.load(f)
        for r in v2.get("results", []):
            v2_results_by_pid[r["probe_id"]] = r

    by_tier_scores = defaultdict(list)
    for r in v1.get("results", []):
        pid = r.get("probe_id", "")
        tier = r.get("tier")
        if not tier: continue
        # Use v2 score if researcher probe and v2 has it
        if r.get("source_type") == "researcher" and pid in v2_results_by_pid:
            score = v2_results_by_pid[pid].get("score", 0.0)
        else:
            v = r.get("verdict")
            if v == "CORRECT": score = 1.0
            elif v == "REFUSAL": score = 0.0
            elif v == "WRONG": score = LAMBDA
            else: score = 0.0
        by_tier_scores[tier].append(score)

    tier_acc = {}
    for t in TIERS:
        scores = by_tier_scores.get(t, [])
        tier_acc[t] = float(np.mean(scores)) if scores else 0.0
    overall = float(np.mean(list(tier_acc.values())))
    return {"overall": overall, "tier": tier_acc}


def main():
    with open(CONFIG) as f:
        cfg = json.load(f)["models"]

    # Compute scores for all models we have results for
    summaries = {}
    for model_name in sorted(cfg.keys()):
        s = model_score(model_name)
        if s is None: continue
        summaries[model_name] = s

    print(f"Computed scores for {len(summaries)} models\n")

    # Prepare calibration data: open models with known params, not in exclude
    cal_x = []  # log10(params_B)
    cal_y = []  # overall score
    cal_names = []
    for name, info in cfg.items():
        if info.get("type") != "open": continue
        params = info.get("params_B")
        if not params or params <= 0: continue
        if name in CALIBRATION_EXCLUDE: continue
        if name not in summaries: continue
        cal_x.append(math.log10(params))
        cal_y.append(summaries[name]["overall"])
        cal_names.append(name)

    cal_x = np.array(cal_x); cal_y = np.array(cal_y)
    if len(cal_x) < 10:
        print(f"Not enough calibration points: {len(cal_x)}")
        return

    slope, intercept, r, _, _ = stats.linregress(cal_x, cal_y)
    r2 = r**2
    pred = slope * cal_x + intercept
    rmse = float(np.sqrt(np.mean((cal_y - pred)**2)))
    print(f"=== Calibration with evidence-judge results (λ={LAMBDA}) ===")
    print(f"  n = {len(cal_x)}")
    print(f"  slope = {slope:.4f} ({slope*100:.2f} pp per decade)")
    print(f"  intercept = {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # LOO median fold error
    fold_errs = []
    for i in range(len(cal_x)):
        mask = np.ones(len(cal_x), dtype=bool); mask[i] = False
        sl, intc, _, _, _ = stats.linregress(cal_x[mask], cal_y[mask])
        if sl == 0: continue
        pred_log = (cal_y[i] - intc) / sl
        fold_errs.append(abs(pred_log - cal_x[i]))
    fold_mult = 10**np.array(fold_errs)
    print(f"  LOO median fold = {np.median(fold_mult):.2f}x")
    print(f"  LOO within 2x = {np.mean(fold_mult <= 2)*100:.1f}%")
    print(f"  LOO within 3x = {np.mean(fold_mult <= 3)*100:.1f}%")

    # ECR for spotlight models
    print(f"\n=== ECR (effective capacity ratio = predicted / actual params) ===")
    spotlight = ["deepseek-v4-flash", "deepseek-v4-pro",
                 "deepseek-v4-flash-think", "deepseek-v4-pro-think",
                 "gemini-3-flash", "gemini-3-flash-think", "gemini-3.1-pro",
                 "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
    for name in spotlight:
        if name not in summaries: continue
        s = summaries[name]["overall"]
        info = cfg.get(name, {})
        actual_B = info.get("params_B")
        if slope == 0: continue
        pred_log = (s - intercept) / slope
        pred_B = 10**pred_log
        if actual_B:
            ecr = 10**(pred_log - math.log10(actual_B))
            print(f"  {name:30s} acc={s:.4f}  pred={pred_B:8.1f}B  actual={actual_B}B  ECR={ecr:.2f}x")
        else:
            print(f"  {name:30s} acc={s:.4f}  pred={pred_B:8.1f}B  actual=?")

    # Save
    out = ROOT / "data" / "results_v2" / "evidence_calibration.json"
    out.write_text(json.dumps({
        "lambda": LAMBDA,
        "slope": slope, "intercept": intercept, "r_squared": r2, "rmse": rmse,
        "n": len(cal_x),
        "summaries": summaries,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
