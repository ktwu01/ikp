#!/usr/bin/env python3
"""Hallucination-penalty (λ) sensitivity table for the IKP calibration.

For λ in a small sweep, recompute each model's accuracy from stored per-tier
verdict counts, refit the log-linear calibration on the paper's open-weight
calibration set, and report fit quality plus the estimated size of a few
flagship proprietary models. λ = 0 (no penalty) is the paper's operating point.

Writes:
  website/public/data/sensitivity.json   (consumed by the Calibration page)
  paper/tables/lambda_sensitivity.tex     (\\input by paper/main.tex)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "data" / "results" / "evaluation_summary.json"
CONFIGS = ROOT / "configs" / "all_models.json"
OUT_JSON = ROOT / "website" / "public" / "data" / "sensitivity.json"
OUT_TEX = ROOT / "paper" / "tables" / "lambda_sensitivity.tex"

TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
LAMBDAS = [0.0, -0.25, -0.5, -1.0, -2.0]
# Keep in sync with scripts/loo_cv_analysis.py / website prepare_data.py
CALIBRATION_EXCLUDE = {
    "minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think", "hermes-3-405b",
    "ling-2.6-flash", "nemotron-ultra-253b", "deepseek-v3.1-nex-n1", "intellect-3-think",
}
# Flagship models to track across λ (label -> summary model name)
SPOTLIGHT = [
    ("GPT-5.5 Pro", "gpt-5.5-pro"),
    ("Claude Fable 5", "claude-fable-5"),
    ("GPT-4.1", "gpt-4.1"),
    ("Claude Opus 4.7", "claude-opus-4.7"),
]


def acc_at(tier_stats, lam):
    accs = []
    for t in TIERS:
        s = tier_stats.get(t)
        if not s or not s.get("total"):
            accs.append(0.0); continue
        accs.append(max((s["correct"] + lam * s["wrong"]) / s["total"], 0.0))
    return float(np.mean(accs))


def fmt_size(b):
    if b is None:
        return "--"
    return f"{b/1000:.1f}T" if b >= 1000 else f"{b:.0f}B"


def main():
    summary = json.load(open(SUMMARY))
    cfg = json.load(open(CONFIGS))["models"]
    by_name = {m["model"]: m for m in summary}

    open_models = [
        m for m in summary
        if cfg.get(m["model"], {}).get("type") == "open"
        and cfg.get(m["model"], {}).get("params_B")
        and cfg[m["model"]]["params_B"] > 0
        and m["model"] not in CALIBRATION_EXCLUDE
    ]
    log_p = np.array([math.log10(cfg[m["model"]]["params_B"]) for m in open_models])

    rows = []
    for lam in LAMBDAS:
        accs = np.array([acc_at(m["tier_stats"], lam) for m in open_models])
        slope, intercept, r, _, _ = stats.linregress(log_p, accs)
        r2 = r ** 2
        resid = accs - (slope * log_p + intercept)
        se = math.sqrt(float(np.sum(resid ** 2)) / max(len(accs) - 2, 1))
        pi_factor = 10 ** (1.645 * se / abs(slope)) if slope else float("inf")
        # LOO median multiplicative parameter error
        folds = []
        n = len(open_models)
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False
            sl, ic, _, _, _ = stats.linregress(log_p[mask], accs[mask])
            if sl > 0:
                pred_log = (accs[i] - ic) / sl
                folds.append(10 ** abs(pred_log - log_p[i]))
        med_fold = float(np.median(folds))
        within2 = float(np.mean(np.array(folds) <= 2))
        # spotlight estimates
        ests = {}
        for label, name in SPOTLIGHT:
            m = by_name.get(name)
            if m and slope > 0:
                a = acc_at(m["tier_stats"], lam)
                ests[label] = 10 ** ((a - intercept) / slope)
            else:
                ests[label] = None
        rows.append({
            "lambda": lam,
            "slope_pp": slope * 100,
            "r_squared": r2,
            "loo_median_fold": med_fold,
            "within_2x": within2,
            "pi_factor": pi_factor,
            "estimates": ests,
        })

    OUT_JSON.write_text(json.dumps({
        "n_calibration": len(open_models),
        "operating_point": 0.0,
        "spotlight": [s[0] for s in SPOTLIGHT],
        "rows": rows,
    }, indent=2))
    print(f"wrote {OUT_JSON} ({len(rows)} λ rows, {len(open_models)} calibration models)")

    # LaTeX table
    spot = [s[0] for s in SPOTLIGHT]
    lines = [
        r"\begin{tabular}{r" + "c" * 4 + "r" * len(spot) + "}",
        r"\toprule",
        r"$\lambda$ & Slope & $R^2$ & LOO$\times$ & PI$\times$ & "
        + " & ".join(spot) + r" \\",
        r" & (pp/dec) & & (med.) & (90\%) & " + " & ".join(["est."] * len(spot)) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        star = r"\;$^\star$" if row["lambda"] == 0.0 else ""
        cells = [
            f"${row['lambda']:.2f}$" + star,
            f"{row['slope_pp']:.1f}",
            f"{row['r_squared']:.3f}",
            f"{row['loo_median_fold']:.2f}",
            f"{row['pi_factor']:.2f}",
        ] + [fmt_size(row["estimates"][s]) for s in spot]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX}")

    # console preview
    hdr = ["λ", "slope", "R²", "LOO×", "PI×"] + spot
    print("  " + "  ".join(f"{h:>10}" for h in hdr))
    for row in rows:
        vals = [f"{row['lambda']:+.2f}", f"{row['slope_pp']:.1f}", f"{row['r_squared']:.3f}",
                f"{row['loo_median_fold']:.2f}", f"{row['pi_factor']:.2f}"] + \
               [fmt_size(row["estimates"][s]) for s in spot]
        print("  " + "  ".join(f"{v:>10}" for v in vals))


if __name__ == "__main__":
    main()
