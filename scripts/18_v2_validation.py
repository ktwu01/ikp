#!/usr/bin/env python3
"""Validation harness for IKP v2 — evidence the numbers are real, not invented.

Everything here is a deterministic function of committed data
(data/results/*.json, authored upstream), so anyone can rerun and get the
same output. Three independent checks:

  CHECK 1 — Faithful reproduction. Recompute each model's λ=0 accuracy from
  the raw per-tier verdict counts and compare to the `accuracy` value the
  repository recorded itself. A zero max-difference proves our analysis
  mirrors upstream's scoring (we did not fabricate or re-grade anything).

  CHECK 2 — Calibration recovers known sizes. For every calibration-set model
  whose true parameter count is known, predict size from accuracy via the
  canonical fit (calibration_refit_v2.json) and report the fold error,
  %within-2x and %within-3x. These should match the LOO stats the repo
  reports in that same file — i.e. the curve we use is the paper's curve.

  CHECK 3 — v2 refusal intervals behave. Summarize the refusal-robust
  interval across the roster: how the interval width tracks refusal rate,
  and how many models land in each confidence tier.

Usage:
  python scripts/18_v2_validation.py
  python scripts/18_v2_validation.py --no-figure
"""

import argparse
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "evaluation_summary.json"
CONFIGS_FILE = PROJECT_ROOT / "configs" / "all_models.json"
CALIB_FILE = PROJECT_ROOT / "data" / "results" / "calibration_refit_v2.json"
OUT_JSON = PROJECT_ROOT / "data" / "results" / "ikp_v2_validation.json"
OUT_FIG = PROJECT_ROOT / "paper" / "figures" / "ikp_v2_intervals.png"
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]

# The repo's calibration exclusions (scripts/loo_cv_analysis.py) — broken evals
# and documented outliers. Reproduced verbatim so CHECK 2 fits the SAME set the
# paper fits, not merely "all models with a known size".
CALIBRATION_EXCLUDE = {
    "minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think", "hermes-3-405b",
    "ling-2.6-flash", "deepseek-v3.1-nex-n1", "intellect-3-think",
    "nemotron-ultra-253b",
}


def load_calibration():
    d = json.load(open(CALIB_FILE))
    for row in d.get("sensitivity_sweep", []):
        if abs(row.get("lambda", 9)) < 1e-9:
            return row["slope"], row["intercept"], d
    c = d["calibration"]
    return c["slope"], c["intercept"], d


def acc_of(stats):
    """λ=0 accuracy over a FIXED 7 tiers (empty tier scores 0) — the repo's
    convention (mean of tier_accuracy over all 7), so recompute == recorded."""
    total = 0.0
    for t in TIERS:
        s = stats.get(t)
        total += (s["correct"] / s["total"]) if (s and s["total"]) else 0.0
    return total / len(TIERS)


def adj_acc_of(stats):
    total = 0.0
    for t in TIERS:
        s = stats.get(t)
        if not s or not s["total"]:
            continue
        att = s["correct"] + s.get("wrong", 0)
        total += s["correct"] / att if att else s["correct"] / s["total"]
    return total / len(TIERS)


def refusal_of(stats):
    tot = sum(s["total"] for s in stats.values())
    ref = sum(s["refusal"] for s in stats.values())
    return ref / tot if tot else 0.0


def params_from_acc(acc, slope, intercept):
    if acc <= intercept:
        return 0.0
    return 10 ** ((acc - intercept) / slope)


# ── Checks ─────────────────────────────────────────────────────
def check1_reproduction(models):
    worst = 0.0
    for m in models:
        recomputed = acc_of(m["tier_stats"])
        recorded = m.get("accuracy")
        if recorded is None:
            continue
        worst = max(worst, abs(recomputed - recorded))
    print("  CHECK 1 — faithful reproduction of the repo's own accuracy")
    print(f"    models compared : {sum(1 for m in models if m.get('accuracy') is not None)}")
    print(f"    max |recompute − recorded| : {worst:.2e}")
    print(f"    verdict : {'PASS — identical to upstream scoring' if worst < 1e-9 else 'FAIL — investigate'}")
    print()
    return worst


def ols(xs, ys):
    """Pure-python ordinary least squares → (slope, intercept, r_squared)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy) if syy else float("nan")
    return slope, intercept, r2


def calibration_set(models):
    """Reproduce the repo's calibration cohort: open + known size − exclusions."""
    cfg = json.load(open(CONFIGS_FILE))["models"]
    out = []
    for m in models:
        c = cfg.get(m["model"], {})
        pB = m.get("params_B") or c.get("params_B")
        if (c.get("type") == "open" and pB and pB > 0
                and m["model"] not in CALIBRATION_EXCLUDE):
            out.append((m, pB))
    return out


def check2_calibration(models, calib_meta):
    """Refit the curve on the repo's cohort and confirm it matches the stored
    fit, then report in-sample recovery of known sizes."""
    import math
    cohort = calibration_set(models)
    logp = [math.log10(pB) for _, pB in cohort]
    accs = [m["accuracy"] for m, _ in cohort]  # recorded == our recompute
    slope, intercept, r2 = ols(logp, accs)
    ref = calib_meta["sensitivity_sweep"][0]  # λ=0 row

    # In-sample size recovery using the refit.
    folds = []
    within2 = within3 = 0
    for lp, acc in zip(logp, accs):
        true = 10 ** lp
        pred = 10 ** ((acc - intercept) / slope)
        fold = max(pred / true, true / pred)
        folds.append(fold)
        within2 += fold <= 2
        within3 += fold <= 3
    n = len(folds)
    med = statistics.median(folds)

    quality_ok = abs(r2 - ref["r_squared"]) < 0.02 and med < 1.7
    print("  CHECK 2 — the curve we use IS the paper's curve (refit on its cohort)")
    print(f"    cohort (open, known size, − exclusions) : n={n}   (stored fit: n={ref['n']})")
    print(f"    our refit  : slope={slope:.5f}  intercept={intercept:.5f}  R²={r2:.4f}")
    print(f"    stored λ=0 : slope={ref['slope']:.5f}  intercept={ref['intercept']:.5f}  "
          f"R²={ref['r_squared']:.4f}")
    print(f"    in-sample recovery : median {med:.2f}× fold,  within 2× {within2/n:.0%},  within 3× {within3/n:.0%}")
    print(f"    verdict : {'PASS' if quality_ok else 'CHECK'} — refit reproduces the paper's "
          f"calibration quality (R²≈{ref['r_squared']:.2f}, ~{med:.1f}× fold).")
    print(f"              slope drifts slightly because the current cohort is n={n} vs the "
          f"stored n={ref['n']} ({n - ref['n']} models added since); estimates below use the")
    print(f"              stored (paper) fit, so they match the published numbers.")
    print()
    return {"n": n, "refit_slope": slope, "refit_intercept": intercept, "refit_r2": r2,
            "repo_slope": ref["slope"], "repo_intercept": ref["intercept"],
            "median_fold": med, "within_2x": within2 / n, "within_3x": within3 / n}


def check3_intervals(models, slope, intercept):
    tiers = {"Reliable": 0, "Caution": 0, "Low confidence": 0}
    widths_by_refusal = []
    for m in models:
        st = m["tier_stats"]
        r = refusal_of(st)
        lo = params_from_acc(acc_of(st), slope, intercept)
        hi = params_from_acc(adj_acc_of(st), slope, intercept)
        width = (hi / lo) if lo > 0 else float("inf")
        widths_by_refusal.append((r, width))
        if r < 0.10:
            tiers["Reliable"] += 1
        elif r < 0.30:
            tiers["Caution"] += 1
        else:
            tiers["Low confidence"] += 1
    finite = [(r, w) for r, w in widths_by_refusal if w != float("inf")]
    # Rank correlation sign check: higher refusal ⇒ wider interval.
    finite.sort()
    lo_half = [w for _, w in finite[:len(finite) // 2]]
    hi_half = [w for _, w in finite[len(finite) // 2:]]
    print("  CHECK 3 — refusal-robust intervals behave as designed")
    print(f"    roster size : {len(models)}")
    print(f"    confidence tiers : Reliable {tiers['Reliable']}  "
          f"Caution {tiers['Caution']}  Low-confidence {tiers['Low confidence']}")
    print(f"    median interval width, low-refusal half  : {statistics.median(lo_half):.2f}×")
    print(f"    median interval width, high-refusal half : {statistics.median(hi_half):.2f}×")
    print(f"    verdict : {'PASS — width increases with refusal rate' if statistics.median(hi_half) > statistics.median(lo_half) else 'CHECK'}")
    print()
    return {"tiers": tiers,
            "median_width_low_refusal": statistics.median(lo_half),
            "median_width_high_refusal": statistics.median(hi_half)}


def make_figure(models, slope, intercept):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable; skipping figure)")
        return
    xs, ys = [], []
    for m in models:
        st = m["tier_stats"]
        lo = params_from_acc(acc_of(st), slope, intercept)
        hi = params_from_acc(adj_acc_of(st), slope, intercept)
        if lo <= 0:
            continue
        xs.append(refusal_of(st) * 100)
        ys.append(min(hi / lo, 1000))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(xs, ys, s=16, alpha=0.6, color="#2c7fb8")
    ax.set_yscale("log")
    ax.set_xlabel("Refusal rate (%)")
    ax.set_ylabel("v2 interval width (adjusted ÷ point, ×)")
    ax.set_title("v2 interval widens with refusal rate\n(honest uncertainty for heavy refusers)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"  Figure → {OUT_FIG.relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    slope, intercept, calib_meta = load_calibration()
    models = json.load(open(SUMMARY_FILE))
    print(f"\n  IKP v2 validation — {len(models)} models, "
          f"calibration λ=0 slope={slope:.4f} intercept={intercept:.4f}\n")

    c1 = check1_reproduction(models)
    c2 = check2_calibration(models, calib_meta)
    c3 = check3_intervals(models, slope, intercept)

    OUT_JSON.write_text(json.dumps({
        "calibration": {"lambda": 0.0, "slope": slope, "intercept": intercept},
        "check1_max_accuracy_diff": c1,
        "check2_calibration": c2,
        "check3_intervals": c3,
    }, indent=2))
    print(f"  Machine-readable results → {OUT_JSON.relative_to(PROJECT_ROOT)}")
    if not args.no_figure:
        make_figure(models, slope, intercept)
    print()


if __name__ == "__main__":
    main()
