#!/usr/bin/env python3
"""Adversarial robustness of IKP: how much can a model operator game its
estimated parameter count?

IKP maps factual accuracy to an estimated parameter count via a calibrated
log-linear curve. That estimate is only meaningful if a black-box operator
cannot cheaply move it. This script quantifies exactly how cheap the two
directions of gaming are, using the *real* per-tier verdicts of every model
in data/results/evaluation_summary.json — no API calls, fully reproducible.

Two threat models (both operate on the per-tier correct/wrong/refusal
counts; at the default no-penalty scoring λ=0, only `correct` moves the
score, so wrong and refusal are interchangeable):

  SANDBAGGING (appear smaller). The operator suppresses a fraction p of the
  answers it actually knows, returning "I don't know" instead. Motivation:
  hide true scale, evade capability disclosure, dodge scaling-based policy.

  CONTAMINATION (appear larger). The operator has seen the probe set and
  converts a fraction q of its non-correct answers into correct ones (rote
  memorization / benchmark leakage). Motivation: look frontier-scale.

Outputs a per-model gaming-resistance table, roster-wide summary stats, a
JSON dump (data/results/adversarial_ikp.json), and — if matplotlib is
available — a figure (paper/figures/adversarial_ikp.png).

Usage:
  python scripts/17_adversarial_robustness.py
  python scripts/17_adversarial_robustness.py --models claude-opus-4.5 gpt-4.1 llama-3.1-8b
  python scripts/17_adversarial_robustness.py --no-figure
"""

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "evaluation_summary.json"
CALIB_FILE = PROJECT_ROOT / "data" / "results" / "calibration_refit_v2.json"
OUT_JSON = PROJECT_ROOT / "data" / "results" / "adversarial_ikp.json"
OUT_FIG = PROJECT_ROOT / "paper" / "figures" / "adversarial_ikp.png"

TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]

# Fallback λ=0 calibration (accuracy = SLOPE·log10(params_B) + INTERCEPT),
# overridden by calibration_refit_v2.json when present.
FALLBACK = {"slope": 0.14922, "intercept": 0.21805}


# ── Calibration ────────────────────────────────────────────────
def load_calibration():
    """Return (slope, intercept) for the λ=0 (no-penalty) fit."""
    try:
        d = json.load(open(CALIB_FILE))
        for row in d.get("sensitivity_sweep", []):
            if abs(row.get("lambda", 1)) < 1e-9:
                return row["slope"], row["intercept"]
    except Exception:
        pass
    return FALLBACK["slope"], FALLBACK["intercept"]


def acc_to_params_B(acc, slope, intercept):
    """Invert accuracy = slope·log10(params_B) + intercept."""
    if acc <= intercept:  # below the smallest calibrated model
        return 0.0
    return 10 ** ((acc - intercept) / slope)


# ── Scoring under attack ───────────────────────────────────────
def tier_counts(model):
    """Extract per-tier (correct, wrong+refusal, total) from a summary row."""
    out = {}
    for t in TIERS:
        s = model["tier_stats"].get(t)
        if not s or not s.get("total"):
            continue
        correct = s["correct"]
        total = s["total"]
        out[t] = (correct, total - correct, total)  # (correct, non_correct, total)
    return out


def base_accuracy(counts):
    """λ=0 accuracy = mean over a FIXED 7 tiers (missing tier = 0), matching the
    repo's convention so this reproduces the recorded `accuracy` exactly."""
    return sum(c / tot for (c, _, tot) in counts.values()) / len(TIERS)


def sandbag_accuracy(counts, p):
    """Suppress fraction p of correct answers uniformly across tiers."""
    return sum((c * (1 - p)) / tot for (c, _, tot) in counts.values()) / len(TIERS)


def contaminate_accuracy(counts, q):
    """Convert fraction q of non-correct answers into correct."""
    return sum((c + q * nc) / tot for (c, nc, tot) in counts.values()) / len(TIERS)


# ── Gaming-resistance metrics ──────────────────────────────────
def effort_to_shrink(counts, slope, intercept, factor):
    """Fraction p of known answers to suppress to look `factor`× smaller.

    At λ=0, sandbagging scales accuracy by (1-p), so
    Δlog10(params) = -acc·p/slope. Solving for a target shrink factor:
        p = log10(factor) · slope / acc
    Returns None if the model is too small to shrink that far.
    """
    acc = base_accuracy(counts)
    if acc <= 0:
        return None
    p = math.log10(factor) * slope / acc
    return p if p <= 1.0 else None


def effort_to_inflate(counts, slope, intercept, factor):
    """Fraction q of wrong/refused answers to convert to appear `factor`× larger.

    Search for the smallest q∈[0,1] whose apparent size ≥ factor × true size.
    Returns None if even full contamination (q=1) can't reach the target.
    """
    true_B = acc_to_params_B(base_accuracy(counts), slope, intercept)
    if true_B <= 0:
        return None
    target = factor * true_B
    lo, hi = 0.0, 1.0
    if acc_to_params_B(contaminate_accuracy(counts, hi), slope, intercept) < target:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        est = acc_to_params_B(contaminate_accuracy(counts, mid), slope, intercept)
        if est >= target:
            hi = mid
        else:
            lo = mid
    return hi


def stealth_sandbag(counts, slope, intercept, factor):
    """Most *plausible* way to look `factor`× smaller: suppress only the
    obscure high tiers (T7→T1), since flubbing rare trivia looks natural but
    flubbing 'capital of France' is an obvious tell.

    Returns (achievable, tiers_touched, frac_of_correct_suppressed) — the
    lowest-tier that stays untouched, so you can report how deep the operator
    must reach into common knowledge.
    """
    target_acc = base_accuracy(counts)  # start; we lower it
    true_B = acc_to_params_B(target_acc, slope, intercept)
    want_B = true_B / factor
    # Suppress whole tiers from the top down until apparent size ≤ want_B.
    remaining = dict(counts)
    order = [t for t in reversed(TIERS) if t in counts]
    suppressed_correct = 0
    total_correct = sum(c for (c, _, _) in counts.values()) or 1
    touched = []
    for t in order:
        c, nc, tot = remaining[t]
        remaining[t] = (0, nc + c, tot)  # all correct in this tier → refusal
        suppressed_correct += c
        touched.append(t)
        acc = base_accuracy(remaining)
        if acc_to_params_B(acc, slope, intercept) <= want_B:
            return True, touched, suppressed_correct / total_correct
    return False, touched, suppressed_correct / total_correct


# ── Driver ─────────────────────────────────────────────────────
def analyze(model, slope, intercept):
    counts = tier_counts(model)
    if not counts:
        return None
    acc = base_accuracy(counts)
    true_B = acc_to_params_B(acc, slope, intercept)
    return {
        "model": model["model"],
        "true_params_B": model.get("params_B"),
        "accuracy": acc,
        "estimated_B": true_B,
        "sandbag_p_for_2x_smaller": effort_to_shrink(counts, slope, intercept, 2.0),
        "sandbag_p_for_5x_smaller": effort_to_shrink(counts, slope, intercept, 5.0),
        "contam_q_for_2x_larger": effort_to_inflate(counts, slope, intercept, 2.0),
        "contam_q_for_5x_larger": effort_to_inflate(counts, slope, intercept, 5.0),
        "stealth_2x": stealth_sandbag(counts, slope, intercept, 2.0),
    }


def fmt_pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def fmt_B(x):
    if not x or x <= 0:
        return "—"
    if x < 1000:
        return f"{x:.0f}B"
    return f"{x/1000:.1f}T"


def print_table(rows):
    print()
    print(f"  Adversarial robustness of IKP  (λ=0; effort = fraction of answers changed)")
    print(f"  {'─' * 92}")
    print(f"  {'Model':28s} {'est.size':>8s} │ "
          f"{'sandbag→½':>10s} {'sandbag→⅕':>10s} │ "
          f"{'contam→2×':>10s} {'contam→5×':>10s}")
    print(f"  {'':28s} {'':>8s} │ {'(refuse)':>10s} {'(refuse)':>10s} │ "
          f"{'(leak)':>10s} {'(leak)':>10s}")
    print(f"  {'─' * 92}")
    for r in rows:
        print(f"  {r['model'][:28]:28s} {fmt_B(r['estimated_B']):>8s} │ "
              f"{fmt_pct(r['sandbag_p_for_2x_smaller']):>10s} "
              f"{fmt_pct(r['sandbag_p_for_5x_smaller']):>10s} │ "
              f"{fmt_pct(r['contam_q_for_2x_larger']):>10s} "
              f"{fmt_pct(r['contam_q_for_5x_larger']):>10s}")
    print(f"  {'─' * 92}")


def summarize(rows):
    def med(key):
        vals = sorted(r[key] for r in rows if r[key] is not None)
        return vals[len(vals) // 2] if vals else None
    print()
    print(f"  Roster-wide medians (n={len(rows)} models):")
    print(f"    To appear 2× smaller: suppress ~{fmt_pct(med('sandbag_p_for_2x_smaller'))} "
          f"of answers the model knows.")
    print(f"    To appear 5× smaller: suppress ~{fmt_pct(med('sandbag_p_for_5x_smaller'))}.")
    print(f"    To appear 2× larger:  leak/memorize ~{fmt_pct(med('contam_q_for_2x_larger'))} "
          f"of the answers it gets wrong.")
    print()
    print(f"  Takeaway: sandbagging (looking smaller) is the cheap, black-box-undetectable")
    print(f"  attack — it needs only refusals. Inflation requires probe-set access, so a")
    print(f"  held-out / rotating probe pool is the primary defense. See ADVERSARIAL_IKP.md.")
    print()


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable; skipping figure)")
        return
    pts = [(r["estimated_B"], r["sandbag_p_for_2x_smaller"])
           for r in rows if r["estimated_B"] and r["sandbag_p_for_2x_smaller"]]
    if not pts:
        return
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(xs, [y * 100 for y in ys], s=18, alpha=0.7, color="#c0392b")
    ax.set_xscale("log")
    ax.set_xlabel("IKP-estimated size (B params, log scale)")
    ax.set_ylabel("% of known answers to suppress\nto appear 2× smaller")
    ax.set_title("Sandbagging IKP is cheap and gets cheaper with scale")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"  Figure → {OUT_FIG.relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="Only show these models in the table")
    ap.add_argument("--no-figure", action="store_true", help="Skip the figure")
    ap.add_argument("--top", type=int, default=12, help="Rows to show (default 12)")
    args = ap.parse_args()

    slope, intercept = load_calibration()
    summary = json.load(open(SUMMARY_FILE))

    rows = [a for a in (analyze(m, slope, intercept) for m in summary) if a]

    # Persist the full analysis for every model.
    OUT_JSON.write_text(json.dumps(
        {"calibration": {"lambda": 0.0, "slope": slope, "intercept": intercept},
         "n_models": len(rows), "models": rows}, indent=2))

    if args.models:
        want = set(args.models)
        shown = [r for r in rows if r["model"] in want]
    else:
        # A spread across the size range: small, mid, frontier.
        rows_sorted = sorted(rows, key=lambda r: r["estimated_B"])
        idx = [int(i * (len(rows_sorted) - 1) / max(args.top - 1, 1))
               for i in range(args.top)]
        shown = [rows_sorted[i] for i in sorted(set(idx))]

    print_table(shown)
    summarize(rows)
    print(f"  Full per-model analysis → {OUT_JSON.relative_to(PROJECT_ROOT)}  "
          f"(calibration λ=0, slope={slope:.4f})")
    if not args.no_figure:
        make_figure(rows)
    print()


if __name__ == "__main__":
    main()
