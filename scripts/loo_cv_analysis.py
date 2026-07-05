#!/usr/bin/env python3
"""
Leave-one-out cross-validation analysis for IKP calibration.
Generates:
  1. LOO-CV statistics (MAE, median error, % within 2x, % within 3x)
  2. Predicted vs Actual parameter count figure (fig7_loo_validation.pdf)
  3. All calibration statistics from current data for paper text updates
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from adjustText import adjust_text
from scipy import stats


def tex_escape(s):
    """Escape special LaTeX characters in a string."""
    s = s.replace('_', r'\_')
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    return s

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
CONFIGS = ROOT / "configs" / "all_models.json"
OUTDIR = ROOT / "paper" / "figures"

# ── Load data ──────────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "evaluation_summary.json") as f:
    summary = json.load(f)
with open(CONFIGS) as f:
    configs = json.load(f)["models"]

# Exclude broken/outlier evaluations
CALIBRATION_EXCLUDE = {
    'minimax-m1-think',      # broken API: most responses empty
    'hunyuan-a13b',          # extreme outlier: 80B scores below 12B dense models
    'hunyuan-a13b-think',    # same issue as hunyuan-a13b
    'hermes-3-405b',         # superseded by hermes-4-405b (3.5σ outlier)
    'ling-2.6-flash',        # extreme outlier: 104B MoE scores like a 1B dense model
    'deepseek-v3.1-nex-n1',  # post-training (nex-agi); -5.58σ below-trend, factual capacity degraded
    'intellect-3-think',     # post-training (Prime Intellect SFT+RL on GLM-4.5-Air-Base);
                             # excluded as a post-training-regime model, separate from pretraining anchors
}
summary = [m for m in summary if m["model"] != "nemotron-ultra-253b"]

# Merge config info
for m in summary:
    name = m["model"]
    if name in configs:
        cfg = configs[name]
        m["arch"] = cfg.get("arch", "unknown")
        m["type"] = cfg.get("type", "unknown")
        m["thinking"] = cfg.get("thinking", False)
        m["active_B"] = cfg.get("active_B")
        m["vendor"] = cfg.get("vendor", m.get("vendor", "unknown"))
        if m.get("params_B") is None and cfg.get("params_B") is not None:
            m["params_B"] = cfg["params_B"]
    else:
        m["arch"] = "unknown"
        m["type"] = "unknown"
        m["thinking"] = False
        m["active_B"] = None

# ── Open models with known params ──────────────────────────────────────────────
open_models = [m for m in summary
               if m["type"] == "open"
               and m.get("params_B") is not None
               and m["params_B"] > 0
               and m["model"] not in CALIBRATION_EXCLUDE]

print(f"Open models for calibration: {len(open_models)}")

# ── Full calibration statistics ────────────────────────────────────────────────
log_params = np.array([math.log10(m["params_B"]) for m in open_models])
pen_accs = np.array([m["accuracy"] for m in open_models])
raw_accs = np.array([m["raw_accuracy"] for m in open_models])

slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, pen_accs)
r_sq = r_value ** 2

slope_raw, intercept_raw, r_raw, _, _ = stats.linregress(log_params, raw_accs)
r_sq_raw = r_raw ** 2

# Residual SE
y_pred_full = slope * log_params + intercept
residuals = pen_accs - y_pred_full
n = len(open_models)
residual_se = np.sqrt(np.sum(residuals**2) / (n - 2))

# 90% PI half-width in log10 space
pi_half_log10 = 1.645 * residual_se / abs(slope)
pi_factor = 10**pi_half_log10

print(f"\n=== FULL CALIBRATION ===")
print(f"n = {n}")
print(f"Slope = {slope:.3f} ({slope*100:.1f} pp per decade)")
print(f"Intercept = {intercept:.4f}")
print(f"R² (penalized) = {r_sq:.3f}")
print(f"R² (raw) = {r_sq_raw:.3f}")
print(f"Residual SE (accuracy) = {residual_se:.4f}")
print(f"90% PI half-width (log10) = {pi_half_log10:.3f}")
print(f"90% PI factor = {pi_factor:.2f}x")

# ── Subgroup R² ────────────────────────────────────────────────────────────────
dense = [(m, lp, a) for m, lp, a in zip(open_models, log_params, pen_accs) if m["arch"] == "dense"]
dense_nt = [(m, lp, a) for m, lp, a in dense if not m["thinking"]]
moe = [(m, lp, a) for m, lp, a in zip(open_models, log_params, pen_accs) if m["arch"] == "moe"]

print(f"\n=== SUBGROUPS ===")
for label, subset in [("Dense only", dense), ("Dense non-thinking", dense_nt), ("MoE (total)", moe)]:
    if len(subset) < 3:
        continue
    lps = np.array([x[1] for x in subset])
    accs_sub = np.array([x[2] for x in subset])
    sl, intc, rv, _, _ = stats.linregress(lps, accs_sub)
    raws = np.array([x[0]["raw_accuracy"] for x in subset])
    sl_r, _, rv_r, _, _ = stats.linregress(lps, raws)
    print(f"{label}: n={len(subset)}, slope={sl:.3f}, R²(pen)={rv**2:.3f}, R²(raw)={rv_r**2:.3f}")

# MoE active params
moe_active = [(m, math.log10(m["active_B"]), a) for m, _, a in moe
              if m.get("active_B") is not None and m["active_B"] > 0]
if len(moe_active) >= 3:
    lps_a = np.array([x[1] for x in moe_active])
    accs_a = np.array([x[2] for x in moe_active])
    sl_a, _, rv_a, _, _ = stats.linregress(lps_a, accs_a)
    raws_a = np.array([x[0]["raw_accuracy"] for x in moe_active])
    sl_ra, _, rv_ra, _, _ = stats.linregress(lps_a, raws_a)
    print(f"MoE (active): n={len(moe_active)}, slope={sl_a:.3f}, R²(pen)={rv_a**2:.3f}, R²(raw)={rv_ra**2:.3f}")

# ── Thinking pairs ─────────────────────────────────────────────────────────────
model_map = {m["model"]: m for m in summary}
pairs = []
for m in summary:
    name = m["model"]
    if name.endswith("-think"):
        base_name = name[:-6]
        if base_name in model_map:
            base = model_map[base_name]
            pairs.append({
                'base_name': base_name,
                'base_acc': base["accuracy"],
                'think_acc': m["accuracy"],
                'delta': m["accuracy"] - base["accuracy"],
            })

n_improve = sum(1 for p in pairs if p['delta'] > 0)
deltas = [p['delta'] for p in pairs]
print(f"\n=== THINKING PAIRS ===")
print(f"Total pairs: {len(pairs)}")
print(f"Improve: {n_improve}/{len(pairs)}")
print(f"Mean delta: {np.mean(deltas)*100:+.1f} pp")
print(f"Range: {min(deltas)*100:+.1f} to {max(deltas)*100:+.1f} pp")
for p in sorted(pairs, key=lambda x: x['delta']):
    print(f"  {p['base_name']}: {p['delta']*100:+.1f} pp")

# ── Model and vendor counts ───────────────────────────────────────────────────
total_evaluated = len(summary)
vendors_evaluated = len(set(m.get("vendor", "unknown") for m in summary))
prop_models = [m for m in summary if m.get("type") != "open" or m.get("params_B") is None or m["params_B"] <= 0]
print(f"\n=== COUNTS ===")
print(f"Total evaluated: {total_evaluated}")
print(f"Vendors: {vendors_evaluated}")
print(f"Open (calibration): {len(open_models)}")
print(f"Proprietary/estimation targets: {len(prop_models)}")

# ── LEAVE-ONE-OUT CROSS-VALIDATION ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"LEAVE-ONE-OUT CROSS-VALIDATION")
print(f"{'='*60}")

loo_results = []
for i in range(n):
    # Train on all except i
    train_lp = np.delete(log_params, i)
    train_acc = np.delete(pen_accs, i)

    sl_loo, int_loo, _, _, _ = stats.linregress(train_lp, train_acc)

    # Predict the held-out model's log params from its accuracy
    held_acc = pen_accs[i]
    pred_log_params = (held_acc - int_loo) / sl_loo
    actual_log_params = log_params[i]

    error_log10 = pred_log_params - actual_log_params
    ratio = 10**error_log10  # predicted / actual

    loo_results.append({
        'model': open_models[i]["model"],
        'actual_B': open_models[i]["params_B"],
        'actual_log10': actual_log_params,
        'pred_log10': pred_log_params,
        'pred_B': 10**pred_log_params,
        'error_log10': error_log10,
        'abs_error_log10': abs(error_log10),
        'ratio': ratio,
        'accuracy': held_acc,
        'arch': open_models[i]["arch"],
        'vendor': open_models[i].get("vendor", "unknown"),
    })

errors_log10 = np.array([r['abs_error_log10'] for r in loo_results])
ratios = np.array([r['ratio'] for r in loo_results])

# Key statistics
mae_log10 = np.mean(errors_log10)
median_error_log10 = np.median(errors_log10)
within_2x = np.mean((ratios >= 0.5) & (ratios <= 2.0)) * 100
within_3x = np.mean((ratios >= 1/3) & (ratios <= 3.0)) * 100
within_5x = np.mean((ratios >= 0.2) & (ratios <= 5.0)) * 100

print(f"\nMAE (log10 space): {mae_log10:.3f}")
print(f"Median absolute error (log10): {median_error_log10:.3f}")
print(f"Median fold error: {10**median_error_log10:.2f}x")
print(f"Mean fold error: {10**mae_log10:.2f}x")
print(f"Within 2x: {within_2x:.1f}%")
print(f"Within 3x: {within_3x:.1f}%")
print(f"Within 5x: {within_5x:.1f}%")

# LOO R² (correlation between predicted and actual)
pred_log = np.array([r['pred_log10'] for r in loo_results])
actual_log = np.array([r['actual_log10'] for r in loo_results])
loo_r2 = stats.pearsonr(pred_log, actual_log)[0]**2
print(f"LOO R² (predicted vs actual log10 params): {loo_r2:.3f}")

# Worst predictions
print(f"\nWorst 10 predictions:")
worst = sorted(loo_results, key=lambda r: r['abs_error_log10'], reverse=True)[:10]
for r in worst:
    print(f"  {r['model']:30s}: actual={r['actual_B']:.1f}B, pred={r['pred_B']:.1f}B, "
          f"ratio={r['ratio']:.2f}x, err={r['error_log10']:+.3f} log10")

# ── FIGURE: Predicted vs Actual Parameter Count ───────────────────────────────
print(f"\n--- Generating predicted vs actual figure ---")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PAL_BLUE = '#2166AC'
PAL_ORANGE = '#D95F02'
PAL_GREEN = '#1B9E77'

fig, ax = plt.subplots(figsize=(8, 8))

# Perfect prediction line
lims = [min(actual_log.min(), pred_log.min()) - 0.15,
        max(actual_log.max(), pred_log.max()) + 0.15]
ax.plot(lims, lims, 'k-', linewidth=1.5, alpha=0.6, label='Perfect prediction', zorder=1)

# 2x band
ax.fill_between(lims, [l - math.log10(2) for l in lims], [l + math.log10(2) for l in lims],
                alpha=0.12, color=PAL_GREEN, linewidth=0, zorder=0, label=r'Within 2$\times$')
# 3x band
ax.fill_between(lims, [l - math.log10(3) for l in lims], [l + math.log10(3) for l in lims],
                alpha=0.07, color=PAL_BLUE, linewidth=0, zorder=0, label=r'Within 3$\times$')

# Scatter points
for r in loo_results:
    is_moe = r['arch'] == 'moe'
    marker = 'D' if is_moe else 'o'
    color = PAL_ORANGE if is_moe else PAL_BLUE
    ax.scatter(r['actual_log10'], r['pred_log10'], c=color, marker=marker,
               s=50, alpha=0.75, edgecolors='white', linewidths=0.5, zorder=3)

# Legend entries for markers
ax.scatter([], [], c=PAL_BLUE, marker='o', s=50, label='Dense',
           edgecolors='white', linewidths=0.5)
ax.scatter([], [], c=PAL_ORANGE, marker='D', s=50, label='MoE',
           edgecolors='white', linewidths=0.5)

# Label worst outliers with per-model offsets to avoid overlap
worst_5 = sorted(loo_results, key=lambda r: r['abs_error_log10'], reverse=True)[:5]
label_offsets = {
    'llama-3-70b': (-60, 15),
    'hermes-4-405b': (10, 12),
    'command-r-plus': (10, 10),
    'llama-4-scout': (10, -15),
}
for r in worst_5:
    dx, dy = label_offsets.get(r['model'], (10, -12))
    ax.annotate(tex_escape(r['model']), (r['actual_log10'], r['pred_log10']),
                textcoords='offset points', xytext=(dx, dy),
                fontsize=7.5, color='#555555',
                arrowprops=dict(arrowstyle='->', color='#888888',
                                lw=0.6, shrinkA=2, shrinkB=3))

# Axes
ax.set_xlabel(r'Actual $\log_{10}$(Parameters, B)', fontsize=13)
ax.set_ylabel(r'LOO-CV Predicted $\log_{10}$(Parameters, B)', fontsize=13)
ax.set_title('Leave-One-Out Cross-Validation:\nPredicted vs Actual Parameter Count',
             fontsize=14, fontweight='bold', pad=12)

# Secondary axes with actual sizes
for axi, setter, getter in [(ax, ax.set_xlim, ax.get_xlim),
                              (ax, ax.set_ylim, ax.get_ylim)]:
    pass

ax.set_xlim(lims)
ax.set_ylim(lims)

# Add tick labels for actual params
param_ticks = [1, 3, 10, 30, 100, 300, 1000]
param_tick_locs = [math.log10(p) for p in param_ticks]
param_tick_labels = ['1B', '3B', '10B', '30B', '100B', '300B', '1T']
ax.set_xticks(param_tick_locs)
ax.set_xticklabels(param_tick_labels)
ax.set_yticks(param_tick_locs)
ax.set_yticklabels(param_tick_labels)

# Stats box
stats_text = (f'LOO-CV $R^2 = {loo_r2:.3f}$\n'
              f'Median error: {10**median_error_log10:.2f}$\\times$\n'
              f'Within 2$\\times$: {within_2x:.0f}\\%\n'
              f'Within 3$\\times$: {within_3x:.0f}\\%\n'
              f'$n = {n}$ open models')
ax.text(0.03, 0.97, stats_text,
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#BBBBBB', alpha=0.95))

ax.legend(loc='lower right', framealpha=0.95, edgecolor='#CCCCCC',
          fontsize=10, fancybox=True)

ax.set_aspect('equal')
fig.tight_layout()

for ext in ('pdf', 'png'):
    fig.savefig(OUTDIR / f"fig7_loo_validation.{ext}", bbox_inches='tight', dpi=300)
print(f"Saved fig7_loo_validation.pdf and fig7_loo_validation.png")
plt.close(fig)

# ── Frontier model estimates with updated calibration ─────────────────────────
print(f"\n=== FRONTIER MODEL ESTIMATES (updated) ===")
frontier_targets = [
    'grok-4', 'gpt-5', 'claude-opus-4.6', 'gpt-4.1', 'gpt-4',
    'gpt-4o', 'gpt-3.5-turbo', 'claude-3-haiku'
]
for name in frontier_targets:
    m = next((x for x in summary if x["model"] == name), None)
    if m is None:
        continue
    acc = m["accuracy"]
    est_log = (acc - intercept) / slope
    est_B = 10**est_log
    lo = 10**(est_log - pi_half_log10)
    hi = 10**(est_log + pi_half_log10)
    print(f"  {name:25s}: acc={acc:.3f}, est={est_B:.0f}B [{lo:.0f}--{hi:.0f}B]")

print("\nDone!")
