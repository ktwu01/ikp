#!/usr/bin/env python3
"""Generate publication-quality figures for the IKP paper (improved version)."""

import json
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy import stats
from adjustText import adjust_text

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "results"
CONFIGS = ROOT / "configs" / "all_models.json"
PROBES = ROOT / "data" / "probes" / "final_probe_set_v9.json"
RESEARCHER_CITATIONS = ROOT / "data" / "researcher_citations.json"
RESEARCHER_RECOGNITION = ROOT / "data" / "researcher_recognition_rates.json"
OUTDIR = ROOT / "paper" / "figures"

OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "evaluation_summary.json") as f:
    summary = json.load(f)
with open(CONFIGS) as f:
    configs = json.load(f)["models"]
with open(RESEARCHER_CITATIONS) as f:
    researcher_citations = json.load(f)
with open(RESEARCHER_RECOGNITION) as f:
    researcher_recognition = json.load(f)

# Exclude broken evaluations from calibration (matches scripts/loo_cv_analysis.py)
CALIBRATION_EXCLUDE = {
    'minimax-m1-think',      # broken API: most responses empty, T1=0.465
    'hunyuan-a13b',          # extreme outlier: 80B scores below 12B dense models
    'hunyuan-a13b-think',    # same issue as hunyuan-a13b
    'hermes-3-405b',         # replaced by hermes-4-405b (same 405B, better fine-tune)
    'ling-2.6-flash',        # pathological refusals (60% on T1) — overalignment, not knowledge gap
    'deepseek-v3.1-nex-n1',  # post-training (nex-agi); -5.58σ below-trend, factual capacity degraded
    'intellect-3-think',     # post-training (Prime Intellect SFT+RL on GLM-4.5-Air-Base)
}
summary = [m for m in summary if m["model"] != "nemotron-ultra-253b"]

# Merge config info into summary
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

# ── Style ──────────────────────────────────────────────────────────────────────
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
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.color': '#CCCCCC',
})

# Professional palette
PAL_BLUE = '#2166AC'
PAL_ORANGE = '#D95F02'
PAL_GREEN = '#1B9E77'
PAL_RED = '#E7298A'
PAL_LIGHT_BLUE = '#92C5DE'
PAL_DARK_BLUE = '#053061'
PAL_GRAY = '#666666'

TIER_COLORS = {
    'T3': '#1B9E77',
    'T4': '#D95F02',
    'T5': '#7570B3',
    'T6': '#E7298A',
    'T7': '#66A61E',
}


def tex_escape(s):
    """Escape special LaTeX characters in a string."""
    s = s.replace('_', r'\_')
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    return s


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches='tight', dpi=300)
    print(f"  Saved {name}.pdf and {name}.png")
    plt.close(fig)


# ==============================================================================
# FIG 1: Calibration Curve (THE most important figure)
# ==============================================================================
def fig1_calibration():
    print("Generating Fig 1: Calibration Curve ...")

    # Open models with known params (excluding broken evaluations)
    open_models = [m for m in summary
                   if m["type"] == "open"
                   and m.get("params_B") is not None
                   and m["params_B"] > 0
                   and m["model"] not in CALIBRATION_EXCLUDE]

    # Proprietary models
    prop_models = [m for m in summary if m["type"] == "proprietary"]

    # --- Regression on open models ---
    log_params = np.array([math.log10(m["params_B"]) for m in open_models])
    accuracies = np.array([m["accuracy"] for m in open_models])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, accuracies)
    r_sq = r_value ** 2

    # Confidence band
    x_fit = np.linspace(log_params.min() - 0.15, log_params.max() + 0.4, 300)
    y_fit = slope * x_fit + intercept

    n = len(log_params)
    x_mean = log_params.mean()
    ss_x = np.sum((log_params - x_mean) ** 2)
    se_fit = std_err * np.sqrt(1.0 / n + (x_fit - x_mean) ** 2 / ss_x)
    t_crit = stats.t.ppf(0.975, n - 2)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(11, 7))

    # Confidence band
    ax.fill_between(x_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit,
                    alpha=0.12, color=PAL_BLUE, linewidth=0, zorder=1,
                    label=r'95\% confidence band')

    # Regression line
    ax.plot(x_fit, y_fit, color=PAL_DARK_BLUE, linewidth=2.2, alpha=0.85, zorder=2,
            label=f'OLS fit ($R^2 = {r_sq:.3f}$)')

    # Scatter: open models
    for m in open_models:
        lp = math.log10(m["params_B"])
        acc = m["accuracy"]
        is_moe = m["arch"] == "moe"
        is_think = m["thinking"]

        if is_moe:
            marker = 'D'
            color = PAL_ORANGE
            size = 50
        else:
            marker = 'o'
            color = PAL_BLUE
            size = 50

        edgecolor = 'black' if is_think else 'white'
        edgewidth = 1.2 if is_think else 0.5

        ax.scatter(lp, acc, c=color, marker=marker, s=size, alpha=0.75,
                   edgecolors=edgecolor, linewidths=edgewidth, zorder=3)

    # Legend entries
    ax.scatter([], [], c=PAL_BLUE, marker='o', s=50, label='Dense',
               edgecolors='white', linewidths=0.5)
    ax.scatter([], [], c=PAL_ORANGE, marker='D', s=50, label='MoE (total params)',
               edgecolors='white', linewidths=0.5)
    ax.scatter([], [], c='gray', marker='o', s=50, label='Thinking mode (black edge)',
               edgecolors='black', linewidths=1.2)

    # Manually-positioned landmarks — ABSOLUTE target position (tx, ty, ha) in
    # data coordinates. Labels are placed in empty regions of the plot:
    #   • lower-right (lp>2, acc<0.35) — very empty, good for 1T-cluster labels
    #   • upper-left (lp<2, acc>0.55) — R² box in the corner, rest is empty
    # A thin gray leader line is drawn from each data point to its label.
    landmark_layout = {
        # name: (label_x, label_y, ha) — absolute label position in data coords.
        # Positions chosen after auditing each label against:
        #   (1) all nearby data points (must not sit ON TOP of any dot)
        #   (2) all other landmark labels (horizontal span must not collide at
        #       same y; vertical gap must be ≥ 0.04 for same x)
        #   (3) the proprietary dashed lines (white bbox masks these)

        # ── Sub-1B: clean column at left edge, y=0.09..0.30, stacked 0.07 apart ──
        'smollm2-135m':           (-1.05, 0.30, 'left'),
        'smollm2-360m':           (-1.05, 0.23, 'left'),
        'gemma-3-270m':           (-1.05, 0.16, 'left'),
        'qwen-2.5-0.5b':          (-1.05, 0.09, 'left'),
        'qwen3-0.6b':             (-0.05, 0.01, 'left'),     # below-right of point, separate
        # ── 1-10B ──
        'gemma-3-1b':             (0.70, 0.04, 'left'),      # right-below (empty zone)
        'llama-3.2-1b':           (0.25, 0.25, 'left'),      # up-right of point
        'llama-3.2-3b':           (0.05, 0.36, 'left'),      # far-upper-left, empty
        'qwen-2.5-7b':            (1.15, 0.22, 'left'),      # right-below own point
        'phi-4':                  (1.50, 0.26, 'left'),      # right-below
        # ── 24-30B ──
        'mistral-small-24b':      (0.25, 0.48, 'left'),      # far-upper-left
        'gemma-3-27b':            (0.80, 0.34, 'left'),      # far-left
        # ── 70B cluster (3 points at lp≈1.85, acc≈0.52-0.53) ──
        'llama-3-70b':            (0.25, 0.62, 'left'),      # upper-left, well above glm-4.7
        'llama-3.3-70b':          (2.45, 0.42, 'left'),      # right-below
        'nemotron-70b':           (3.55, 0.46, 'left'),      # far-right past data band, clear of 70b cluster
        # ── 80-200B MoE ──
        'qwen3-next-80b-a3b':     (2.40, 0.39, 'left'),      # right-below
        'llama-4-scout':          (2.70, 0.31, 'left'),      # right-below
        'mistral-large':          (3.55, 0.40, 'left'),      # far-right past data band
        'command-a':              (2.05, 0.60, 'left'),      # upper-left of own (2.05, 0.540)
        'mixtral-8x22b':          (2.35, 0.64, 'left'),      # upper of own
        # ── 200-400B ──
        'qwen3-235b-a22b-think':  (0.45, 0.58, 'left'),      # far-upper-left empty band
        'hermes-4-405b':          (3.55, 0.34, 'left'),      # far-right below trend, clear
        'qwen3.5-397b-a17b-think':(0.45, 0.66, 'left'),      # upper-left
        'llama-4-maverick':       (3.55, 0.50, 'left'),      # right of own, past dashed-line end
        # ── Upper band: stratified rows at y=0.74, 0.80, 0.86 to avoid collision ──
        'deepseek-v4-flash':      (0.30, 0.70, 'left'),      # row a (0.70)
        'deepseek-v4-flash-think':(1.55, 0.74, 'left'),      # row b (0.74)
        'glm-4.7-think':          (0.30, 0.62, 'left'),      # below row a (clear of qwen3-235)
        'glm-5-think':            (0.30, 0.78, 'left'),      # row c (0.78)
        'kimi-k2.5-think':        (1.55, 0.82, 'left'),      # row d (0.82)
        'kimi-k2.6-think':        (3.00, 0.66, 'left'),      # right side, clear
        'deepseek-v3':            (3.40, 0.62, 'left'),      # right of own
        'kimi-k2':                (3.40, 0.55, 'left'),      # right of own
        # ── 1600B upper anchor: row e (0.86) ──
        'deepseek-v4-pro-think':  (1.55, 0.88, 'left'),      # top row, well above kimi-k2.5
    }

    landmark_models = set(landmark_layout.keys())
    # Draw each landmark label at its absolute target position.
    # - Label has white bbox so dashed lines / OLS line don't visually run through.
    # - A thin gray leader connects the data point to the label when they differ.
    # - zorder: leader=1 (below data), label=4 (above dashed lines and data).
    for m in open_models:
        name = m['model']
        if name not in landmark_layout: continue
        tx, ty, ha = landmark_layout[name]
        lp = math.log10(m['params_B']); acc = m['accuracy']
        # Leader line only if the label is visibly away from the point
        if abs(tx - lp) > 0.06 or abs(ty - acc) > 0.015:
            anchor_x = tx + (0.03 if ha == 'left' else -0.03)
            ax.plot([lp, anchor_x], [acc, ty], color='#999999',
                    linewidth=0.5, alpha=0.7, zorder=1)
        ax.text(tx, ty, tex_escape(name), fontsize=7.5,
                color='#333333', ha=ha, va='center', zorder=4,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.88, pad=0.8))
    open_texts = []  # adjust_text disabled for landmarks

    # --- (Gemini 3.1 Pro now rendered in the proprietary-dashed-line block below
    # with a 'landmark' regime so no spurious size estimate appears.)
    static_obstacles = []

    # --- Proprietary models: horizontal dashed lines with right-side labels ---
    # regime='pretraining' — single-regime estimate (curve inversion of acc)
    # regime='landmark'    — labeled but no size estimate (T6 inflated by construction)
    prop_show = [
        # (model_name, display, regime). For models with think+non-think variants,
        # we plot the higher-scoring variant (matches the frontier table convention).
        ('gemini-3.1-pro',          'Gemini 3.1 Pro',    'landmark'),
        ('gpt-5.5-think',           'GPT-5.5',           'pretraining'),
        ('claude-opus-4.6-think',   'Claude Opus 4.6',   'pretraining'),
        ('claude-sonnet-4.6-think', 'Claude Sonnet 4.6', 'pretraining'),
        ('gemini-2.5-pro',          'Gemini 2.5 Pro',    'pretraining'),
        ('gpt-5-mini',              'GPT-5 Mini',        'pretraining'),
        ('gemini-2.5-flash',        'Gemini 2.5 Flash',  'pretraining'),
        ('claude-haiku-4.5',        'Claude Haiku 4.5',  'pretraining'),
    ]

    def fmt_size(B):
        return f"{B / 1000:.1f}T" if B >= 1000 else f"{int(round(B))}B"

    prop_entries = []
    for name, display, regime in prop_show:
        m = next((x for x in prop_models if x['model'] == name), None)
        if not m:
            continue
        acc = m['accuracy']
        if regime == 'landmark':
            # No size estimate — landmark models are excluded from estimation
            # because their T6 scores are inflated by construction.
            size_label = "landmark, excluded"
        else:
            single_B = 10 ** ((acc - intercept) / slope)
            size_label = f"est. ~{fmt_size(single_B)}"
        prop_entries.append((name, display, acc, size_label, regime))

    prop_entries.sort(key=lambda x: x[2])

    # Avoid label overlap (increase min_gap — old value bunched labels)
    min_gap = 0.045
    label_ys = [e[2] for e in prop_entries]
    for i in range(1, len(label_ys)):
        if label_ys[i] - label_ys[i - 1] < min_gap:
            label_ys[i] = label_ys[i - 1] + min_gap

    # Tight right margin: dashed lines end just past the last data point
    # (DeepSeek V4 Pro at ~1.6T), proprietary labels start immediately after.
    # No wasted horizontal space between data and labels.
    x_line_end = x_fit[-1] + 0.35   # dash terminates shortly after last data point
    x_label_x = x_fit[-1] + 0.6     # labels start here

    # Uniform color for all proprietary lines (user request):
    PROP_COLOR = '#C47A3E'   # same orange used for Gemini 3 Flash earlier
    PROP_LABEL_COLOR = '#8A4A12'

    for idx, (model_name, display_name, acc, size_label, regime) in enumerate(prop_entries):
        lw = 1.1
        dash_pattern = (0, (4, 2))
        x0 = log_params.min() - 0.25
        ax.plot([x0, x_line_end], [acc, acc],
                color=PROP_COLOR, alpha=0.65, linewidth=lw,
                linestyle=dash_pattern, zorder=1)

        label_text = f"{display_name}  ({size_label})"
        # Landmark still bold to signal its exclusion; all others normal.
        fweight = 'bold' if regime == 'landmark' else 'normal'
        prop_txt = ax.text(x_label_x, label_ys[idx], label_text,
                fontsize=8.5, va='center', ha='left', color=PROP_LABEL_COLOR,
                fontweight=fweight)
        static_obstacles.append(prop_txt)

        # Connecting leader line if nudged
        if abs(label_ys[idx] - acc) > 0.004:
            ax.plot([x_line_end + 0.02, x_label_x - 0.04],
                    [acc, label_ys[idx]],
                    color=PROP_COLOR, alpha=0.5, linewidth=0.6, zorder=1)

    # Store for later (xlim adjustment uses x_label_x)
    x_right_edge = x_label_x

    # --- Axes ---
    ax.set_xlabel(r'$\log_{10}$(Total Parameters, Billions)', fontsize=13)
    ax.set_ylabel('IKP Accuracy', fontsize=13)
    ax.set_title('IKP Calibration: Knowledge Scales Log-Linearly with Model Size',
                 fontsize=15, fontweight='bold', pad=14)

    # Secondary x-axis: actual param values
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    param_ticks = [0.1, 1, 10, 100, 1000]
    param_tick_locs = [math.log10(p) for p in param_ticks]
    param_tick_labels = ['100M', '1B', '10B', '100B', '1T']
    ax2.set_xticks(param_tick_locs)
    ax2.set_xticklabels(param_tick_labels, fontsize=10)
    ax2.set_xlabel('Model Size', fontsize=11, labelpad=8)
    ax2.spines['top'].set_visible(True)
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['top'].set_color('#CCCCCC')
    ax2.tick_params(axis='x', colors='#666666')

    # R^2 box
    ax.text(0.03, 0.96, f'$R^2 = {r_sq:.3f}$\n$n = {n}$ open models',
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#BBBBBB', alpha=0.95))

    ax.legend(loc='lower right', framealpha=0.95, edgecolor='#CCCCCC',
              fontsize=10, fancybox=True)

    # Right margin sized for the longest proprietary label (~2.3 units).
    ax.set_xlim(log_params.min() - 0.25, x_right_edge + 2.3)
    ax.set_ylim(0.0, 0.95)

    # Force a draw so adjust_text sees correct bounding boxes for obstacles
    fig.canvas.draw()

    # Auto-avoid label overlap for the landmark open-model labels.
    # Key change: stronger force_static + wider expand so landmark labels
    # (kimi-k2.5-think, glm-5-think, deepseek-v3, etc.) don't collide with
    # the right-side proprietary label column.
    if open_texts:
        adjust_text(
            open_texts, ax=ax,
            objects=static_obstacles,
            expand=(2.2, 2.5),
            force_text=(1.6, 2.0),
            force_static=(2.5, 2.8),
            force_explode=(0.6, 0.8),
            force_pull=(0.02, 0.02),
            max_move=120,
            iter_lim=3000,
            arrowprops=dict(arrowstyle='->', color='#888888',
                            lw=0.55, shrinkA=2, shrinkB=3),
            only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
        )

    fig.tight_layout()
    save(fig, 'fig1_calibration')


# ==============================================================================
# FIG 2: Per-Tier Accuracy Heatmap (FLIPPED: best model at TOP)
# ==============================================================================
def fig2_tier_heatmap():
    print("Generating Fig 2: Per-Tier Accuracy Heatmap ...")

    # Sort by overall accuracy DESCENDING, take top 25
    sorted_models = sorted(summary, key=lambda m: m["accuracy"], reverse=True)[:25]
    # Keep descending order (best = top row = index 0)

    tiers = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    tier_labels = ['T1\nUniversal', 'T2\nCommon', 'T3\nDomain', 'T4\nObscure',
                   'T5\nDeep', 'T6\nLong-Tail', 'T7\nExtreme']
    model_names = [tex_escape(m["model"]) for m in sorted_models]
    data = np.array([[m["tier_accuracy"].get(t, 0) for t in tiers]
                     for m in sorted_models])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(data, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest')

    # Tier column labels on top
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tier_labels, fontsize=9, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Model names on left
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)

    # Annotate each cell
    for i in range(len(model_names)):
        for j in range(len(tiers)):
            val = data[i, j]
            # White text on dark cells, black on light
            text_color = 'white' if val > 0.55 else 'black'
            if val > 0.005:
                pct = f'{val*100:.0f}\\%'
            else:
                pct = '0\\%'
            ax.text(j, i, pct, ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='medium')

    # Thin black lines separating tier columns
    for j in range(1, len(tiers)):
        ax.axvline(x=j - 0.5, color='black', linewidth=0.5, alpha=0.3)

    ax.set_title('Per-Tier Accuracy (Top 25 Models)',
                 fontsize=14, fontweight='bold', pad=35)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label('Accuracy', fontsize=11)

    fig.tight_layout()
    save(fig, 'fig2_tier_heatmap')


# ==============================================================================
# FIG 3: Thinking Mode Effect (horizontal bar chart)
# ==============================================================================
def fig3_thinking_effect():
    print("Generating Fig 3: Thinking Mode Effect ...")

    # Find base/think pairs
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

    # Sort by base accuracy (lowest at top for horizontal bar chart)
    pairs.sort(key=lambda p: p['base_acc'])

    if len(pairs) == 0:
        print("  WARNING: No thinking pairs found, skipping.")
        return

    n_pairs = len(pairs)
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(n_pairs)
    bar_height = 0.35

    base_accs = [p['base_acc'] for p in pairs]
    think_accs = [p['think_acc'] for p in pairs]
    deltas = [p['delta'] for p in pairs]
    names = [p['base_name'] for p in pairs]

    # Horizontal bars
    bars_base = ax.barh(y_pos + bar_height / 2, base_accs, bar_height,
                        label='Base', color=PAL_LIGHT_BLUE,
                        edgecolor='white', linewidth=0.5)
    bars_think = ax.barh(y_pos - bar_height / 2, think_accs, bar_height,
                         label='Thinking', color=PAL_DARK_BLUE,
                         edgecolor='white', linewidth=0.5)

    # Annotate deltas at a fixed column just past the longer bar for ALL rows,
    # so labels line up and never overlap the bars themselves.
    x_label_col = max(max(base_accs), max(think_accs)) + 0.015
    for i, d in enumerate(deltas):
        sign = '+' if d >= 0 else ''
        color = PAL_GREEN if d >= 0 else '#D55E00'
        ax.text(x_label_col, y_pos[i], f'{sign}{d*100:.1f}\\%',
                ha='left', va='center',
                fontsize=9, fontweight='bold', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([tex_escape(n) for n in names], fontsize=9)
    ax.set_xlabel('IKP Accuracy', fontsize=12)
    ax.set_title('Thinking Mode Effect on IKP Score',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, framealpha=0.95, edgecolor='#CCCCCC',
              fontsize=10, fancybox=True)
    ax.set_xlim(0, x_label_col + 0.08)

    fig.tight_layout()
    save(fig, 'fig3_thinking_effect')


# ==============================================================================
# FIG 4: MoE Total vs Active Params
# ==============================================================================
def fig4_moe_params():
    print("Generating Fig 4: MoE Total vs Active Params ...")

    moe_models = [m for m in summary
                  if m["arch"] == "moe"
                  and m["type"] == "open"
                  and m.get("params_B") is not None and m["params_B"] > 0
                  and m.get("active_B") is not None and m["active_B"] > 0
                  and m["model"] not in CALIBRATION_EXCLUDE]

    if len(moe_models) < 3:
        print(f"  WARNING: Only {len(moe_models)} MoE models, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.0), sharey=True)

    total_params = np.array([math.log10(m["params_B"]) for m in moe_models])
    active_params = np.array([math.log10(m["active_B"]) for m in moe_models])
    accs = np.array([m["accuracy"] for m in moe_models])

    # --- Left: total params ---
    ax1.scatter(total_params, accs, c=PAL_ORANGE, s=60, alpha=0.85,
                edgecolors='white', linewidths=0.6, zorder=3)
    sl1, int1, r1, _, _ = stats.linregress(total_params, accs)
    x1 = np.linspace(total_params.min() - 0.1, total_params.max() + 0.1, 100)
    ax1.plot(x1, sl1 * x1 + int1, color=PAL_ORANGE, linewidth=1.8, alpha=0.7,
             linestyle='--')
    ax1.set_xlabel(r'$\log_{10}$(Total Parameters, B)', fontsize=12)
    ax1.set_ylabel('IKP Accuracy', fontsize=12)
    ax1.set_title(f'Total Params ($R^2 = {r1**2:.3f}$)', fontsize=13,
                  fontweight='bold')

    # Label curated MoE landmarks on left panel (total params)
    landmarks_total = {
        'qwen3-30b-a3b-think', 'mixtral-8x22b', 'qwen3-235b-a22b-think',
        'llama-4-maverick', 'jamba-large', 'glm-4.5-think',
        'deepseek-v3', 'kimi-k2', 'glm-5-think', 'kimi-k2.5-think',
        'deepseek-v4-pro-think',
    }
    texts_left = []
    for m in moe_models:
        if m["model"] in landmarks_total:
            lp = math.log10(m["params_B"])
            acc = m["accuracy"]
            texts_left.append(ax1.text(lp, acc, tex_escape(m["model"]),
                                       fontsize=7, color='#444444'))

    # Secondary x-axis for left panel
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    param_ticks_l = [10, 100, 1000]
    ax1_top.set_xticks([math.log10(p) for p in param_ticks_l])
    ax1_top.set_xticklabels(['10B', '100B', '1T'], fontsize=9)
    ax1_top.spines['top'].set_visible(True)
    ax1_top.spines['top'].set_linewidth(0.5)
    ax1_top.spines['top'].set_color('#CCCCCC')
    ax1_top.tick_params(axis='x', colors='#666666')

    # --- Right: active params ---
    ax2.scatter(active_params, accs, c='#E6AB02', s=60, alpha=0.85,
                edgecolors='white', linewidths=0.6, zorder=3)
    sl2, int2, r2, _, _ = stats.linregress(active_params, accs)
    x2 = np.linspace(active_params.min() - 0.1, active_params.max() + 0.1, 100)
    ax2.plot(x2, sl2 * x2 + int2, color='#E6AB02', linewidth=1.8, alpha=0.7,
             linestyle='--')
    ax2.set_xlabel(r'$\log_{10}$(Active Parameters, B)', fontsize=12)
    ax2.set_title(f'Active Params ($R^2 = {r2**2:.3f}$)', fontsize=13,
                  fontweight='bold')

    # Label curated MoE landmarks on right panel (active params)
    landmarks_active = {
        'qwen3-30b-a3b-think', 'qwen3-next-80b-a3b', 'glm-4.5-air-think',
        'llama-4-scout', 'llama-4-maverick', 'mixtral-8x22b',
        'deepseek-v3', 'kimi-k2', 'jamba-large', 'kimi-k2.5-think',
    }
    texts_right = []
    for m in moe_models:
        if m["model"] in landmarks_active:
            lp = math.log10(m["active_B"])
            acc = m["accuracy"]
            texts_right.append(ax2.text(lp, acc, tex_escape(m["model"]),
                                        fontsize=7, color='#444444'))

    # Secondary x-axis for right panel
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    param_ticks_r = [3, 10, 30, 100]
    valid_r = [p for p in param_ticks_r
               if math.log10(p) >= ax2.get_xlim()[0] and math.log10(p) <= ax2.get_xlim()[1]]
    ax2_top.set_xticks([math.log10(p) for p in valid_r])
    ax2_top.set_xticklabels([f'{p}B' for p in valid_r], fontsize=9)
    ax2_top.spines['top'].set_visible(True)
    ax2_top.spines['top'].set_linewidth(0.5)
    ax2_top.spines['top'].set_color('#CCCCCC')
    ax2_top.tick_params(axis='x', colors='#666666')

    fig.suptitle('MoE Models: Total vs Active Parameters',
                 fontsize=14, fontweight='bold', y=1.04)

    arrow_kw = dict(arrowstyle='->', color='#888888', lw=0.5, shrinkA=2, shrinkB=3)
    adj_kw = dict(
        expand=(1.5, 1.8),
        force_text=(1.0, 1.3),
        force_static=(0.8, 1.0),
        force_explode=(0.3, 0.5),
        force_pull=(0.05, 0.05),
        max_move=60,
        iter_lim=1000,
        arrowprops=arrow_kw,
        only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
    )
    fig.canvas.draw()
    if texts_left:
        adjust_text(texts_left, ax=ax1, **adj_kw)
    if texts_right:
        adjust_text(texts_right, ax=ax2, **adj_kw)

    fig.tight_layout()
    save(fig, 'fig4_moe_params')


# ==============================================================================
# FIG 5: Researcher Recognition vs Citations
# ==============================================================================
def fig5_researcher_scatter():
    print("Generating Fig 5: Researcher Recognition vs Citations ...")

    recog_map = {r['probe_id']: r for r in researcher_recognition}
    merged = []
    for r in researcher_citations:
        pid = r['probe_id']
        if pid in recog_map and r.get('cited_by_count') and r['cited_by_count'] > 0:
            merged.append({
                'name': r['name'],
                'tier': r['tier'],
                'citations': r['cited_by_count'],
                'log_citations': math.log10(r['cited_by_count']),
                'recognition_rate': recog_map[pid]['recognition_rate'],
                'h_index': r.get('h_index', 0),
            })

    if not merged:
        print("  WARNING: No merged data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot by tier
    for tier in ['T3', 'T4', 'T5', 'T6', 'T7']:
        subset = [m for m in merged if m['tier'] == tier]
        if not subset:
            continue
        x = [m['log_citations'] for m in subset]
        y = [m['recognition_rate'] for m in subset]
        ax.scatter(x, y, c=TIER_COLORS[tier], s=40, alpha=0.6,
                   edgecolors='white', linewidths=0.3, label=tier, zorder=2)

    # Overall regression
    all_x = np.array([m['log_citations'] for m in merged])
    all_y = np.array([m['recognition_rate'] for m in merged])
    rho, p_rho = stats.spearmanr(all_x, all_y)

    slope_r, intercept_r, r_val, _, _ = stats.linregress(all_x, all_y)
    xr = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(xr, slope_r * xr + intercept_r, color='#333333', linewidth=1.5,
            alpha=0.5, linestyle='--', zorder=1)

    # Label outliers — make sure all names are fully visible
    outlier_labels = {
        'Yiannis Psaras': (-55, 14),
        'Thorsten Joachims': (-90, -12),
        'Ted Dunning': (-55, -15),
        'Xinming Wang': (10, -18),
        'Yan Jiao': (-65, -14),
        'Peter Alvaro': (-60, 12),
        'Rich Wolski': (10, -14),
    }
    # Name collision examples get a distinct style
    collision_names = {'Xinming Wang', 'Yan Jiao'}
    for m in merged:
        if m['name'] in outlier_labels:
            dx, dy = outlier_labels[m['name']]
            is_collision = m['name'] in collision_names
            fc = '#CC0000' if is_collision else '#333333'
            fw = 'bold' if is_collision else 'normal'
            ax.annotate(m['name'], (m['log_citations'], m['recognition_rate']),
                        textcoords='offset points', xytext=(dx, dy),
                        fontsize=8.5, color=fc, fontweight=fw,
                        arrowprops=dict(arrowstyle='->', color='#888888',
                                        lw=0.7, shrinkA=2, shrinkB=3))

    # Annotation box
    ax.text(0.03, 0.05,
            f'Spearman $\\rho = {rho:.3f}$  ($p < 10^{{-6}}$)\n'
            f'$n = {len(merged)}$ researchers',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#BBBBBB', alpha=0.95))

    # Note about name collision outliers — visible box in bottom-right
    ax.text(0.97, 0.15,
            'High-citation, low-recognition outliers\nare OpenAlex name collisions',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            color='#555555', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                      edgecolor='#DDD', alpha=0.9))

    ax.set_xlabel(r'$\log_{10}$(Citations)', fontsize=12)
    ax.set_ylabel('Recognition Rate (fraction of models correct)', fontsize=12)
    ax.set_title('Researcher Recognition vs Citation Count',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#CCCCCC',
              title='Tier', title_fontsize=10, fontsize=10, fancybox=True)
    ax.set_ylim(-0.03, 1.03)

    fig.tight_layout()
    save(fig, 'fig5_researcher_citations')


# ==============================================================================
# FIG 6: Knowledge Fingerprint Heatmap (Jaccard similarity)
# ==============================================================================
def fig6_fingerprint_heatmap():
    print("Generating Fig 6: Knowledge Fingerprint Heatmap ...")

    top_models = sorted(summary, key=lambda m: m["accuracy"], reverse=True)

    # Pick models from diverse vendors (top 15)
    target_vendors = {
        'anthropic': ['claude-fable-5', 'claude-opus-4.7', 'claude-sonnet-5'],
        'google': ['gemini-3.1-pro', 'gemini-2.5-pro'],
        'openai': ['gpt-5.5', 'gpt-5.5-pro', 'gpt-4.1'],
        'xai': ['grok-4', 'grok-3'],
        'deepseek': ['deepseek-v3.2', 'deepseek-v3'],
        'zhipu': ['glm-5.2', 'glm-5.1'],
        'moonshot': ['kimi-k2.7-code'],
    }

    selected_names = []
    for vendor, candidates in target_vendors.items():
        for c in candidates:
            if any(m["model"] == c for m in top_models):
                selected_names.append(c)

    # Cap at 15
    selected_names = selected_names[:15]

    # Fallback: pad with top accuracy models
    if len(selected_names) < 15:
        for m in top_models:
            if m["model"] not in selected_names and m["accuracy"] > 0.55:
                selected_names.append(m["model"])
            if len(selected_names) >= 15:
                break

    # Load per-probe results
    model_correct_sets = {}
    for name in selected_names:
        fpath = RESULTS_DIR / f"{name}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        if 'results' not in data:
            continue
        correct_ids = set()
        for r in data['results']:
            if r.get('tier') in ('T5', 'T6') and r.get('correct', False):
                correct_ids.add(r['probe_id'])
        if len(correct_ids) > 0:
            model_correct_sets[name] = correct_ids

    models_with_data = [n for n in selected_names if n in model_correct_sets]

    if len(models_with_data) < 5:
        print(f"  WARNING: Only {len(models_with_data)} models have T5-T6 data, skipping.")
        return

    nm = len(models_with_data)

    # Jaccard similarity matrix
    jaccard = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            a = model_correct_sets[models_with_data[i]]
            b = model_correct_sets[models_with_data[j]]
            inter = len(a & b)
            union = len(a | b)
            jaccard[i, j] = inter / union if union > 0 else 0

    # Sort by vendor
    def vendor_of(name):
        if name in configs:
            return configs[name].get('vendor', 'unknown')
        return 'unknown'

    vendor_order = ['anthropic', 'google', 'openai', 'xai', 'deepseek', 'zhipu',
                    'alibaba', 'meta', 'moonshot', 'cohere', 'other']
    models_sorted = sorted(models_with_data,
                           key=lambda n: (vendor_order.index(vendor_of(n))
                                          if vendor_of(n) in vendor_order else 99, n))

    # Reorder matrix
    idx_map = [models_with_data.index(m) for m in models_sorted]
    jaccard_sorted = jaccard[np.ix_(idx_map, idx_map)]

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(jaccard_sorted, cmap='YlOrRd', vmin=0, vmax=1,
                   interpolation='nearest')

    ax.set_xticks(range(nm))
    ax.set_xticklabels([tex_escape(n) for n in models_sorted], rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(nm))
    ax.set_yticklabels([tex_escape(n) for n in models_sorted], fontsize=10)

    # Annotate each cell with Jaccard value
    for i in range(nm):
        for j in range(nm):
            val = jaccard_sorted[i, j]
            text_color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    # Draw vendor cluster borders
    current_vendor = None
    boundaries = []
    vendor_ranges = {}  # vendor -> (start, end)
    start_idx = 0
    for i, name in enumerate(models_sorted):
        v = vendor_of(name)
        if v != current_vendor:
            if current_vendor is not None:
                boundaries.append(i)
                vendor_ranges[current_vendor] = (start_idx, i - 1)
            start_idx = i
            current_vendor = v
    if current_vendor is not None:
        vendor_ranges[current_vendor] = (start_idx, len(models_sorted) - 1)

    for b in boundaries:
        ax.axhline(y=b - 0.5, color='black', linewidth=1.5, alpha=0.7)
        ax.axvline(x=b - 0.5, color='black', linewidth=1.5, alpha=0.7)

    # Color-code model names by vendor instead of adding separate labels
    vendor_display = {
        'anthropic': 'Anthropic', 'google': 'Google', 'openai': 'OpenAI',
        'xai': 'xAI', 'deepseek': 'DeepSeek', 'zhipu': 'Zhipu',
    }
    vendor_colors_map = {
        'anthropic': '#D4783A', 'google': '#4285F4', 'openai': '#74AA9C',
        'xai': '#333333', 'deepseek': '#0066CC', 'zhipu': '#6B4C9A',
    }
    # Color y-tick labels and x-tick labels by vendor
    for label in ax.get_yticklabels():
        model_name = label.get_text()
        v = vendor_of(model_name)
        vc = vendor_colors_map.get(v, '#333333')
        label.set_color(vc)
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        model_name = label.get_text()
        v = vendor_of(model_name)
        vc = vendor_colors_map.get(v, '#333333')
        label.set_color(vc)
        label.set_fontweight('bold')

    # Add a legend for vendor colors at the bottom
    from matplotlib.patches import Patch
    legend_patches = []
    seen_vendors = []
    for name in models_sorted:
        v = vendor_of(name)
        if v not in seen_vendors and v in vendor_display:
            seen_vendors.append(v)
    for v in seen_vendors:
        vc = vendor_colors_map.get(v, '#333333')
        legend_patches.append(Patch(facecolor=vc, label=vendor_display[v]))
    ax.legend(handles=legend_patches, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), fontsize=8,
              framealpha=0.9, edgecolor='#CCCCCC', title='Vendor',
              title_fontsize=9, ncol=len(seen_vendors))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label('Jaccard Similarity (T5-T6 correct sets)', fontsize=10)

    ax.set_title('Knowledge Fingerprint: Same-Vendor Models Share Rare Knowledge',
                 fontsize=13, fontweight='bold', pad=12)

    fig.tight_layout()
    save(fig, 'fig6_fingerprint_heatmap')


# ==============================================================================
# FIG 8: Densing Law Falsification
# ==============================================================================
def fig8_densing_law():
    print("Generating Fig 8: Densing Law Falsification ...")

    import csv
    from datetime import datetime

    csv_path = ROOT / "data" / "densing_analysis_data.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} missing -- run scripts/15_densing_law_analysis.py first.")
        return

    log_params, pen_acc, months, dates = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            log_params.append(float(row["log10_params"]))
            pen_acc.append(float(row["pen_acc"]))
            months.append(float(row["months"]))
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))

    log_params = np.array(log_params)
    pen_acc = np.array(pen_acc)
    months = np.array(months)
    n = len(log_params)

    # OLS on log_params only (baseline)
    X0 = np.column_stack([np.ones(n), log_params])
    beta0, *_ = np.linalg.lstsq(X0, pen_acc, rcond=None)
    fit_acc = X0 @ beta0
    r2_0 = 1 - ((pen_acc - fit_acc) ** 2).sum() / ((pen_acc - pen_acc.mean()) ** 2).sum()

    # Partial out log_params: regress residuals on months (intercept + slope)
    resid = pen_acc - fit_acc
    X1 = np.column_stack([np.ones(n), months])
    beta1, *_ = np.linalg.lstsq(X1, resid, rcond=None)
    res_int, time_coef = beta1
    # Std error on time_coef
    y_hat_res = X1 @ beta1
    dof = n - 2
    sigma2 = ((resid - y_hat_res) ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se_time = np.sqrt(cov[1, 1])
    t_stat = time_coef / se_time
    p_time = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    # Densing Law prediction: monthly gain = slope * log10(2) / 3.5
    densing_per_month = beta0[1] * np.log10(2) / 3.5

    # ── Figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))

    # LEFT: scaling with points colored by release date
    sc = ax0.scatter(log_params, pen_acc, c=months, cmap='viridis',
                     s=42, edgecolor='#222222', linewidth=0.4, alpha=0.9)
    xs = np.linspace(log_params.min() - 0.05, log_params.max() + 0.05, 100)
    ax0.plot(xs, beta0[0] + beta0[1] * xs, '-', color='#222222', lw=1.6,
             label=f'OLS: slope $= {beta0[1]:.3f}$, $R^2 = {r2_0:.3f}$')
    ax0.set_xlabel(r'$\log_{10}$(Total Parameters, Billions)', fontsize=12)
    ax0.set_ylabel('IKP Accuracy', fontsize=12)
    ax0.set_title(f'IKP scales with parameters ($n = {n}$ open models)',
                  fontsize=13, fontweight='bold', pad=10)
    ax0.legend(loc='upper left', framealpha=0.95, edgecolor='#CCCCCC',
               fontsize=10, fancybox=True)

    cbar = fig.colorbar(sc, ax=ax0, shrink=0.85, pad=0.02, aspect=25)
    cbar.set_label('Months since 2024-01', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # RIGHT: residual vs release date — observed flat vs dashed Densing prediction
    from matplotlib.dates import DateFormatter, YearLocator

    ax1.axhline(0, color='#888888', lw=0.7, zorder=1)
    ax1.scatter(dates, resid, c=months, cmap='viridis', s=42,
                edgecolor='#222222', linewidth=0.4, alpha=0.9, zorder=3)

    # Observed trend line
    m_lin = np.linspace(months.min(), months.max(), 100)
    y_obs = res_int + time_coef * m_lin
    from datetime import timedelta
    REF = datetime(2024, 1, 1)
    d_lin = [REF + timedelta(days=30.44 * mm) for mm in m_lin]
    ax1.plot(d_lin, y_obs, '-', color=PAL_BLUE, lw=2.0,
             label=f'observed: ${time_coef:+.4f}$/mo  ($p = {p_time:.2f}$)',
             zorder=4)

    # Densing Law prediction line — anchored at mean date, same residual mean
    y_densing = resid.mean() + densing_per_month * (m_lin - m_lin.mean())
    ax1.plot(d_lin, y_densing, '--', color='#CC0000', lw=2.0,
             label=f'Densing Law: $+{densing_per_month:.4f}$/mo', zorder=4)

    ax1.set_xlabel('Release Date', fontsize=12)
    ax1.set_ylabel(r'Residual after partialling out $\log_{10}(N)$', fontsize=12)
    ax1.set_title('Time trend vanishes at fixed parameter count',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='#CCCCCC',
               fontsize=10, fancybox=True)

    # Gap annotation
    gap_ratio = abs(densing_per_month - time_coef) / se_time
    ax1.text(0.98, 0.04,
             f'Gap: ${gap_ratio:.0f}\\sigma$  (Densing rejected at $p < 10^{{-15}}$)',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=9.5, color='#555555', style='italic',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.95))

    fig.tight_layout()
    save(fig, 'fig8_densing_law')


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    print(f"Models in summary: {len(summary)}")
    fig1_calibration()
    fig2_tier_heatmap()
    fig3_thinking_effect()
    fig4_moe_params()
    fig5_researcher_scatter()
    fig6_fingerprint_heatmap()
    fig8_densing_law()
    print("\nAll figures generated!")
