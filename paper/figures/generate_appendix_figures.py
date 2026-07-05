#!/usr/bin/env python3
"""Generate appendix figures for the IKP paper (A1-A4)."""

import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "results"
CONFIGS = ROOT / "configs" / "all_models.json"
OUTDIR = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "evaluation_summary.json") as f:
    summary = json.load(f)
with open(CONFIGS) as f:
    configs = json.load(f)["models"]

# Exclude nemotron-ultra-253b (0% accuracy, API failure)
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

N_MODELS = len(summary)

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


def tex_escape(s):
    """Escape special LaTeX characters in a string."""
    s = s.replace('_', r'\_')
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    return s

# Professional palette (matching main figures)
PAL_BLUE = '#2166AC'
PAL_ORANGE = '#D95F02'
PAL_GREEN = '#1B9E77'
PAL_RED = '#E7298A'
PAL_LIGHT_BLUE = '#92C5DE'
PAL_DARK_BLUE = '#053061'
PAL_GRAY = '#666666'


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches='tight', dpi=300)
    print(f"  Saved {name}.pdf and {name}.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig A1: Per-Tier Accuracy Distribution (Box Plots)
# ══════════════════════════════════════════════════════════════════════════════
def fig_a1_tier_boxplots():
    print("Generating Fig A1: Per-Tier Accuracy Distribution...")

    tiers = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    # Blue gradient from light (T1) to dark (T7)
    blue_gradient = ['#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6',
                     '#4292C6', '#2171B5', '#084594']

    # Collect per-tier accuracy data
    tier_data = {t: [] for t in tiers}
    for m in summary:
        ta = m.get("tier_accuracy", {})
        for t in tiers:
            if t in ta:
                tier_data[t].append(ta[t])

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(1, len(tiers) + 1)
    data_arrays = [tier_data[t] for t in tiers]

    # Draw box plots
    bp = ax.boxplot(data_arrays, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='#444444', linewidth=1.0),
                    capprops=dict(color='#444444', linewidth=1.0))

    for patch, color in zip(bp['boxes'], blue_gradient):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(0.8)
        patch.set_alpha(0.85)

    # Overlay jittered individual points
    rng = np.random.RandomState(42)
    for i, (t, vals) in enumerate(zip(tiers, data_arrays)):
        jitter = rng.normal(0, 0.08, size=len(vals))
        ax.scatter(positions[i] + jitter, vals,
                   c=blue_gradient[i], edgecolors='#333333',
                   linewidths=0.3, s=12, alpha=0.45, zorder=5)

    # Annotate median and IQR below the box for compressed tiers, above otherwise
    for i, (t, vals) in enumerate(zip(tiers, data_arrays)):
        arr = np.array(vals)
        median = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        whisker_bottom = max(q1 - 1.5 * iqr, arr.min())

        # For T1/T2 (compressed at top), place label below the whisker
        if t in ('T1', 'T2'):
            y_annot = whisker_bottom - 0.04
            va = 'top'
        # For T6/T7 (compressed at bottom), place above the whisker
        elif t in ('T6', 'T7'):
            whisker_top = min(q3 + 1.5 * iqr, arr.max())
            y_annot = whisker_top + 0.03
            va = 'bottom'
        else:
            whisker_top = min(q3 + 1.5 * iqr, arr.max())
            y_annot = whisker_top + 0.03
            if y_annot > 1.05:
                y_annot = whisker_bottom - 0.04
                va = 'top'
            else:
                va = 'bottom'

        ax.annotate(f'Med={median:.2f}\nIQR={iqr:.2f}',
                    xy=(positions[i], y_annot),
                    fontsize=7.5, ha='center', va=va,
                    color='#333333',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='#CCCCCC', alpha=0.85))

    ax.set_xlabel('Tier')
    ax.set_ylabel('Penalized Accuracy')
    ax.set_xticks(positions)
    ax.set_xticklabels(tiers)
    ax.set_ylim(-0.08, 1.15)
    ax.set_title(f'Accuracy Distribution by Tier Across {N_MODELS} Models',
                 fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    save(fig, 'fig_a1_tier_boxplots')


# ══════════════════════════════════════════════════════════════════════════════
# Fig A2: Hallucination Rate by Vendor
# ══════════════════════════════════════════════════════════════════════════════
def fig_a2_vendor_hallucination():
    print("Generating Fig A2: Hallucination Rate by Vendor...")

    # Calculate hallucination rate on T5+T6+T7 per model
    vendor_models = defaultdict(list)
    for m in summary:
        vendor = m.get("vendor", "unknown")
        ts = m.get("tier_stats", {})
        wrong = 0
        correct = 0
        refusal = 0
        for t in ['T5', 'T6', 'T7']:
            if t in ts:
                wrong += ts[t].get("wrong", 0)
                correct += ts[t].get("correct", 0)
                refusal += ts[t].get("refusal", 0)
        denom = wrong + correct + refusal
        if denom > 0:
            hall_rate = wrong / denom
        else:
            hall_rate = 0.0
        vendor_models[vendor].append((m["model"], hall_rate))

    # Filter: only vendors with 2+ models
    vendor_stats = {}
    for vendor, models in vendor_models.items():
        if len(models) >= 2:
            rates = [r for _, r in models]
            vendor_stats[vendor] = {
                'mean': np.mean(rates),
                'models': models,
            }

    # Sort by mean hallucination rate (highest at top for horizontal bars)
    sorted_vendors = sorted(vendor_stats.keys(),
                            key=lambda v: vendor_stats[v]['mean'])

    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = np.arange(len(sorted_vendors))
    means = [vendor_stats[v]['mean'] for v in sorted_vendors]

    # Color by rate
    colors = []
    for rate in means:
        if rate > 0.60:
            colors.append('#D32F2F')  # red
        elif rate > 0.30:
            colors.append('#F57C00')  # orange
        else:
            colors.append('#388E3C')  # green

    bars = ax.barh(y_positions, means, height=0.6, color=colors, alpha=0.80,
                   edgecolor='#333333', linewidth=0.6)

    # Overlay individual model points
    for i, vendor in enumerate(sorted_vendors):
        models = vendor_stats[vendor]['models']
        for model_name, rate in models:
            ax.scatter(rate, i, c='black', s=18, alpha=0.55,
                       zorder=5, edgecolors='white', linewidths=0.3)

    ax.set_yticks(y_positions)
    # Capitalize vendor names
    labels = [v.capitalize() if v != 'ai21' else 'AI21'
              for v in sorted_vendors]
    labels = [v.replace('Xai', 'xAI').replace('Openai', 'OpenAI')
              .replace('Deepseek', 'DeepSeek').replace('Zhipu', 'Zhipu')
              for v in labels]
    ax.set_yticklabels(labels)
    ax.set_xlabel('T5-T7 Hallucination Rate')
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 1.0)
    ax.set_title('T5-T7 Hallucination Rate by Vendor', fontweight='bold')

    # Add legend for color coding
    legend_elements = [
        mpatches.Patch(facecolor='#D32F2F', edgecolor='#333', label=r'High ($>$60\%)'),
        mpatches.Patch(facecolor='#F57C00', edgecolor='#333', label=r'Medium (30--60\%)'),
        mpatches.Patch(facecolor='#388E3C', edgecolor='#333', label=r'Low ($<$30\%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9,
              edgecolor='#CCCCCC')

    # Annotate mean values at end of bars
    for i, (vendor, mean_rate) in enumerate(zip(sorted_vendors, means)):
        ax.text(mean_rate + 0.012, i, f'{mean_rate*100:.0f}\\%',
                va='center', fontsize=8.5, color='#333333')

    save(fig, 'fig_a2_vendor_hallucination')


# ══════════════════════════════════════════════════════════════════════════════
# Fig A3: Cross-Generation Knowledge Trajectories
# ══════════════════════════════════════════════════════════════════════════════
def fig_a3_generation_trajectories():
    print("Generating Fig A3: Cross-Generation Knowledge Trajectories...")

    # Build model lookup
    model_acc = {m["model"]: m["accuracy"] for m in summary}

    # Define families with (generation_label, model_key) pairs
    # Only include models that exist in the data
    families_raw = {
        'Claude Opus': [
            ('4', 'claude-opus-4'),
            ('4.5', 'claude-opus-4.5'),
            ('4.6', 'claude-opus-4.6'),
        ],
        'Claude Sonnet': [
            ('3.7', 'claude-3.7-sonnet'),
            ('4', 'claude-sonnet-4'),
            ('4.5', 'claude-sonnet-4.5'),
            ('4.6', 'claude-sonnet-4.6'),
        ],
        'Claude Haiku': [
            ('3', 'claude-3-haiku'),
            ('3.5', 'claude-3.5-haiku'),
            ('4.5', 'claude-haiku-4.5'),
        ],
        'GPT': [
            ('3.5', 'gpt-3.5-turbo'),
            ('4', 'gpt-4'),
            ('4o', 'gpt-4o'),
            ('4.1', 'gpt-4.1'),
        ],
        'GLM': [
            ('4', 'glm-4-32b'),
            ('4.5', 'glm-4.5-think'),
            ('4.6', 'glm-4.6-think'),
            ('4.7', 'glm-4.7-think'),
            ('5', 'glm-5-think'),
            ('5.1', 'glm-5.1-think'),
        ],
        r'Qwen ($\sim$8B)': [
            ('2.5-7B', 'qwen-2.5-7b'),
            ('3-8B', 'qwen3-8b-think'),
            ('3.5-9B', 'qwen3.5-9b-think'),
        ],
        'DeepSeek': [
            ('v3', 'deepseek-v3'),
            ('v3.1', 'deepseek-v3.1'),
            ('v3.2', 'deepseek-v3.2'),
        ],
    }

    # Filter to only models that exist
    families = {}
    for fam_name, gens in families_raw.items():
        valid = [(g, k) for g, k in gens if k in model_acc]
        if len(valid) >= 2:
            families[fam_name] = valid

    # Color and marker assignments
    family_styles = {
        'Claude Opus':   {'color': '#6A0DAD', 'marker': 'D', 'ls': '-'},
        'Claude Sonnet': {'color': '#2166AC', 'marker': 's', 'ls': '-'},
        'Claude Haiku':  {'color': '#67A9CF', 'marker': 'o', 'ls': '-'},
        'GPT':           {'color': '#D32F2F', 'marker': '^', 'ls': '-'},
        'GLM':           {'color': '#388E3C', 'marker': 'v', 'ls': '-'},
        r'Qwen ($\sim$8B)':    {'color': '#F57C00', 'marker': 'p', 'ls': '--'},
        'DeepSeek':      {'color': '#7570B3', 'marker': 'h', 'ls': '-'},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Each family gets its own x-axis positions (categorical per family)
    # We'll spread families along shared x-axis using per-family local coords
    # Better approach: each family is a separate line with its own x coords
    # mapped to evenly spaced positions

    for fam_name, gens in families.items():
        style = family_styles[fam_name]
        gen_labels = [g for g, _ in gens]
        accs = [model_acc[k] for _, k in gens]
        xs = np.arange(len(gens))

        line, = ax.plot(xs, accs,
                        color=style['color'], marker=style['marker'],
                        markersize=8, linewidth=2.0, linestyle=style['ls'],
                        label=fam_name, markeredgecolor='white',
                        markeredgewidth=0.8, zorder=10)

        # Label generation points along each line (only first and last)
        for j, (gl, acc) in enumerate(zip(gen_labels, accs)):
            y_offset = 0.012
            ax.annotate(gl, (xs[j], acc + y_offset),
                        fontsize=6.5, ha='center', va='bottom',
                        color=style['color'], fontweight='bold',
                        alpha=0.85)

    # Annotate notable regressions with red arrows
    regressions = []

    # GPT 4 -> 4o regression
    if 'GPT' in families:
        gpt_gens = families['GPT']
        gpt_labels = [g for g, _ in gpt_gens]
        gpt_accs = [model_acc[k] for _, k in gpt_gens]
        if '4' in gpt_labels and '4o' in gpt_labels:
            i4 = gpt_labels.index('4')
            i4o = gpt_labels.index('4o')
            regressions.append(('GPT', i4, i4o, gpt_accs[i4], gpt_accs[i4o],
                                r'GPT-4$\rightarrow$4o'))

    # Claude Haiku 3.5 -> 4.5 regression
    if 'Claude Haiku' in families:
        haiku_gens = families['Claude Haiku']
        haiku_labels = [g for g, _ in haiku_gens]
        haiku_accs = [model_acc[k] for _, k in haiku_gens]
        if '3.5' in haiku_labels and '4.5' in haiku_labels:
            i35 = haiku_labels.index('3.5')
            i45 = haiku_labels.index('4.5')
            if haiku_accs[i45] < haiku_accs[i35]:
                regressions.append(('Claude Haiku', i35, i45,
                                    haiku_accs[i35], haiku_accs[i45],
                                    r'Haiku 3.5$\rightarrow$4.5'))

    # Claude Sonnet 3.7 -> 4 regression
    if 'Claude Sonnet' in families:
        sonnet_gens = families['Claude Sonnet']
        sonnet_labels = [g for g, _ in sonnet_gens]
        sonnet_accs = [model_acc[k] for _, k in sonnet_gens]
        if '3.7' in sonnet_labels and '4' in sonnet_labels:
            i37 = sonnet_labels.index('3.7')
            i4 = sonnet_labels.index('4')
            if sonnet_accs[i4] < sonnet_accs[i37]:
                regressions.append(('Claude Sonnet', i37, i4,
                                    sonnet_accs[i37], sonnet_accs[i4],
                                    r'Sonnet 3.7$\rightarrow$4'))

    # DeepSeek v3 -> v3.1 regression
    if 'DeepSeek' in families:
        ds_gens = families['DeepSeek']
        ds_labels = [g for g, _ in ds_gens]
        ds_accs = [model_acc[k] for _, k in ds_gens]
        if 'v3' in ds_labels and 'v3.1' in ds_labels:
            iv3 = ds_labels.index('v3')
            iv31 = ds_labels.index('v3.1')
            if ds_accs[iv31] < ds_accs[iv3]:
                regressions.append(('DeepSeek', iv3, iv31,
                                    ds_accs[iv3], ds_accs[iv31],
                                    r'DS v3$\rightarrow$v3.1'))

    # Stagger regression labels to avoid overlap
    reg_offsets = {
        r'GPT-4$\rightarrow$4o': (-0.15, -0.045),
        r'Haiku 3.5$\rightarrow$4.5': (0.15, -0.045),
        r'Sonnet 3.7$\rightarrow$4': (-0.15, 0.030),
        r'DS v3$\rightarrow$v3.1': (0.15, 0.030),
    }
    for fam, i_from, i_to, acc_from, acc_to, label in regressions:
        mid_x = (i_from + i_to) / 2
        mid_y = (acc_from + acc_to) / 2
        dx, dy = reg_offsets.get(label, (0, -0.03))
        ax.annotate('',
                    xy=(i_to, acc_to + 0.005),
                    xytext=(i_from, acc_from - 0.005),
                    arrowprops=dict(arrowstyle='->', color='red',
                                    lw=2.0, shrinkA=8, shrinkB=8))
        ax.annotate(label, (mid_x + dx, mid_y + dy),
                    fontsize=7.5, ha='center', color='red',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#FFF0F0',
                              ec='red', alpha=0.85, lw=0.6))

    # X-axis: use the longest family for tick positions
    max_gens = max(len(v) for v in families.values())
    ax.set_xlim(-0.3, max_gens - 0.7)
    ax.set_xticks(range(max_gens))
    ax.set_xticklabels([f'Gen {i+1}' for i in range(max_gens)])
    ax.set_xlabel('Model Generation (family-specific)')
    ax.set_ylabel('Penalized Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_title('Knowledge Evolution Across Model Generations', fontweight='bold')

    # Legend outside plot area
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#CCCCCC',
              ncol=2)

    save(fig, 'fig_a3_generation_trajectories')


# ══════════════════════════════════════════════════════════════════════════════
# Fig A4: GPT-5 Family Lineup
# ══════════════════════════════════════════════════════════════════════════════
def fig_a4_gpt5_family():
    print("Generating Fig A4: GPT-5 Family Lineup...")

    model_acc = {m["model"]: m for m in summary}

    # GPT-5 variants to include (in order specified)
    gpt5_names = [
        'gpt-5-nano', 'gpt-5-nano-think',
        'gpt-5-mini', 'gpt-5-mini-think',
        'gpt-5', 'gpt-5-think', 'gpt-5-pro',
        'gpt-5.1', 'gpt-5.2', 'gpt-5.2-pro',
        'gpt-5.3', 'gpt-5.4', 'gpt-5.4-mini',
        'gpt-5.4-nano', 'gpt-5.4-pro',
    ]

    # Filter to those present in data
    gpt5_models = [(n, model_acc[n]) for n in gpt5_names if n in model_acc]

    # Sort by accuracy
    gpt5_models.sort(key=lambda x: x[1]['accuracy'])

    # Assign color by model tier/size
    def get_tier_color(name):
        if 'nano' in name:
            return '#BBDEFB'   # light blue (nano)
        elif 'mini' in name:
            return '#64B5F6'   # medium blue (mini)
        elif 'pro' in name:
            return '#0D47A1'   # darkest blue (pro)
        else:
            return '#1976D2'   # dark blue (base)

    fig, ax = plt.subplots(figsize=(8, 5))

    names = [n for n, _ in gpt5_models]
    accs = [m['accuracy'] for _, m in gpt5_models]
    t5_accs = [m['tier_accuracy'].get('T5', 0.0) for _, m in gpt5_models]
    colors = [get_tier_color(n) for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, height=0.65, color=colors, alpha=0.9,
                   edgecolor='#333333', linewidth=0.6)

    # Annotate T5 accuracy inside or above each bar
    for i, (name, acc, t5) in enumerate(zip(names, accs, t5_accs)):
        # Place T5 annotation near the end of the bar
        t5_label = f'T5={t5*100:.0f}\\%'
        if acc > 0.15:
            ax.text(acc - 0.01, i, t5_label,
                    va='center', ha='right', fontsize=8,
                    color='white', fontweight='bold')
        else:
            ax.text(acc + 0.01, i, t5_label,
                    va='center', ha='left', fontsize=8,
                    color='#333333', fontweight='bold')

    # Display labels
    display_names = [n.replace('gpt-5', 'GPT-5') for n in names]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel('Penalized Accuracy')
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 0.82)
    ax.set_title('GPT-5 Family: Size Stratification Revealed by IKP',
                 fontweight='bold')

    # Legend for size tiers
    legend_elements = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#333', label='Nano'),
        mpatches.Patch(facecolor='#64B5F6', edgecolor='#333', label='Mini'),
        mpatches.Patch(facecolor='#1976D2', edgecolor='#333', label='Base'),
        mpatches.Patch(facecolor='#0D47A1', edgecolor='#333', label='Pro'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9,
              edgecolor='#CCCCCC', title='Size Tier')

    save(fig, 'fig_a4_gpt5_family')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"Loaded {N_MODELS} models (excluding nemotron-ultra-253b)")
    fig_a1_tier_boxplots()
    fig_a2_vendor_hallucination()
    fig_a3_generation_trajectories()
    fig_a4_gpt5_family()
    print("\nAll appendix figures generated.")
