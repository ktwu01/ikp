#!/usr/bin/env python3
"""Phase 5: Analysis, visualization, and ablation studies.

Generates all figures and tables for the paper.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from calibration import logistic, fit_aggregate_log_linear, fit_tier_logistic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FAMILY_COLORS = {
    "qwen": "#1f77b4",
    "llama": "#ff7f0e",
    "llama2": "#ff7f0e",
    "mistral": "#2ca02c",
    "phi": "#d62728",
    "gemma": "#9467bd",
    "deepseek": "#8c564b",
}

FAMILY_MARKERS = {
    "qwen": "o",
    "llama": "s",
    "llama2": "D",
    "mistral": "^",
    "phi": "v",
    "gemma": "P",
    "deepseek": "*",
}


def load_data():
    """Load all results."""
    cal_results_file = PROJECT_ROOT / "data" / "calibration" / "all_calibration_results.json"
    cal_fit_file = PROJECT_ROOT / "data" / "calibration" / "calibration_fit.json"
    target_file = PROJECT_ROOT / "results" / "target_estimates.json"

    data = {}
    if cal_results_file.exists():
        with open(cal_results_file) as f:
            data["calibration_results"] = json.load(f)
    if cal_fit_file.exists():
        with open(cal_fit_file) as f:
            data["calibration_fit"] = json.load(f)
    if target_file.exists():
        with open(target_file) as f:
            data["target_estimates"] = json.load(f)

    return data


def plot_calibration_curve(data):
    """Figure 1: Aggregate accuracy vs log(params) with calibration fit."""
    cal = data.get("calibration_results", [])
    fit = data.get("calibration_fit", {}).get("aggregate_fit", {})
    if not cal or not fit:
        logger.warning("No calibration data for plotting")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for r in cal:
        family = r.get("family", "unknown")
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        moe_label = " (MoE)" if r.get("architecture") == "moe" else ""
        ax.scatter(
            r["params_billion"], r["aggregate_accuracy"],
            c=color, marker=marker, s=100, zorder=5,
            label=f"{r['model_name']}{moe_label}",
        )

    # Fit line
    alpha, beta = fit["alpha"], fit["beta"]
    x_range = np.logspace(-0.5, 3.5, 100)
    y_pred = (np.log(x_range) - beta) / alpha
    ax.plot(x_range, y_pred, "k--", alpha=0.6, linewidth=2,
            label=f"Log-linear fit (R²={fit['r_squared']:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)", fontsize=14)
    ax.set_ylabel("Aggregate IKP Accuracy", fontsize=14)
    ax.set_title("IKP Calibration: Accuracy vs Model Size", fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_calibration_curve.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig1_calibration_curve.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved fig1_calibration_curve")


def plot_tier_sigmoids(data):
    """Figure 2: Per-tier accuracy sigmoids."""
    cal = data.get("calibration_results", [])
    tier_fits = data.get("calibration_fit", {}).get("tier_fits", {})
    if not cal or not tier_fits:
        return

    tiers = sorted(tier_fits.keys())
    n_tiers = len(tiers)
    fig, axes = plt.subplots(1, n_tiers, figsize=(4 * n_tiers, 5), sharey=True)
    if n_tiers == 1:
        axes = [axes]

    tier_colors = plt.cm.viridis(np.linspace(0, 0.9, n_tiers))

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        fit = tier_fits[tier]

        # Data points
        params = [r["params_billion"] for r in cal]
        accs = [r["per_tier_accuracy"].get(tier, 0) for r in cal]
        ax.scatter(params, accs, c="black", s=40, zorder=5)

        # Fitted sigmoid
        if fit.get("converged", False):
            x = np.logspace(-0.5, 3.5, 200)
            y = logistic(np.log(x), fit["L"], fit["k"], fit["m"])
            ax.plot(x, y, color=tier_colors[idx], linewidth=2,
                    label=f"R²={fit['r_squared']:.3f}")

        ax.set_xscale("log")
        ax.set_title(f"{tier}\n{fit.get('tier', tier)}", fontsize=12)
        ax.set_xlabel("Params (B)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Tier Accuracy")
    fig.suptitle("Per-Tier Logistic Sigmoids", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_tier_sigmoids.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig2_tier_sigmoids.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved fig2_tier_sigmoids")


def plot_loocv(data):
    """Figure 3: LOO-CV predicted vs actual parameter counts."""
    loocv = data.get("calibration_fit", {}).get("loocv", {})
    if not loocv:
        return

    per_model = loocv["per_model"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    actual = [r["actual_params_B"] for r in per_model]
    predicted = [r["predicted_params_agg_B"] for r in per_model]

    ax.scatter(actual, predicted, c="steelblue", s=80, zorder=5)

    for r in per_model:
        ax.annotate(r["model"], (r["actual_params_B"], r["predicted_params_agg_B"]),
                     fontsize=7, ha="left", va="bottom", xytext=(5, 5),
                     textcoords="offset points")

    # Perfect prediction line
    lim = [min(min(actual), min(predicted)) * 0.5,
           max(max(actual), max(predicted)) * 2]
    ax.plot(lim, lim, "k--", alpha=0.5, label="Perfect prediction")
    # 2x error bands
    ax.fill_between(lim, [l * 0.5 for l in lim], [l * 2 for l in lim],
                     alpha=0.1, color="gray", label="2x error band")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual Parameters (B)", fontsize=14)
    ax.set_ylabel("Predicted Parameters (B)", fontsize=14)
    ax.set_title("LOO-CV: Predicted vs Actual", fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_loocv.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig3_loocv.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved fig3_loocv")


def plot_target_estimates(data):
    """Figure 4: Frontier model parameter estimates with CIs."""
    estimates = data.get("target_estimates", [])
    if not estimates:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Sort by estimated size
    estimates = sorted(estimates, key=lambda x: x["estimated_params_aggregate_B"])

    names = [e["model_name"] for e in estimates]
    est = [e["estimated_params_aggregate_B"] for e in estimates]
    ci_lo = [e["ci_95_aggregate_B"][0] for e in estimates]
    ci_hi = [e["ci_95_aggregate_B"][1] for e in estimates]

    y_pos = range(len(names))

    # Error bars
    xerr_lo = [e - lo for e, lo in zip(est, ci_lo)]
    xerr_hi = [hi - e for e, hi in zip(est, ci_hi)]
    ax.barh(y_pos, est, xerr=[xerr_lo, xerr_hi], align="center",
            color="steelblue", alpha=0.7, capsize=5)

    # Reference estimates
    for i, e in enumerate(estimates):
        ref = e.get("reference_estimate_billion")
        if ref:
            ax.scatter(ref, i, c="red", marker="x", s=100, zorder=5,
                      label="Reference" if i == 0 else None)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated Parameters (billions)", fontsize=14)
    ax.set_title("Frontier Model Parameter Estimates", fontsize=16)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_target_estimates.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig4_target_estimates.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved fig4_target_estimates")


def plot_tier_heatmap(data):
    """Figure 5: Per-tier accuracy heatmap for all models."""
    cal = data.get("calibration_results", [])
    estimates = data.get("target_estimates", [])
    if not cal:
        return

    all_models = []
    for r in sorted(cal, key=lambda x: x["params_billion"]):
        all_models.append({
            "name": r["model_name"],
            "per_tier": r["per_tier_accuracy"],
            "type": "calibration",
        })
    for e in sorted(estimates or [], key=lambda x: x["estimated_params_aggregate_B"]):
        all_models.append({
            "name": e["model_name"],
            "per_tier": e["per_tier_accuracy"],
            "type": "target",
        })

    tiers = sorted(set(t for m in all_models for t in m["per_tier"].keys()))
    matrix = np.array([[m["per_tier"].get(t, 0) for t in tiers] for m in all_models])
    names = [m["name"] for m in all_models]

    fig, ax = plt.subplots(1, 1, figsize=(10, max(8, len(names) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(tiers)):
            val = matrix[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, label="Accuracy")
    ax.set_title("Per-Tier Accuracy Heatmap", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_tier_heatmap.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig5_tier_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved fig5_tier_heatmap")


def generate_tables(data):
    """Generate LaTeX tables for the paper."""
    cal = data.get("calibration_results", [])
    estimates = data.get("target_estimates", [])
    cal_fit = data.get("calibration_fit", {})

    # Table 1: Calibration results
    if cal:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Calibration model results. Per-tier and aggregate IKP accuracy.}",
            r"\label{tab:calibration}",
            r"\small",
            r"\begin{tabular}{lrrrrrrrrrr}",
            r"\toprule",
            r"Model & Params (B) & T1 & T2 & T3 & T4 & T5 & T6 & T7 & Agg \\",
            r"\midrule",
        ]
        for r in sorted(cal, key=lambda x: x["params_billion"]):
            ta = r["per_tier_accuracy"]
            row = f"{r['model_name']} & {r['params_billion']:.1f}"
            for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
                row += f" & {ta.get(t, 0):.2f}"
            row += f" & {r['aggregate_accuracy']:.3f} \\\\"
            lines.append(row)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        with open(TABLE_DIR / "table1_calibration.tex", "w") as f:
            f.write("\n".join(lines))
        logger.info("Saved table1_calibration.tex")

    # Table 2: Target estimates
    if estimates:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Frontier model parameter estimates from IKP probes.}",
            r"\label{tab:estimates}",
            r"\small",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Model & Vendor & Agg Acc & Est (B) & 95\% CI & Ref (B) \\",
            r"\midrule",
        ]
        for e in sorted(estimates, key=lambda x: x["estimated_params_aggregate_B"]):
            ref = f"{e['reference_estimate_billion']:.0f}" if e.get("reference_estimate_billion") else "--"
            lines.append(
                f"{e['model_name']} & {e.get('vendor','--')} & "
                f"{e['aggregate_accuracy']:.3f} & "
                f"{e['estimated_params_aggregate_B']:.0f} & "
                f"[{e['ci_95_aggregate_B'][0]:.0f}, {e['ci_95_aggregate_B'][1]:.0f}] & "
                f"{ref} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        with open(TABLE_DIR / "table2_estimates.tex", "w") as f:
            f.write("\n".join(lines))
        logger.info("Saved table2_estimates.tex")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    if not data:
        logger.error("No data found. Run calibration and target experiments first.")
        sys.exit(1)

    plot_calibration_curve(data)
    plot_tier_sigmoids(data)
    plot_loocv(data)
    plot_target_estimates(data)
    plot_tier_heatmap(data)
    generate_tables(data)

    logger.info("\nAll figures and tables generated.")


if __name__ == "__main__":
    main()
