#!/usr/bin/env python3
"""Phase 5: Robustness and ablation studies.

1. Tier ablation: estimate using subsets of tiers
2. Probe count ablation: subsample probes per tier
3. Geographic ablation: restrict to one region
4. Frequency correlation: per-probe accuracy vs web frequency
5. MoE analysis: total vs active params
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    config = json.load(open(PROJECT_ROOT / "configs" / "models.json"))
    cal_models = config["calibration_models"]

    data = []
    for f in sorted(Path(PROJECT_ROOT / "data" / "raw_responses").glob("*_results.json")):
        r = json.load(open(f))
        name = r['model_name']
        if name not in cal_models:
            continue
        data.append({
            'name': name,
            'params': cal_models[name]['params_billion'],
            'family': cal_models[name]['family'],
            'arch': cal_models[name]['architecture'],
            'active_params': cal_models[name].get('active_params_billion', cal_models[name]['params_billion']),
            'per_tier': r['per_tier_accuracy'],
            'agg': r['aggregate_accuracy'],
        })
    data.sort(key=lambda x: x['params'])
    return data


def tier_ablation(data):
    """Estimate parameters using subsets of tiers."""
    logger.info("=== Tier Ablation ===")
    dense = [d for d in data if d['arch'] == 'dense']

    tier_subsets = {
        'All (T1-T7)': ['T1','T2','T3','T4','T5','T6','T7'],
        'Core (T3-T6)': ['T3','T4','T5','T6'],
        'T1-T5 only': ['T1','T2','T3','T4','T5'],
        'T3-T5 only': ['T3','T4','T5'],
        'T4-T6 only': ['T4','T5','T6'],
        'Single: T3': ['T3'],
        'Single: T4': ['T4'],
        'Single: T5': ['T5'],
        'Single: T6': ['T6'],
    }

    results = {}
    for label, tiers in tier_subsets.items():
        # Compute aggregate using only selected tiers
        accs = np.array([np.mean([d['per_tier'].get(t, 0) for t in tiers]) for d in dense])
        log_params = np.log(np.array([d['params'] for d in dense]))

        A = np.column_stack([accs, np.ones_like(accs)])
        res = np.linalg.lstsq(A, log_params, rcond=None)
        alpha, beta = res[0]
        pred = alpha * accs + beta
        ss_res = np.sum((log_params - pred)**2)
        ss_tot = np.sum((log_params - np.mean(log_params))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # LOOCV
        errors = []
        for i in range(len(dense)):
            train_idx = [j for j in range(len(dense)) if j != i]
            A_t = np.column_stack([accs[train_idx], np.ones(len(train_idx))])
            res_t = np.linalg.lstsq(A_t, log_params[train_idx], rcond=None)
            a, b = res_t[0]
            pred_log = a * accs[i] + b
            pred_p = np.exp(pred_log)
            actual_p = dense[i]['params']
            errors.append(max(pred_p/actual_p, actual_p/pred_p))

        results[label] = {
            'r2': r2,
            'median_error': np.median(errors),
            'within_2x': np.mean(np.array(errors) <= 2.0),
        }
        logger.info(f"  {label:20s}: R²={r2:.3f}, median_err={np.median(errors):.2f}x, within_2x={np.mean(np.array(errors)<=2.0):.0%}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = list(results.keys())
    r2s = [results[l]['r2'] for l in labels]
    errs = [results[l]['median_error'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2s, width, label='R²', color='steelblue')
    ax.set_ylabel('R²', color='steelblue')
    ax.set_ylim(0, 1.1)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, errs, width, label='Median Error (x)', color='coral')
    ax2.set_ylabel('Median Error (x)', color='coral')
    ax2.set_ylim(0, max(errs) * 1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_title('Tier Ablation: Impact of Tier Selection on Estimation Quality')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_tier_ablation.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig6_tier_ablation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved fig6_tier_ablation")
    return results


def moe_analysis(data):
    """Analyze MoE models: total vs active params."""
    logger.info("\n=== MoE Analysis ===")
    dense = [d for d in data if d['arch'] == 'dense']
    moe = [d for d in data if d['arch'] == 'moe']

    if not moe:
        logger.info("  No MoE models found")
        return

    # Fit on dense models
    accs_d = np.array([d['agg'] for d in dense])
    log_p_d = np.log(np.array([d['params'] for d in dense]))
    A = np.column_stack([accs_d, np.ones_like(accs_d)])
    res = np.linalg.lstsq(A, log_p_d, rcond=None)
    alpha, beta = res[0]

    for d in moe:
        est = np.exp(alpha * d['agg'] + beta)
        ratio_total = est / d['params']
        ratio_active = est / d['active_params']
        logger.info(f"  {d['name']:25s}: total={d['params']:.0f}B, active={d['active_params']:.0f}B")
        logger.info(f"    IKP estimate: {est:.0f}B")
        logger.info(f"    vs total: {ratio_total:.2f}x, vs active: {ratio_active:.2f}x")
        logger.info(f"    IKP is closer to {'total' if abs(np.log(ratio_total)) < abs(np.log(ratio_active)) else 'active'} params")


def probe_count_ablation(data):
    """How does estimation quality degrade with fewer probes per tier?"""
    logger.info("\n=== Probe Count Ablation ===")

    # Load raw probe-level data
    dense = [d for d in data if d['arch'] == 'dense']
    all_raw = {}
    for d in dense:
        raw_file = PROJECT_ROOT / "data" / "raw_responses" / f"{d['name']}_responses.json"
        if raw_file.exists():
            raw = json.load(open(raw_file))
            all_raw[d['name']] = raw['probe_results']

    if not all_raw:
        logger.info("  No raw response data found")
        return

    # For each subsample size, bootstrap estimation quality
    subsample_sizes = [10, 20, 30, 50, 100, 150, 200]
    results = {}

    for n_per_tier in subsample_sizes:
        bootstrap_errors = []
        rng = np.random.RandomState(42)

        for _ in range(50):  # 50 bootstrap iterations
            # Subsample probes per model
            model_accs = []
            for d in dense:
                if d['name'] not in all_raw:
                    continue
                raw = all_raw[d['name']]
                by_tier = {}
                for r in raw:
                    t = r['tier']
                    if t not in by_tier:
                        by_tier[t] = []
                    by_tier[t].append(r)

                tier_accs = []
                for t in sorted(by_tier.keys()):
                    probes = by_tier[t]
                    n = min(n_per_tier, len(probes))
                    sample = [probes[i] for i in rng.choice(len(probes), n, replace=True)]
                    scored = [r for r in sample if not r.get('excluded')]
                    if scored:
                        acc = np.mean([r['correct'] for r in scored])
                    else:
                        acc = 0.0
                    tier_accs.append(acc)

                model_accs.append((d['params'], np.mean(tier_accs)))

            if len(model_accs) < 3:
                continue

            params = np.array([m[0] for m in model_accs])
            accs = np.array([m[1] for m in model_accs])
            log_p = np.log(params)

            # Fit and compute LOOCV error
            for i in range(len(model_accs)):
                idx = [j for j in range(len(model_accs)) if j != i]
                A = np.column_stack([accs[idx], np.ones(len(idx))])
                try:
                    res = np.linalg.lstsq(A, log_p[idx], rcond=None)
                    a, b = res[0]
                    pred = np.exp(a * accs[i] + b)
                    err = max(pred/params[i], params[i]/pred)
                    bootstrap_errors.append(err)
                except:
                    pass

        if bootstrap_errors:
            results[n_per_tier] = {
                'median_error': np.median(bootstrap_errors),
                'p25': np.percentile(bootstrap_errors, 25),
                'p75': np.percentile(bootstrap_errors, 75),
            }
            logger.info(f"  {n_per_tier:3d} probes/tier: median_err={np.median(bootstrap_errors):.2f}x [{np.percentile(bootstrap_errors, 25):.2f}-{np.percentile(bootstrap_errors, 75):.2f}]")

    # Plot
    if results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        sizes = sorted(results.keys())
        medians = [results[n]['median_error'] for n in sizes]
        p25s = [results[n]['p25'] for n in sizes]
        p75s = [results[n]['p75'] for n in sizes]

        ax.plot(sizes, medians, 'o-', color='steelblue', linewidth=2, label='Median error')
        ax.fill_between(sizes, p25s, p75s, alpha=0.2, color='steelblue', label='IQR')
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='2x target')
        ax.set_xlabel('Probes per tier')
        ax.set_ylabel('Multiplicative error (x)')
        ax.set_title('Probe Count Ablation: Minimum Probes for <2x Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig7_probe_count_ablation.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIG_DIR / "fig7_probe_count_ablation.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("Saved fig7_probe_count_ablation")

    return results


def main():
    data = load_data()
    if not data:
        logger.error("No calibration data found")
        return

    logger.info(f"Loaded {len(data)} calibration models")

    tier_results = tier_ablation(data)
    moe_analysis(data)
    probe_results = probe_count_ablation(data)

    # Save all ablation results
    output = {
        'tier_ablation': tier_results,
        'probe_count_ablation': {str(k): v for k, v in (probe_results or {}).items()},
    }
    output_file = PROJECT_ROOT / "results" / "ablation_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
