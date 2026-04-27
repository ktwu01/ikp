"""Calibration curve fitting and parameter estimation.

Implements:
1. Aggregate log-linear fit: log(N) = alpha * A(N) + beta
2. Per-tier logistic fit: T_i(N) = L_i / (1 + exp(-k_i * (log(N) - m_i)))
3. Per-tier logistic inversion for estimation
4. Leave-one-out cross-validation
5. Bootstrap confidence intervals
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # logistic function
from scipy.stats import pearsonr
import warnings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def logistic(x, L, k, m):
    """Logistic sigmoid: L / (1 + exp(-k*(x - m)))"""
    return L * expit(k * (x - m))


def fit_aggregate_log_linear(log_params: np.ndarray, accuracies: np.ndarray):
    """Fit log(N) = alpha * accuracy + beta using OLS.

    Returns: (alpha, beta, r_squared, rmse)
    """
    # OLS: accuracy -> log_params
    A = np.column_stack([accuracies, np.ones_like(accuracies)])
    result = np.linalg.lstsq(A, log_params, rcond=None)
    alpha, beta = result[0]

    # Predictions and fit quality
    predicted = alpha * accuracies + beta
    ss_res = np.sum((log_params - predicted) ** 2)
    ss_tot = np.sum((log_params - np.mean(log_params)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / len(log_params))

    r, p_value = pearsonr(accuracies, log_params)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "pearson_r": float(r),
        "pearson_p": float(p_value),
        "n_points": len(log_params),
    }


def fit_tier_logistic(
    log_params: np.ndarray,
    tier_accuracies: np.ndarray,
    tier_name: str,
) -> dict:
    """Fit a logistic sigmoid to one tier's accuracy vs log(params).

    T_i(log_N) = L / (1 + exp(-k * (log_N - m)))

    Returns: dict with fitted parameters and fit quality.
    """
    # Initial guesses
    L_init = min(1.0, np.max(tier_accuracies) * 1.1)
    k_init = 2.0
    m_init = np.median(log_params)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                logistic,
                log_params,
                tier_accuracies,
                p0=[L_init, k_init, m_init],
                bounds=([0.01, 0.01, -5], [1.5, 50.0, 40.0]),
                maxfev=10000,
            )
        L, k, m = popt
        predicted = logistic(log_params, L, k, m)
        ss_res = np.sum((tier_accuracies - predicted) ** 2)
        ss_tot = np.sum((tier_accuracies - np.mean(tier_accuracies)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "tier": tier_name,
            "L": float(L),
            "k": float(k),
            "m": float(m),
            "r_squared": float(r_squared),
            "residuals": (tier_accuracies - predicted).tolist(),
            "converged": True,
        }
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Logistic fit failed for {tier_name}: {e}")
        return {
            "tier": tier_name,
            "L": float(L_init),
            "k": float(k_init),
            "m": float(m_init),
            "r_squared": 0.0,
            "converged": False,
            "error": str(e),
        }


def estimate_log_params_aggregate(accuracy: float, alpha: float, beta: float) -> float:
    """Estimate log(N) from aggregate accuracy using log-linear fit."""
    return alpha * accuracy + beta


def estimate_log_params_per_tier(
    tier_accuracies: dict,
    tier_fits: dict,
) -> float:
    """Estimate log(N) from per-tier accuracies using logistic inversion.

    Uses weighted median where weight = information content
    (highest near 50% accuracy, lowest near 0% or 100%).
    """
    estimates = []
    weights = []

    for tier, acc in tier_accuracies.items():
        fit = tier_fits.get(tier)
        if fit is None or not fit.get("converged", False):
            continue

        L, k, m = fit["L"], fit["k"], fit["m"]

        # Clamp accuracy to avoid log(0) or log(inf)
        acc_clamped = max(0.01 * L, min(0.99 * L, acc))

        # Invert logistic: log_N = m + (1/k) * log(acc / (L - acc))
        log_n = m + (1.0 / k) * np.log(acc_clamped / (L - acc_clamped))

        # Weight: information content = how close to L/2 (transition region)
        # Using the derivative of the logistic at this accuracy as weight
        # |dT/d(logN)| = k * acc * (1 - acc/L) ... maximized at acc = L/2
        weight = k * acc_clamped * (1 - acc_clamped / L)
        weight = max(weight, 1e-6)

        estimates.append(log_n)
        weights.append(weight)

    if not estimates:
        return np.nan

    estimates = np.array(estimates)
    weights = np.array(weights)
    weights /= weights.sum()

    # Weighted mean (more stable than weighted median for small sample)
    return float(np.average(estimates, weights=weights))


def run_full_calibration(calibration_data: list[dict]) -> dict:
    """Run full calibration pipeline.

    Args:
        calibration_data: list of dicts, each with:
            - model_name: str
            - params_billion: float
            - per_tier_accuracy: dict[str, float]  (e.g., {"T1": 0.95, "T2": 0.80, ...})
            - aggregate_accuracy: float

    Returns:
        dict with all calibration results.
    """
    # Sort by parameter count
    calibration_data = sorted(calibration_data, key=lambda x: x["params_billion"])

    log_params = np.array([np.log(d["params_billion"]) for d in calibration_data])
    aggregates = np.array([d["aggregate_accuracy"] for d in calibration_data])

    # 1. Aggregate log-linear fit
    agg_fit = fit_aggregate_log_linear(log_params, aggregates)
    logger.info(f"Aggregate log-linear: R²={agg_fit['r_squared']:.4f}, α={agg_fit['alpha']:.4f}, β={agg_fit['beta']:.4f}")

    # 2. Per-tier logistic fits
    all_tiers = sorted(set(t for d in calibration_data for t in d["per_tier_accuracy"].keys()))
    tier_fits = {}
    for tier in all_tiers:
        tier_accs = np.array([d["per_tier_accuracy"].get(tier, 0.0) for d in calibration_data])
        fit = fit_tier_logistic(log_params, tier_accs, tier)
        tier_fits[tier] = fit
        status = f"R²={fit['r_squared']:.4f}" if fit["converged"] else "FAILED"
        logger.info(f"  {tier}: {status}, L={fit['L']:.3f}, k={fit['k']:.3f}, m={fit['m']:.3f}")

    # 3. LOO-CV
    loocv_results = run_loocv(calibration_data, log_params, aggregates, tier_fits, all_tiers)

    # 4. Bootstrap confidence intervals for the aggregate fit
    bootstrap_ci = bootstrap_aggregate_fit(log_params, aggregates, n_bootstrap=1000)

    return {
        "aggregate_fit": agg_fit,
        "tier_fits": tier_fits,
        "loocv": loocv_results,
        "bootstrap_ci": bootstrap_ci,
        "calibration_models": [
            {"name": d["model_name"], "params_billion": d["params_billion"],
             "aggregate_accuracy": d["aggregate_accuracy"],
             "per_tier_accuracy": d["per_tier_accuracy"]}
            for d in calibration_data
        ],
    }


def run_loocv(calibration_data, log_params, aggregates, tier_fits, all_tiers):
    """Leave-one-out cross-validation."""
    n = len(calibration_data)
    results = []

    for i in range(n):
        # Leave out model i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        # Refit aggregate
        loo_agg = fit_aggregate_log_linear(log_params[mask], aggregates[mask])

        # Predict held-out model
        predicted_log_n_agg = estimate_log_params_aggregate(
            aggregates[i], loo_agg["alpha"], loo_agg["beta"]
        )

        # Refit per-tier and predict
        loo_tier_fits = {}
        for tier in all_tiers:
            tier_accs = np.array([d["per_tier_accuracy"].get(tier, 0.0) for d in calibration_data])
            loo_tier_fits[tier] = fit_tier_logistic(log_params[mask], tier_accs[mask], tier)

        predicted_log_n_tier = estimate_log_params_per_tier(
            calibration_data[i]["per_tier_accuracy"],
            loo_tier_fits,
        )

        actual_log_n = log_params[i]
        actual_params = calibration_data[i]["params_billion"]

        # Multiplicative error factor
        pred_params_agg = np.exp(predicted_log_n_agg)
        mult_error_agg = max(pred_params_agg / actual_params, actual_params / pred_params_agg)

        pred_params_tier = np.exp(predicted_log_n_tier) if not np.isnan(predicted_log_n_tier) else np.nan
        mult_error_tier = (
            max(pred_params_tier / actual_params, actual_params / pred_params_tier)
            if not np.isnan(pred_params_tier) else np.nan
        )

        results.append({
            "model": calibration_data[i]["model_name"],
            "actual_params_B": actual_params,
            "predicted_params_agg_B": float(pred_params_agg),
            "predicted_params_tier_B": float(pred_params_tier) if not np.isnan(pred_params_tier) else None,
            "mult_error_agg": float(mult_error_agg),
            "mult_error_tier": float(mult_error_tier) if not np.isnan(mult_error_tier) else None,
        })

    # Summary stats
    mult_errors_agg = [r["mult_error_agg"] for r in results]
    mult_errors_tier = [r["mult_error_tier"] for r in results if r["mult_error_tier"] is not None]

    summary = {
        "per_model": results,
        "aggregate_estimator": {
            "median_mult_error": float(np.median(mult_errors_agg)),
            "mean_mult_error": float(np.mean(mult_errors_agg)),
            "max_mult_error": float(np.max(mult_errors_agg)),
            "pct_within_2x": float(np.mean(np.array(mult_errors_agg) <= 2.0)),
        },
    }
    if mult_errors_tier:
        summary["tier_estimator"] = {
            "median_mult_error": float(np.median(mult_errors_tier)),
            "mean_mult_error": float(np.mean(mult_errors_tier)),
            "max_mult_error": float(np.max(mult_errors_tier)),
            "pct_within_2x": float(np.mean(np.array(mult_errors_tier) <= 2.0)),
        }

    logger.info(f"LOOCV aggregate: median error = {summary['aggregate_estimator']['median_mult_error']:.2f}x")
    return summary


def bootstrap_aggregate_fit(log_params, accuracies, n_bootstrap=1000, seed=42):
    """Bootstrap confidence intervals for the aggregate log-linear fit."""
    rng = np.random.RandomState(seed)
    n = len(log_params)
    alphas, betas = [], []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        A = np.column_stack([accuracies[idx], np.ones(n)])
        try:
            result = np.linalg.lstsq(A, log_params[idx], rcond=None)
            alphas.append(result[0][0])
            betas.append(result[0][1])
        except np.linalg.LinAlgError:
            continue

    return {
        "alpha_ci_95": [float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))],
        "beta_ci_95": [float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))],
        "n_bootstrap": n_bootstrap,
    }


def estimate_target_model(
    target_results: dict,
    calibration: dict,
) -> dict:
    """Estimate parameter count for a target (unknown) model.

    Args:
        target_results: dict with per_tier_accuracy and aggregate_accuracy
        calibration: output of run_full_calibration()

    Returns:
        dict with point estimates and confidence intervals.
    """
    agg_fit = calibration["aggregate_fit"]
    tier_fits = calibration["tier_fits"]

    # Aggregate estimate
    log_n_agg = estimate_log_params_aggregate(
        target_results["aggregate_accuracy"],
        agg_fit["alpha"], agg_fit["beta"],
    )
    params_agg = np.exp(log_n_agg)

    # Per-tier estimate
    log_n_tier = estimate_log_params_per_tier(
        target_results["per_tier_accuracy"],
        tier_fits,
    )
    params_tier = np.exp(log_n_tier) if not np.isnan(log_n_tier) else None

    # Bootstrap CI for aggregate
    boot = calibration["bootstrap_ci"]
    log_n_lo = boot["alpha_ci_95"][0] * target_results["aggregate_accuracy"] + boot["beta_ci_95"][0]
    log_n_hi = boot["alpha_ci_95"][1] * target_results["aggregate_accuracy"] + boot["beta_ci_95"][1]
    ci_lo = np.exp(min(log_n_lo, log_n_hi))
    ci_hi = np.exp(max(log_n_lo, log_n_hi))

    return {
        "model_name": target_results.get("model_name", "unknown"),
        "aggregate_accuracy": target_results["aggregate_accuracy"],
        "per_tier_accuracy": target_results["per_tier_accuracy"],
        "estimated_params_aggregate_B": float(params_agg),
        "estimated_params_tier_B": float(params_tier) if params_tier else None,
        "ci_95_aggregate_B": [float(ci_lo), float(ci_hi)],
    }
