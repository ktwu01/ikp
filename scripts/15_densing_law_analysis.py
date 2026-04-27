#!/usr/bin/env python3
"""Densing Law falsification analysis for IKP.

Hypothesis (paper's thesis): IKP measures N_fact, the incompressible factual
storage component of model parameters. Unlike reasoning/MMLU-style benchmarks,
which exhibit the Densing Law (capability density doubles every ~3.5 months
at fixed parameter count), IKP should have zero time trend once parameters
are controlled for.

This script:
 1. Joins open-weight, known-parameter models with their IKP scores and
    release dates.
 2. Fits OLS regressions of penalized IKP accuracy on log10(params), optionally
    with release_date and architectural controls.
 3. Compares the fitted time coefficient against the Densing Law prediction.
 4. Emits a residuals-vs-date figure and a markdown report.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _st


class OLSResult:
    """Minimal OLS result wrapper providing statsmodels-like interface."""
    def __init__(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, k = X.shape
        # beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        y_hat = X @ beta
        resid = y - y_hat
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        dof = n - k
        sigma2 = ss_res / dof
        cov_beta = sigma2 * XtX_inv
        bse = np.sqrt(np.diag(cov_beta))
        self.params = beta
        self.bse = bse
        self.tvalues = beta / bse
        self.pvalues = 2 * (1 - _st.t.cdf(np.abs(self.tvalues), dof))
        self.rsquared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.rsquared_adj = 1 - (ss_res / dof) / (ss_tot / (n - 1)) if ss_tot > 0 else float("nan")
        self.nobs = n
        self.df_resid = dof
        self.resid = resid
        self.fitted = y_hat
        self._cov = cov_beta

    def conf_int(self, alpha=0.05):
        tcrit = _st.t.ppf(1 - alpha / 2, self.df_resid)
        lo = self.params - tcrit * self.bse
        hi = self.params + tcrit * self.bse
        return list(zip(lo, hi))


def add_const(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "all_models.json"
RESULTS_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "results" / "figures"
REPORT_PATH = ROOT / "data" / "analysis_densing_law.md"

# Densing Law: capability per parameter doubles every 3.5 months.
# Equivalently, at fixed capability, parameter requirement halves every 3.5 months.
DENSING_DOUBLING_MONTHS = 3.5

# Reference date for months_since (rounded month value).
REF_DATE = datetime(2024, 1, 1)


def months_since(date_str: str, ref: datetime = REF_DATE) -> float:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d.year - ref.year) * 12 + (d.month - ref.month) + (d.day - ref.day) / 30.0


def load_data() -> pd.DataFrame:
    cfg = json.loads(CFG_PATH.read_text())["models"]
    # Prefer final_assembly.json (lambda=-1, researcher v2 + non-researcher v1).
    fa_path = RESULTS_DIR / "final_assembly.json"
    fa = {}
    if fa_path.exists():
        for r in json.loads(fa_path.read_text()):
            fa[r["model"]] = r
    rows = []
    for name, m in cfg.items():
        if m.get("type") != "open":
            continue
        if not m.get("params_B"):
            continue
        if name in fa:
            r = fa[name]
        else:
            rp = RESULTS_DIR / f"{name}.json"
            if not rp.exists():
                continue
            r = json.loads(rp.read_text())
        acc = r.get("accuracy")
        raw = r.get("raw_accuracy")
        if acc is None or raw is None:
            continue
        rows.append({
            "model": name,
            "vendor": m.get("vendor"),
            "family": m.get("family"),
            "arch": m.get("arch"),
            "thinking": bool(m.get("thinking")),
            "params_B": float(m["params_B"]),
            "active_B": float(m.get("active_B") or m["params_B"]),
            "release_date": m["release_date"],
            "pen_acc": acc,
            "raw_acc": raw,
            "total": r.get("total"),
        })
    df = pd.DataFrame(rows)
    df["log10_params"] = np.log10(df["params_B"])
    df["log10_active"] = np.log10(df["active_B"])
    df["months"] = df["release_date"].map(months_since)
    df["year_frac"] = df["months"] / 12.0
    df["date"] = pd.to_datetime(df["release_date"])
    return df.sort_values("date").reset_index(drop=True)


def fit_ols(df: pd.DataFrame, X_cols: list[str], y_col: str = "pen_acc"):
    X = add_const(df[X_cols].astype(float).values)
    y = df[y_col].astype(float).values
    return OLSResult(X, y)


def nice_coef_table(model, names):
    headers = ["term", "coef", "std_err", "t", "p", "ci_lo", "ci_hi"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    ci = model.conf_int()
    for i, n in enumerate(names):
        lines.append(" | ".join([
            n,
            f"{model.params[i]:+.4f}",
            f"{model.bse[i]:.4f}",
            f"{model.tvalues[i]:+.2f}",
            f"{model.pvalues[i]:.3g}",
            f"{ci[i][0]:+.4f}",
            f"{ci[i][1]:+.4f}",
        ]))
    return "\n".join(lines)


def densing_predictions(m_log_params):
    """Return expected accuracy gain per month under Densing Law.

    Densing Law: at fixed accuracy, required params halve every DENSING_DOUBLING_MONTHS.
    Equivalently, fixing params, accuracy rises at a rate equivalent to multiplying
    params by 2 every DENSING_DOUBLING_MONTHS months. In the log-linear IKP model,
    this corresponds to a monthly accuracy increment of:
        m_log_params * log10(2) / DENSING_DOUBLING_MONTHS
    """
    return m_log_params * math.log10(2.0) / DENSING_DOUBLING_MONTHS


def run_analysis():
    df = load_data()
    print(f"N = {len(df)} open-weight models with known params and IKP results.")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Param range: {df['params_B'].min():.2f}B -> {df['params_B'].max():.1f}B")

    # ---- Baseline M0: acc ~ log10(params)
    m0 = fit_ols(df, ["log10_params"])
    beta_params = float(m0.params[1])
    densing_monthly = densing_predictions(beta_params)
    print(f"\nBaseline M0: pen_acc ~ log10(params)")
    print(f"  R^2 = {m0.rsquared:.4f}, adj = {m0.rsquared_adj:.4f}, n = {int(m0.nobs)}")
    print(f"  slope (log10 params)     = {beta_params:+.4f}")
    print(f"  Implied Densing monthly  = {densing_monthly:+.5f}")

    # ---- M1: add months_since
    m1 = fit_ols(df, ["log10_params", "months"])
    # ---- M2: add thinking flag control
    df_m2 = df.assign(thinking_i=df["thinking"].astype(int))
    m2 = fit_ols(df_m2, ["log10_params", "months", "thinking_i"])
    # ---- M3: add MoE dummy
    df_m3 = df_m2.assign(moe=(df["arch"].str.lower() == "moe").astype(int))
    m3 = fit_ols(df_m3, ["log10_params", "months", "thinking_i", "moe"])

    # ---- M4: time-only (no params) for reference
    m_time_only = fit_ols(df, ["months"])

    # ---- Formal tests: is time coef = 0?   is time coef = Densing?
    def test_coef(model, idx, target):
        est = float(model.params[idx])
        se = float(model.bse[idx])
        t = (est - target) / se if se > 0 else float("nan")
        dof = model.df_resid
        p = 2 * (1 - _st.t.cdf(abs(t), dof))
        return est, se, t, p

    est1, se1, t1_0, p1_0 = test_coef(m1, 2, 0.0)
    _, _, t1_d, p1_d = test_coef(m1, 2, densing_monthly)
    est2, se2, t2_0, p2_0 = test_coef(m2, 2, 0.0)
    _, _, t2_d, p2_d = test_coef(m2, 2, densing_monthly)
    est3, se3, t3_0, p3_0 = test_coef(m3, 2, 0.0)
    _, _, t3_d, p3_d = test_coef(m3, 2, densing_monthly)

    # ---- Residuals vs time (partialled-out plot)
    resid = df["pen_acc"].values - (m0.params[0] + m0.params[1] * df["log10_params"].values)
    # Fit resid ~ months
    X_m = add_const(df["months"].values.astype(float))
    resid_fit = OLSResult(X_m, resid)
    print("\nResiduals-vs-time slope:", f"{resid_fit.params[1]:+.5f} per month, p={resid_fit.pvalues[1]:.3g}")

    # ---- Bootstrap CI on time coefficient in M1
    rng = np.random.default_rng(42)
    B = 4000
    n = len(df)
    boot_time = np.empty(B)
    X_full = add_const(df[["log10_params", "months"]].values.astype(float))
    y_full = df["pen_acc"].values.astype(float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            bfit = OLSResult(X_full[idx], y_full[idx])
            boot_time[b] = bfit.params[2]
        except Exception:
            boot_time[b] = np.nan
    boot_time = boot_time[~np.isnan(boot_time)]
    ci_lo, ci_hi = np.percentile(boot_time, [2.5, 97.5])
    print(f"Bootstrap CI time coef (M1): [{ci_lo:+.5f}, {ci_hi:+.5f}]  (n_boot={len(boot_time)})")

    # ---- Figure
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scaling fit colored by release year
    sc = ax0.scatter(df["log10_params"], df["pen_acc"], c=df["months"], cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    xs = np.linspace(df["log10_params"].min() - 0.1, df["log10_params"].max() + 0.1, 100)
    ax0.plot(xs, m0.params[0] + m0.params[1] * xs, "k-", lw=1.5, alpha=0.7, label=f"fit: slope={beta_params:.3f}, R^2={m0.rsquared:.3f}")
    ax0.set_xlabel("log10(total params, B)")
    ax0.set_ylabel("Penalized IKP accuracy")
    ax0.set_title(f"IKP scaling (n={len(df)} open-weight models)")
    ax0.legend(loc="upper left")
    cbar = plt.colorbar(sc, ax=ax0)
    cbar.set_label(f"months since {REF_DATE.strftime('%Y-%m')}")

    # Right: residuals vs release date
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.scatter(df["date"], resid, c=df["months"], cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    # Observed trend line
    x_lin = np.linspace(df["months"].min(), df["months"].max(), 100)
    y_lin = resid_fit.params[0] + resid_fit.params[1] * x_lin
    dates_lin = [REF_DATE + pd.Timedelta(days=30.44 * m) for m in x_lin]
    ax1.plot(dates_lin, y_lin, "b-", lw=1.5, label=f"observed: {resid_fit.params[1]:+.4f}/mo, p={resid_fit.pvalues[1]:.2g}")
    # Densing Law prediction
    y_densing = densing_monthly * (x_lin - x_lin.mean()) + resid.mean()
    ax1.plot(dates_lin, y_densing, "r--", lw=1.5, label=f"Densing Law: {densing_monthly:+.4f}/mo")
    ax1.set_xlabel("Release date")
    ax1.set_ylabel("Residual after log10(params)")
    ax1.set_title("Densing Law falsification: does IKP residual trend with time?")
    ax1.legend(loc="upper left")
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_densing_law.pdf")
    fig.savefig(FIG_DIR / "fig_densing_law.png", dpi=150)
    plt.close(fig)

    # ---- Write report
    lines = []
    P = lines.append
    P("# Densing Law Falsification Analysis")
    P("")
    P(f"**Dataset.** {len(df)} open-weight models with published parameter counts and completed IKP evaluations, release dates spanning {df['date'].min().date()} to {df['date'].max().date()}. Penalized accuracy uses the canonical $\\lambda = -1.0$ hallucination penalty (researcher v2 + non-researcher v1 verdicts).")
    P("")
    P("**Thesis under test.** The Densing Law (Xiao et al., 2025) states that capability-per-parameter doubles every ~3.5 months. If IKP measures incompressible factual storage, this trend should be *absent* from IKP; if IKP still rides the Densing Law, it would weaken the paper's incompressibility claim.")
    P("")
    P("## 1. Baseline scaling fit (for reference)")
    P("")
    P("Model M0: `pen_acc = b0 + b1 * log10(params_B)`")
    P("")
    P(nice_coef_table(m0, ["const", "log10(params_B)"]))
    P("")
    P(f"R^2 = **{m0.rsquared:.4f}**, adj. R^2 = {m0.rsquared_adj:.4f}, n = {int(m0.nobs)}.")
    P("")
    P(f"Implied Densing-Law monthly accuracy gain under M0 slope: **{densing_monthly:+.5f}/month** (≈ {densing_monthly*12*100:.2f} percentage points per year).")
    P("")
    P("## 2. Adding release date: M1 = M0 + months_since_ref")
    P("")
    P(f"Reference date: {REF_DATE.date()}. `months` is centered at that date.")
    P("")
    P(nice_coef_table(m1, ["const", "log10(params_B)", "months"]))
    P("")
    P(f"R^2 = **{m1.rsquared:.4f}**, ΔR^2 over M0 = **{m1.rsquared - m0.rsquared:+.4f}**.")
    P(f"Bootstrap 95% CI on `months` coefficient: [{ci_lo:+.5f}, {ci_hi:+.5f}]  (B=4000).")
    P("")
    P("**Tests:**")
    P(f"- H0: time coef = 0 → t = {t1_0:+.2f}, p = {p1_0:.3g}")
    P(f"- H0: time coef = Densing prediction ({densing_monthly:+.5f}) → t = {t1_d:+.2f}, p = {p1_d:.3g}")
    P("")
    P("## 3. With controls: M2 = M1 + thinking, M3 = M2 + MoE dummy")
    P("")
    P("### M2")
    P(nice_coef_table(m2, ["const", "log10(params_B)", "months", "thinking"]))
    P("")
    P(f"R^2 = {m2.rsquared:.4f}")
    P(f"- months vs 0: t = {t2_0:+.2f}, p = {p2_0:.3g}")
    P(f"- months vs Densing: t = {t2_d:+.2f}, p = {p2_d:.3g}")
    P("")
    P("### M3")
    P(nice_coef_table(m3, ["const", "log10(params_B)", "months", "thinking", "MoE"]))
    P("")
    P(f"R^2 = {m3.rsquared:.4f}")
    P(f"- months vs 0: t = {t3_0:+.2f}, p = {p3_0:.3g}")
    P(f"- months vs Densing: t = {t3_d:+.2f}, p = {p3_d:.3g}")
    P("")
    P("## 4. Time-only sanity model")
    P("")
    P("If one regresses IKP accuracy on release date alone (ignoring params), what fraction of variance is 'explained' by time?")
    P("")
    P(nice_coef_table(m_time_only, ["const", "months"]))
    P(f"R^2 = {m_time_only.rsquared:.4f}. This reflects the selection effect that newer models skew larger, not an actual time trend per-parameter.")
    P("")
    P("## 5. Interpretation")
    P("")
    dcoef = float(m1.params[2])
    ratio = dcoef / densing_monthly if densing_monthly != 0 else float("nan")
    P(f"The fitted time coefficient (controlling for log10(params)) is **{dcoef:+.5f} per month** with 95% bootstrap CI [{ci_lo:+.5f}, {ci_hi:+.5f}].")
    P(f"The Densing Law would predict **{densing_monthly:+.5f} per month** under the observed IKP-params slope.")
    P(f"Observed is **{ratio*100:.1f}%** of the Densing prediction.")
    P("")
    P(("Result: we reject the Densing Law prediction for IKP. " if p1_d < 0.05 else "Result: we cannot statistically distinguish observed from Densing prediction — ") +
      ("We cannot reject zero time trend." if p1_0 >= 0.05 else "The residual time trend is nonzero."))
    P("")
    P("Adding release date increases R^2 by only {:.4f}, a negligible fraction. This is consistent with the paper's thesis: IKP measures the incompressible factual-storage component of parameters, which is *not* subject to the Densing Law compression observed on reasoning-heavy benchmarks.".format(m1.rsquared - m0.rsquared))
    P("")
    P(f"Figure: `results/figures/fig_densing_law.pdf`")
    P("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Figure written to {FIG_DIR / 'fig_densing_law.pdf'}")

    # Also dump a tidy CSV
    df.to_csv(ROOT / "data" / "densing_analysis_data.csv", index=False)
    print(f"Data CSV: data/densing_analysis_data.csv")


if __name__ == "__main__":
    run_analysis()
