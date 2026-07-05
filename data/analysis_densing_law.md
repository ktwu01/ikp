# Densing Law Falsification Analysis

**Dataset.** 100 open-weight models with published parameter counts and completed IKP evaluations, release dates spanning 2023-09-27 to 2026-06-12. Accuracy uses no-penalty scoring ($\lambda = 0$).

**Thesis under test.** The Densing Law (Xiao et al., 2025) states that capability-per-parameter doubles every ~3.5 months. If IKP measures incompressible factual storage, this trend should be *absent* from IKP; if IKP still rides the Densing Law, it would weaken the paper's incompressibility claim.

## 1. Baseline scaling fit (for reference)

Model M0: `pen_acc = b0 + b1 * log10(params_B)`

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.2464 | 0.0140 | +17.60 | 0 | +0.2186 | +0.2742
log10(params_B) | +0.1494 | 0.0072 | +20.77 | 0 | +0.1352 | +0.1637

R^2 = **0.8148**, adj. R^2 = 0.8129, n = 100.

Implied Densing-Law monthly accuracy gain under M0 slope: **+0.01285/month** (≈ 15.42 percentage points per year).

## 2. Adding release date: M1 = M0 + months_since_ref

Reference date: 2024-01-01. `months` is centered at that date.

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.2327 | 0.0173 | +13.43 | 0 | +0.1983 | +0.2671
log10(params_B) | +0.1445 | 0.0081 | +17.89 | 0 | +0.1285 | +0.1605
months | +0.0013 | 0.0010 | +1.32 | 0.188 | -0.0007 | +0.0033

R^2 = **0.8181**, ΔR^2 over M0 = **+0.0033**.
Bootstrap 95% CI on `months` coefficient: [-0.00037, +0.00326]  (B=4000).

**Tests:**
- H0: time coef = 0 → t = +1.32, p = 0.188
- H0: time coef = Densing prediction (+0.01285) → t = -11.64, p = 0

## 3. With controls: M2 = M1 + thinking, M3 = M2 + MoE dummy

### M2
term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.2367 | 0.0173 | +13.68 | 0 | +0.2024 | +0.2711
log10(params_B) | +0.1421 | 0.0081 | +17.51 | 0 | +0.1260 | +0.1582
months | +0.0008 | 0.0010 | +0.73 | 0.465 | -0.0013 | +0.0028
thinking | +0.0277 | 0.0159 | +1.75 | 0.0841 | -0.0038 | +0.0592

R^2 = 0.8237
- months vs 0: t = +0.73, p = 0.465
- months vs Densing: t = -11.72, p = 0

### M3
term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.2375 | 0.0190 | +12.52 | 0 | +0.1999 | +0.2752
log10(params_B) | +0.1415 | 0.0103 | +13.80 | 0 | +0.1211 | +0.1618
months | +0.0007 | 0.0011 | +0.64 | 0.521 | -0.0015 | +0.0029
thinking | +0.0276 | 0.0160 | +1.73 | 0.0869 | -0.0041 | +0.0593
MoE | +0.0022 | 0.0210 | +0.10 | 0.918 | -0.0395 | +0.0438

R^2 = 0.8237
- months vs 0: t = +0.64, p = 0.521
- months vs Densing: t = -10.93, p = 0

## 4. Time-only sanity model

If one regresses IKP accuracy on release date alone (ignoring params), what fraction of variance is 'explained' by time?

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.3409 | 0.0335 | +10.17 | 0 | +0.2744 | +0.4074
months | +0.0095 | 0.0018 | +5.23 | 9.72e-07 | +0.0059 | +0.0131
R^2 = 0.2180. This reflects the selection effect that newer models skew larger, not an actual time trend per-parameter.

## 5. Interpretation

The fitted time coefficient (controlling for log10(params)) is **+0.00131 per month** with 95% bootstrap CI [-0.00037, +0.00326].
The Densing Law would predict **+0.01285 per month** under the observed IKP-params slope.
Observed is **10.2%** of the Densing prediction.

Result: we reject the Densing Law prediction for IKP. We cannot reject zero time trend.

Adding release date increases R^2 by only 0.0033, a negligible fraction. This is consistent with the paper's thesis: IKP measures the incompressible factual-storage component of parameters, which is *not* subject to the Densing Law compression observed on reasoning-heavy benchmarks.

Figure: `results/figures/fig_densing_law.pdf`
