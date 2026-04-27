# Densing Law Falsification Analysis

**Dataset.** 81 open-weight models with published parameter counts and completed IKP evaluations, release dates spanning 2023-09-27 to 2026-04-22. Penalized accuracy uses the standard -0.5 hallucination penalty.

**Thesis under test.** The Densing Law (Xiao et al., 2025) states that capability-per-parameter doubles every ~3.5 months. If IKP measures incompressible factual storage, this trend should be *absent* from IKP; if IKP still rides the Densing Law, it would weaken the paper's incompressibility claim.

## 1. Baseline scaling fit (for reference)

Model M0: `pen_acc = b0 + b1 * log10(params_B)`

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1683 | 0.0190 | +8.84 | 1.99e-13 | +0.1304 | +0.2062
log10(params_B) | +0.1500 | 0.0100 | +14.95 | 0 | +0.1300 | +0.1699

R^2 = **0.7389**, adj. R^2 = 0.7356, n = 81.

Implied Densing-Law monthly accuracy gain under M0 slope: **+0.01290/month** (≈ 15.48 percentage points per year).

## 2. Adding release date: M1 = M0 + months_since_ref

Reference date: 2024-01-01. `months` is centered at that date.

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1729 | 0.0217 | +7.97 | 1.1e-11 | +0.1297 | +0.2161
log10(params_B) | +0.1520 | 0.0111 | +13.74 | 0 | +0.1300 | +0.1741
months | -0.0005 | 0.0012 | -0.45 | 0.653 | -0.0029 | +0.0018

R^2 = **0.7396**, ΔR^2 over M0 = **+0.0007**.
Bootstrap 95% CI on `months` coefficient: [-0.00305, +0.00160]  (B=4000).

**Tests:**
- H0: time coef = 0 → t = -0.45, p = 0.653
- H0: time coef = Densing prediction (+0.01290) → t = -11.47, p = 0

## 3. With controls: M2 = M1 + thinking, M3 = M2 + MoE dummy

### M2
term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1802 | 0.0223 | +8.07 | 7.39e-12 | +0.1358 | +0.2247
log10(params_B) | +0.1495 | 0.0112 | +13.36 | 0 | +0.1272 | +0.1717
months | -0.0014 | 0.0013 | -1.02 | 0.311 | -0.0040 | +0.0013
thinking | +0.0258 | 0.0198 | +1.30 | 0.196 | -0.0136 | +0.0652

R^2 = 0.7453
- months vs 0: t = -1.02, p = 0.311
- months vs Densing: t = -10.75, p = 0

### M3
term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1731 | 0.0248 | +6.97 | 1.01e-09 | +0.1236 | +0.2226
log10(params_B) | +0.1551 | 0.0141 | +10.99 | 0 | +0.1270 | +0.1833
months | -0.0011 | 0.0014 | -0.81 | 0.423 | -0.0039 | +0.0016
thinking | +0.0266 | 0.0199 | +1.34 | 0.184 | -0.0130 | +0.0663
MoE | -0.0156 | 0.0234 | -0.66 | 0.508 | -0.0623 | +0.0311

R^2 = 0.7467
- months vs 0: t = -0.81, p = 0.423
- months vs Densing: t = -10.16, p = 8.88e-16

## 4. Time-only sanity model

If one regresses IKP accuracy on release date alone (ignoring params), what fraction of variance is 'explained' by time?

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.3330 | 0.0337 | +9.89 | 1.78e-15 | +0.2660 | +0.4000
months | +0.0061 | 0.0020 | +3.11 | 0.0026 | +0.0022 | +0.0100
R^2 = 0.1091. This reflects the selection effect that newer models skew larger, not an actual time trend per-parameter.

## 5. Interpretation

The fitted time coefficient (controlling for log10(params)) is **-0.00053 per month** with 95% bootstrap CI [-0.00305, +0.00160].
The Densing Law would predict **+0.01290 per month** under the observed IKP-params slope.
Observed is **-4.1%** of the Densing prediction.

Result: we reject the Densing Law prediction for IKP. We cannot reject zero time trend.

Adding release date increases R^2 by only 0.0007, a negligible fraction. This is consistent with the paper's thesis: IKP measures the incompressible factual-storage component of parameters, which is *not* subject to the Densing Law compression observed on reasoning-heavy benchmarks.

Figure: `results/figures/fig_densing_law.pdf`
