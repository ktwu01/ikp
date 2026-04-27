# Densing Law Falsification Analysis

**Dataset.** 96 open-weight models with published parameter counts and completed IKP evaluations, release dates spanning 2023-09-27 to 2026-04-24. Penalized accuracy uses the canonical $\lambda = -1.0$ hallucination penalty (researcher v2 + non-researcher v1 verdicts).

**Thesis under test.** The Densing Law (Xiao et al., 2025) states that capability-per-parameter doubles every ~3.5 months. If IKP measures incompressible factual storage, this trend should be *absent* from IKP; if IKP still rides the Densing Law, it would weaken the paper's incompressibility claim.

## 1. Baseline scaling fit (for reference)

Model M0: `pen_acc = b0 + b1 * log10(params_B)`

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1358 | 0.0149 | +9.10 | 1.51e-14 | +0.1062 | +0.1655
log10(params_B) | +0.1362 | 0.0079 | +17.31 | 0 | +0.1206 | +0.1518

R^2 = **0.7612**, adj. R^2 = 0.7586, n = 96.

Implied Densing-Law monthly accuracy gain under M0 slope: **+0.01171/month** (≈ 14.06 percentage points per year).

## 2. Adding release date: M1 = M0 + months_since_ref

Reference date: 2024-01-01. `months` is centered at that date.

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1470 | 0.0189 | +7.78 | 9.52e-12 | +0.1095 | +0.1845
log10(params_B) | +0.1397 | 0.0087 | +16.11 | 0 | +0.1225 | +0.1569
months | -0.0010 | 0.0011 | -0.96 | 0.338 | -0.0032 | +0.0011

R^2 = **0.7635**, ΔR^2 over M0 = **+0.0024**.
Bootstrap 95% CI on `months` coefficient: [-0.00308, +0.00079]  (B=4000).

**Tests:**
- H0: time coef = 0 → t = -0.96, p = 0.338
- H0: time coef = Densing prediction (+0.01171) → t = -11.85, p = 0

## 3. With controls: M2 = M1 + thinking, M3 = M2 + MoE dummy

### M2
term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1485 | 0.0194 | +7.65 | 1.93e-11 | +0.1099 | +0.1870
log10(params_B) | +0.1390 | 0.0089 | +15.54 | 0 | +0.1212 | +0.1567
months | -0.0012 | 0.0012 | -1.02 | 0.308 | -0.0035 | +0.0011
thinking | +0.0066 | 0.0182 | +0.36 | 0.718 | -0.0295 | +0.0427

R^2 = 0.7639
- months vs 0: t = -1.02, p = 0.308
- months vs Densing: t = -11.04, p = 0

### M3
term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.1341 | 0.0208 | +6.44 | 5.63e-09 | +0.0927 | +0.1754
log10(params_B) | +0.1507 | 0.0110 | +13.66 | 0 | +0.1288 | +0.1726
months | -0.0004 | 0.0012 | -0.36 | 0.72 | -0.0029 | +0.0020
thinking | +0.0079 | 0.0180 | +0.44 | 0.663 | -0.0278 | +0.0436
MoE | -0.0396 | 0.0223 | -1.78 | 0.079 | -0.0838 | +0.0047

R^2 = 0.7718
- months vs 0: t = -0.36, p = 0.72
- months vs Densing: t = -9.87, p = 4.44e-16

## 4. Time-only sanity model

If one regresses IKP accuracy on release date alone (ignoring params), what fraction of variance is 'explained' by time?

term | coef | std_err | t | p | ci_lo | ci_hi
--- | --- | --- | --- | --- | --- | ---
const | +0.2587 | 0.0340 | +7.60 | 2.13e-11 | +0.1911 | +0.3262
months | +0.0062 | 0.0019 | +3.29 | 0.0014 | +0.0025 | +0.0100
R^2 = 0.1033. This reflects the selection effect that newer models skew larger, not an actual time trend per-parameter.

## 5. Interpretation

The fitted time coefficient (controlling for log10(params)) is **-0.00104 per month** with 95% bootstrap CI [-0.00308, +0.00079].
The Densing Law would predict **+0.01171 per month** under the observed IKP-params slope.
Observed is **-8.9%** of the Densing prediction.

Result: we reject the Densing Law prediction for IKP. We cannot reject zero time trend.

Adding release date increases R^2 by only 0.0024, a negligible fraction. This is consistent with the paper's thesis: IKP measures the incompressible factual-storage component of parameters, which is *not* subject to the Densing Law compression observed on reasoning-heavy benchmarks.

Figure: `results/figures/fig_densing_law.pdf`
