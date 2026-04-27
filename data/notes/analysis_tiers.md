# IKP Tier Design and Probe Quality Analysis

Analysis of 118 models evaluated across 1,400 probes (200 per tier, T1--T7).
Probe set: `final_probe_set_v8.json`. Results: `evaluation_summary.json`.

---

## 1. Tier Discrimination Power

### 1.1 Accuracy distribution across all 118 models

| Tier | Mean   | Median | StdDev | Min    | Max    | IQR    | Variance  |
|------|--------|--------|--------|--------|--------|--------|-----------|
| T1   | 0.9630 | 0.9875 | 0.1101 | 0.0000 | 1.0000 | 0.0225 | 0.012113  |
| T2   | 0.9266 | 0.9775 | 0.1504 | 0.0000 | 1.0000 | 0.0450 | 0.022628  |
| T3   | 0.6952 | 0.8200 | 0.3033 | 0.0000 | 0.9925 | 0.3875 | 0.092009  |
| T4   | 0.4738 | 0.5275 | 0.3347 | 0.0000 | 0.9775 | 0.6625 | 0.112013  |
| T5   | 0.2437 | 0.1400 | 0.2809 | 0.0000 | 0.9700 | 0.4575 | 0.078879  |
| T6   | 0.0482 | 0.0000 | 0.1322 | 0.0000 | 0.9025 | 0.0400 | 0.017472  |
| T7   | 0.0000 | 0.0000 | 0.0005 | 0.0000 | 0.0050 | 0.0000 | 0.000000  |

**Key finding:** T4 has the highest variance (0.1120) and best discriminates across the full model population. T3 has the second-highest variance (0.0920). T1 and T2 are saturated (ceiling effect), T6 and T7 exhibit floor effects. The Gini coefficient progression confirms this:

| Tier | Gini  |
|------|-------|
| T1   | 0.012 |
| T2   | 0.041 |
| T3   | 0.214 |
| T4   | 0.388 |
| T5   | 0.599 |
| T6   | 0.842 |
| T7   | 0.975 |

### 1.2 Spearman correlation with overall IKP accuracy

| Tier | Spearman rho |
|------|-------------|
| T1   | 0.720       |
| T2   | 0.782       |
| T3   | **0.978**   |
| T4   | **0.963**   |
| T5   | 0.873       |
| T6   | 0.258       |
| T7   | -0.956      |

T3 and T4 are the strongest rank-order predictors of overall model quality. T7 shows a strong negative Spearman correlation because its penalized accuracy is dominated by hallucination penalties rather than knowledge.

### 1.3 Discrimination by parameter range

**< 3B (5 models):** Only T1 (mean 0.85) and T2 (mean 0.53) discriminate. All higher tiers are zero.

**3--10B (12 models):** T2 (mean 0.85) and T3 (mean 0.23) differentiate within this range. T4 mean is 0.03.

**10--35B (15 models):** T3 is the discriminating tier (mean 0.56, sd 0.16). T4 starts appearing (mean 0.09).

**35--80B (8 models):** T3 nears saturation (mean 0.77). T4 (mean 0.47) and T5 (mean 0.14) discriminate.

**80--250B (7 models):** T4 (mean 0.51) and T5 (mean 0.06) are the active tiers.

**> 250B / Frontier API (58+ models):** T4 (mean 0.65--0.67), T5 (mean 0.37--0.39), and T6 (mean 0.04--0.09) discriminate. The top models (gemini-3.1-pro, grok-4, gpt-5-think) are separated by T5 and T6.

**Summary:** Each tier becomes "useful" at approximately:
- T1: All models (floor ~0.7 even at 1B)
- T2: >= 1B parameters
- T3: >= 7B parameters
- T4: >= 25B parameters
- T5: >= 70B parameters
- T6: >= frontier-class models only
- T7: No model reliably passes (see Section 3)

---

## 2. Tier Monotonicity

### 2.1 Overall violation rate

| Metric | Value |
|--------|-------|
| Total tier pairs checked (6 per model x 118 models) | 708 |
| Total monotonicity violations | 25 |
| Violation rate | 3.53% |
| Models with >= 1 violation | 24 / 118 (20.3%) |

### 2.2 Violations by tier pair

| Pair   | Violations | Pct of models |
|--------|-----------|---------------|
| T1 > T2 | 20      | 17.1%         |
| T2 > T3 | 3       | 2.6%          |
| T3 > T4 | 0       | 0.0%          |
| T4 > T5 | 2       | 1.7%          |
| T5 > T6 | 0       | 0.0%          |
| T6 > T7 | 0       | 0.0%          |

### 2.3 Analysis of violations

**T1 > T2 violations (20 models):** These are the most common, but nearly all are small (< 2%). The largest is llama-3.1-70b (T1=0.858, T2=0.985, gap=0.128), caused by an anomalously high T1 refusal rate (27 refusals on T1 vs 0 on T2). This is a model behavioral artifact -- the model refuses easy trivia but answers harder questions. Most other T1>T2 violations are within 1--2 percentage points and reflect the near-identical difficulty of these two tiers at the top end.

**T2 > T3 violations (3 models):**
- minimax-m1-think: T2=0.420, T3=0.500 (gap 0.08). This model has extreme hallucination rates (76 wrong on T2 vs 56 on T3), making its penalized T3 higher than T2.
- glm-5.1-think: T2=0.970, T3=0.975 (gap 0.005). Negligible.
- gpt-4.1: T2=0.978, T3=0.985 (gap 0.008). Negligible.

**T4 > T5 violations (2 models):**
- gemini-3-flash: T4=0.933, T5=0.955 (gap 0.023)
- gemini-3-flash-think: T4=0.940, T5=0.970 (gap 0.030)

These Gemini 3 models show a genuine T4>T5 inversion, likely due to the Gemini training data containing very strong coverage of the T5 probe domains (computer science researchers, founding years) while having gaps on some T4 geography/culture probes.

**Conclusion:** Monotonicity is well-preserved. The 3.5% violation rate is low, and most violations are < 2 percentage points. The T1/T2 boundary is effectively indistinguishable for strong models. No violations occur in the T3--T7 range (except the Gemini outliers at T4/T5), confirming the tier ordering is sound for the core discriminating tiers.

---

## 3. T7 Ceiling Effect

### 3.1 Key statistics

| Metric | Value |
|--------|-------|
| Models with any T7 correct (raw) | 112 / 118 (94.9%) |
| Models with penalized T7 > 0 | **1** / 118 (0.8%) |
| Only model with T7_pen > 0 | claude-sonnet-4-think (0.005) |
| Mean raw T7 correct count | 9.8 / 200 |
| Mean penalized T7 accuracy | -0.160 |
| T7 probes with >= 1 correct answer | 147 / 200 (73.5%) |
| T7 probes with 0 correct answers | 53 / 200 (26.5%) |

### 3.2 Raw T7 correct count distribution

| Correct answers | Models |
|----------------|--------|
| 0              | 5      |
| 1              | 9      |
| 2--4           | 23     |
| 5--9           | 33     |
| 10--14         | 20     |
| 15--19         | 13     |
| 20--29         | 12     |
| 30--49         | 3      |

The top T7 raw performers are gemini-3-flash-think (45/200), gemini-3-flash (32/200), gemini-2.5-pro-think (30/200), and glm-4.6-think (27/200).

### 3.3 T7 is dominated by hallucination

The critical issue with T7 is that **wrong answers vastly outnumber correct ones**:

- Total T7 responses across all models: 23,600
- Correct: 1,157 (4.9%)
- Wrong: 9,860 (41.8%)
- Refusal: 12,583 (53.3%)

For comparison, T6: correct 6.6%, wrong 20.5%, refusal 72.9%.

The wrong rate at T7 (41.8%) is roughly 8.5x the correct rate (4.9%). After applying the -0.5 hallucination penalty, virtually all models go negative. The only model with positive penalized T7 is claude-sonnet-4-think, barely above zero at 0.005.

### 3.4 Does T7 correlate with model quality?

| Metric | Pearson r with log(params) |
|--------|---------------------------|
| T7 raw correct rate | 0.074 |
| T7 wrong rate | -0.497 |
| T7 refusal rate | 0.460 |
| T7 penalized accuracy | 0.571 |

T7 raw correct rate shows near-zero correlation with model size (r=0.074). Instead, what predicts T7 performance is the wrong/refusal tradeoff: larger models refuse more and hallucinate less. The penalized T7 accuracy (r=0.571 with size) is driven entirely by hallucination avoidance, not knowledge.

### 3.5 T7 probes most often answered correctly

| Probe ID | Domain | Correct by N models | Question |
|----------|--------|--------------------:|----------|
| IKP_T7_1221 | computer_science | 70 | Research subfield of Ivan Beschastnikh |
| IKP_T7_1393 | founding_year | 42 | Year Wollo University was founded |
| IKP_T7_1334 | places | 40 | Year Camp Babbitt was founded |
| IKP_T7_1240 | computer_science | 38 | Research subfield of Insik Shin |
| IKP_T7_1226 | computer_science | 36 | Research subfield of Rajesh Bordawekar |
| IKP_T7_1275 | computer_science | 35 | Research subfield of Chenren Xu |

IKP_T7_1221 (Ivan Beschastnikh, difficulty 59.8%) is a clear misplacement -- this probe is easier than many T5 probes. Other probes like IKP_T7_1393 (35.9%) and IKP_T7_1334 (34.2%) are also too easy for T7.

### 3.6 Conclusion on T7

**T7 is too hard in its current form.** It measures hallucination avoidance rather than knowledge. Only 1/118 models achieves positive penalized accuracy. The tier contributes no useful discrimination between models on the knowledge dimension. However, 30 T7 probes have difficulty > 10%, and a few (like IKP_T7_1221 at 59.8%) are clearly misclassified.

**Recommendation:** Either (a) drop T7 from the overall scoring and use it only as a hallucination analysis tool, (b) move the ~30 probes with difficulty > 10% to T6 and accept T7 as ultra-hard, or (c) recalibrate T7 probes using the empirical difficulty data.

---

## 4. Domain Analysis Within Tiers

### 4.1 Domain difficulty by tier

Domains are normalized into broader categories. "Mean Diff" is the average fraction of models answering correctly.

**T1 (all domains > 93% correct):** Near-uniform difficulty. Hardest: publications (93.8%), founding years (96.2%). Easiest: general (98.5%), people (99.2%).

**T2:** engineering drops notably (88.8% mean) due to probes like "first transatlantic cable" (47.9%). Geography varies widely (65.8%--99.2%). Most domains remain > 90%.

**T3:** Clear separation emerges:
- Easiest: events (99.2%), history (93.6%), institutions (91.9%)
- Hardest: museum (58.1%), founding_year (66.8%), bridge (70.1%)
- computer_science (72.0%) and journal (76.9%) are in the mid-range

**T4:** Strong domain stratification:
- Easiest: currency (96.6%), people (85.0%), architecture (79.1%)
- Hardest: engineering (46.2%), founding_year (53.8%), geography (55.6%)
- computer_science (59.1%) remains mid-range

**T5:** Most probes are computer_science (100) or founding_year (65):
- journal (59.5%) is the easiest
- geography (24.8%) is the hardest

**T6 and T7:** Dominated by computer_science and founding_year probes. Difficulty is uniformly low. In T7, computer_science (6.8%) is actually easier than founding_year (2.8%) -- the researcher subfield probes are partially guessable.

### 4.2 Key domain patterns

1. **Founding years have a smooth difficulty gradient:** T3 (66.8%) -> T4 (53.8%) -> T5 (33.6%) -> T6 (11.6%) -> T7 (2.8%). This domain works well across the tier structure.

2. **Computer science researcher probes are well-calibrated in T3--T5** (72.0% -> 59.1% -> 33.3%) but become noisy in T6--T7 (18.1% -> 6.8%) because models can guess "systems" or "machine learning" as subfields.

3. **General knowledge / history / culture probes cluster in T1--T3** and are consistently easy. They provide limited discrimination beyond confirming the model has basic world knowledge.

4. **Geography has high within-tier variance** at every level, suggesting heterogeneous difficulty within this domain.

---

## 5. Probe Discrimination

### 5.1 Point-biserial correlation by tier

| Tier | Mean r_pb | Median r_pb | StdDev | Min    | Max    | N <= 0 |
|------|-----------|-------------|--------|--------|--------|--------|
| T1   | 0.308     | 0.295       | 0.054  | 0.130  | 0.535  | 0      |
| T2   | 0.374     | 0.355       | 0.107  | 0.053  | 0.674  | 0      |
| T3   | 0.542     | 0.574       | 0.125  | 0.027  | 0.778  | 0      |
| T4   | **0.591** | **0.626**   | 0.122  | 0.116  | 0.791  | 0      |
| T5   | 0.542     | 0.574       | 0.133  | 0.005  | 0.756  | 0      |
| T6   | 0.367     | 0.371       | 0.145  | -0.043 | 0.682  | 2      |
| T7   | 0.056     | 0.000       | 0.160  | -0.235 | 0.675  | 109    |

**T4 probes have the highest average discrimination** (mean r_pb = 0.591), followed by T3 and T5 (both 0.542). T1 probes have low discrimination (0.308) because nearly all models get them right. T7 probes are largely noise: 109/200 have zero or negative correlation.

### 5.2 Overall probe quality

| Category | Count | Percentage |
|----------|------:|----------:|
| r_pb > 0.2 (strong discriminators) | 1,191 | 85.1% |
| r_pb = 0 (no signal) | 54 | 3.9% |
| r_pb < 0 (negative / noise) | 57 | 4.1% |

85% of all probes are strong discriminators. Nearly all noise probes (111 / 111 with r_pb <= 0) are in T6 and T7.

### 5.3 Top 10 most discriminating probes

| Probe ID | Tier | r_pb | Difficulty | Domain | Question (truncated) |
|----------|------|------|-----------|--------|---------------------|
| IKP_T4_0618 | T4 | 0.791 | 0.641 | Classical Music | Violin Concerto 'To the Memory of an Angel' |
| IKP_T3_0507 | T3 | 0.778 | 0.709 | founding_year | Year University of Brescia founded |
| IKP_T5_0948 | T5 | 0.756 | 0.598 | journal | Year journal Historical Materialism published |
| IKP_T4_0709 | T4 | 0.751 | 0.564 | places | Year Equality Colony founded |
| IKP_T4_0662 | T4 | 0.747 | 0.709 | computer_science | Research subfield of ... |
| IKP_T5_0961 | T5 | 0.746 | 0.530 | founding_year | Year Westbrook College founded |
| IKP_T3_0543 | T3 | 0.744 | 0.752 | university | Year Jaume I University founded |
| IKP_T4_0732 | T4 | 0.744 | 0.556 | founding_year | Year Dominican University College founded |
| IKP_T3_0521 | T3 | 0.743 | 0.598 | founding_year | Year Stroganov Academy founded |
| IKP_T4_0616 | T4 | 0.741 | 0.692 | Classical Music | Symphony No. 4 'The Inextinguishable' |

The best discriminating probes are in T3--T5 with difficulty around 50--70% -- the classic "sweet spot" for item discrimination in psychometrics.

### 5.4 Worst 10 probes (noise)

All 10 worst probes are in T7 (computer_science domain) with r_pb between -0.235 and -0.168. These probes are answered correctly by some small models but missed by most large ones -- likely due to random guessing of research subfields.

---

## 6. Tier Boundary Analysis

### 6.1 Per-tier probe difficulty distribution

| Tier | Mean   | Median | StdDev | P10    | P90    | Min    | Max    | CV     |
|------|--------|--------|--------|--------|--------|--------|--------|--------|
| T1   | 0.973  | 0.983  | 0.039  | 0.957  | 0.992  | 0.658  | 0.992  | 0.040  |
| T2   | 0.948  | 0.966  | 0.059  | 0.889  | 0.992  | 0.479  | 0.992  | 0.063  |
| T3   | 0.771  | 0.795  | 0.140  | 0.590  | 0.957  | 0.231  | 0.992  | 0.183  |
| T4   | 0.580  | 0.598  | 0.173  | 0.342  | 0.795  | 0.103  | 0.992  | 0.300  |
| T5   | 0.337  | 0.316  | 0.162  | 0.137  | 0.564  | 0.051  | 0.829  | 0.483  |
| T6   | 0.135  | 0.094  | 0.118  | 0.026  | 0.291  | 0.009  | 0.692  | 0.881  |
| T7   | 0.049  | 0.026  | 0.077  | 0.000  | 0.128  | 0.000  | 0.598  | 1.573  |

Within-tier variability increases monotonically from T1 (CV = 0.04) to T7 (CV = 1.57). This is expected: higher tiers test niche knowledge where individual probe difficulty is harder to calibrate.

### 6.2 Cross-tier overlap

| Boundary | Probes in harder tier easier than easier tier's median | Probes in easier tier harder than harder tier's median |
|----------|-------------------------------------------------------|-------------------------------------------------------|
| T1/T2    | 35 / 200 (17.5%)                                      | 26 / 200 (13.0%)                                     |
| T2/T3    | 13 / 200 (6.5%)                                       | 3 / 200 (1.5%)                                       |
| T3/T4    | 19 / 200 (9.5%)                                       | 21 / 200 (10.5%)                                     |
| T4/T5    | 12 / 200 (6.0%)                                       | 17 / 200 (8.5%)                                      |
| T5/T6    | 17 / 200 (8.5%)                                       | 6 / 200 (3.0%)                                       |
| T6/T7    | 30 / 200 (15.0%)                                      | 16 / 200 (8.0%)                                      |

### 6.3 Interpretation

**T1/T2 boundary is weak:** 17.5% of T2 probes are easier than the T1 median, and 13% of T1 probes are harder. These two tiers are barely distinguishable in difficulty. This explains the frequent T1>T2 monotonicity violations.

**T2/T3 is the sharpest boundary:** Only 6.5% overlap from T3 to T2, and just 1.5% from T2 to T3. This clean separation corresponds to the jump from "common knowledge" to "specific factual knowledge."

**T3/T4 and T4/T5 have moderate overlap (6--10%).** This is acceptable given that these tiers target adjacent difficulty levels.

**T6/T7 has notable overlap:** 30 T7 probes (15%) are easier than T6's median. These include IKP_T7_1221 (59.8% correct -- easier than most T5 probes), IKP_T7_1393 (35.9%), and IKP_T7_1334 (34.2%).

### 6.4 Notable misplacements

**Probes too easy for their tier:**
- IKP_T4_0609 (T4, "Who founded Motown?" -- 99.2% correct): Should be T1 or T2
- IKP_T4_0758 (T4, "Currency of Eswatini" -- 96.6%): Should be T2
- IKP_T3_0558 (T3, "Currency of Mozambique" -- 99.2%): Should be T1
- IKP_T3_0544 (T3, "Camp David Accords year" -- 99.2%): Should be T1
- IKP_T7_1221 (T7, "Research subfield of Ivan Beschastnikh" -- 59.8%): Should be T4/T5

**Probes too hard for their tier:**
- IKP_T1_0133 (T1, "Year Oxford University founded" -- 65.8%): Should be T3
- IKP_T1_0043 (T1, "Small sea between Sweden and Finland" -- 74.4%): Should be T2
- IKP_T2_0347 (T2, "First transatlantic cable year" -- 47.9%): Should be T4
- IKP_T4_0737 (T4, "Lawrence Heritage State Park founding" -- 10.3%): Should be T6

---

## 7. Hallucination Penalty Impact

The IKP scoring applies a -0.5 penalty per wrong answer: `penalized = (correct - 0.5 * wrong) / total`.

### 7.1 Raw vs penalized accuracy by tier

| Tier | Mean Raw | Mean Penalized | Mean Penalty | Max Penalty | Mean Wrong% |
|------|----------|---------------|-------------|-------------|-------------|
| T1   | 0.970    | 0.963         | 0.007       | 0.165       | 1.5%        |
| T2   | 0.946    | 0.926         | 0.020       | 0.253       | 4.0%        |
| T3   | 0.765    | 0.681         | 0.084       | 0.443       | 16.8%       |
| T4   | 0.574    | 0.430         | 0.144       | 0.483       | 28.8%       |
| T5   | 0.333    | 0.170         | 0.163       | 0.480       | 32.6%       |
| T6   | 0.133    | **-0.072**    | 0.205       | 0.493       | 41.0%       |
| T7   | 0.049    | **-0.160**    | 0.208       | 0.485       | 41.7%       |

**T6 and T7 have negative mean penalized accuracy** because the average wrong rate exceeds twice the correct rate. The penalty is most impactful at T5--T7, where it shifts mean accuracy by 16--21 percentage points.

### 7.2 Wrong rate by model size

| Size bucket | T1 wrong | T3 wrong | T5 wrong | T7 wrong |
|-------------|----------|----------|----------|----------|
| < 3B        | 9.0%     | 73.3%    | 72.2%    | 71.6%    |
| 3--10B      | 2.6%     | 45.4%    | 57.7%    | 53.2%    |
| 10--35B     | 1.0%     | 26.7%    | 60.9%    | 54.2%    |
| 35--80B     | 0.8%     | 11.5%    | 22.4%    | 24.7%    |
| 80B+ (open) | 0.7%     | 6.3%     | 20.7%    | 27.8%    |
| API/Unknown | 1.1%     | 7.2%     | 21.7%    | 40.2%    |

Small models (< 10B) hallucinate massively at T3+ (45--73% wrong). Larger models shift from "wrong" to "refusal" as knowledge limits are reached. The API/Unknown category has high T7 wrong rates (40.2%) because many models attempt answers rather than refusing.

### 7.3 Does the penalty improve correlation with model size?

| Metric | Raw r | Penalized r | Improvement |
|--------|-------|-------------|-------------|
| Overall accuracy | 0.723 | **0.792** | +0.068 |
| T1 accuracy | -0.042 | 0.023 | +0.065 |
| T2 accuracy | 0.281 | 0.404 | +0.124 |
| T3 accuracy | 0.758 | **0.821** | +0.063 |
| T4 accuracy | 0.842 | **0.879** | +0.038 |
| T5 accuracy | 0.787 | **0.846** | +0.060 |
| T6 accuracy | 0.692 | 0.691 | -0.001 |
| T7 accuracy | 0.074 | **0.571** | **+0.497** |

**The penalty improves size correlation at every tier** (or is neutral at T6). The most dramatic improvement is at T7 (+0.497): raw T7 correct rate is uncorrelated with size (r=0.074), but penalized T7 jumps to r=0.571 because larger models refuse rather than hallucinate.

The overall accuracy correlation improves from r=0.723 to r=0.792, confirming the penalty rewards calibrated models that know what they don't know.

### 7.4 Most-penalized models

| Model | Raw Accuracy | Penalized | Penalty |
|-------|-------------|-----------|---------|
| minimax-m1-think | 0.411 | 0.269 | 0.143 |
| gemma-3-27b | 0.461 | 0.348 | 0.113 |
| gemma-2-2b | 0.321 | 0.213 | 0.108 |
| grok-4.20 | 0.679 | 0.572 | 0.108 |
| ministral-8b | 0.448 | 0.340 | 0.108 |
| gemini-2.0-flash | 0.646 | 0.543 | 0.103 |
| gemma-4-31b | 0.469 | 0.366 | 0.103 |

The penalty has the largest impact on models that aggressively attempt all questions (low refusal, high wrong). gemma-family models and grok-4.20 are notably affected.

---

## Summary of Findings

1. **T3 and T4 are the most valuable tiers** for discriminating model quality, with the highest variance, strongest Spearman correlations (0.978, 0.963), and best probe-level discrimination (mean r_pb = 0.54, 0.59).

2. **T1 and T2 are largely redundant** -- both are saturated for models >= 8B. The T1/T2 boundary is the weakest, with 17.5% probe overlap and 17.1% monotonicity violation rate. Consider merging them or using T1+T2 combined as a single "common knowledge" baseline.

3. **T7 does not function as designed.** Only 1/118 models achieves positive penalized accuracy. It measures hallucination avoidance (wrong rate correlates -0.50 with size) rather than knowledge (correct rate correlates 0.07 with size). Drop from scoring or reclassify.

4. **The hallucination penalty is well-justified.** It improves the correlation of overall accuracy with model size from r=0.723 to r=0.792 and improves every tier's correlation (or is neutral). The penalty's strongest effect is on T7 (+0.50) and T2 (+0.12).

5. **Tier boundaries are generally well-ordered** (only 3.5% monotonicity violations), but there are ~50 misplaced probes total: probes in T1 that are T3-hard (e.g., "Year Oxford was founded"), probes in T4 that are T1-easy (e.g., "Who founded Motown"), and ~30 T7 probes that belong in T5--T6.

6. **Domain coverage is uneven across tiers.** T1--T2 have diverse domains (geography, history, culture, science). T5--T7 are dominated by two domains: computer_science (50%) and founding_year (33--36%). This may bias results toward models trained on academic/Wikipedia data.

7. **85% of probes are strong discriminators** (r_pb > 0.2). Noise probes are concentrated in T7 (109/200 have r_pb <= 0). T1 probes have universally positive but modest discrimination (mean r_pb = 0.31), confirming they are too easy to differentiate models.
