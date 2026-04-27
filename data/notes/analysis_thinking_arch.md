# Thinking Mode, Architecture, and Generation Effects on IKP Benchmark

This analysis covers 117 models evaluated on the IKP benchmark (1,400 probes across 7 tiers of increasing obscurity, T1--T7). Two accuracy metrics are reported: **penalized** (hallucinated answers receive -0.5 penalty) and **raw** (fraction correct ignoring hallucinations). All figures are on a 0--1 scale unless otherwise noted.

---

## 1. Thinking Mode Effect

### 1.1 Overview

We identified **12 base/think pairs** where both the base model and its thinking-mode variant have results. The thinking variant uses extended reasoning (chain-of-thought) before answering.

| Base Model | Base Pen. | Think Pen. | Delta Pen. | Base Raw | Think Raw | Delta Raw |
|---|---|---|---|---|---|---|
| hunyuan-a13b | 0.2589 | 0.2671 | +0.0082 | 0.3293 | 0.3686 | +0.0393 |
| gpt-5-nano | 0.4325 | 0.4443 | +0.0118 | 0.4486 | 0.4579 | +0.0093 |
| claude-haiku-4.5 | 0.4400 | 0.4725 | +0.0325 | 0.4679 | 0.5043 | +0.0364 |
| gemini-2.5-flash-lite | 0.4804 | 0.5039 | +0.0236 | 0.5464 | 0.5957 | +0.0493 |
| claude-opus-4 | 0.4893 | 0.5875 | **+0.0982** | 0.4971 | 0.5993 | +0.1021 |
| claude-sonnet-4 | 0.4996 | 0.5593 | +0.0596 | 0.5079 | 0.5764 | +0.0686 |
| gemini-2.5-flash | 0.5550 | 0.5946 | +0.0396 | 0.6007 | 0.6893 | +0.0886 |
| mimo-v2-flash | 0.5575 | 0.5618 | +0.0043 | 0.6400 | 0.6364 | -0.0036 |
| grok-4.20 | 0.5718 | 0.6207 | +0.0489 | 0.6793 | 0.6579 | -0.0214 |
| deepseek-v3.2 | 0.6075 | 0.6211 | +0.0136 | 0.6443 | 0.6821 | +0.0379 |
| claude-opus-4.5 | 0.6682 | 0.6721 | +0.0039 | 0.7050 | 0.6936 | -0.0114 |
| gemini-3-flash | 0.7882 | 0.8029 | +0.0146 | 0.8336 | 0.8529 | +0.0193 |

### 1.2 Summary Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Penalized delta | +0.0299 | +0.0191 | +0.0039 | +0.0982 |
| Raw delta | +0.0345 | +0.0371 | -0.0214 | +0.1021 |

- **Positive penalized deltas: 12/12 (100%).** Thinking mode never hurts on the penalized metric.
- **Positive raw deltas: 9/12 (75%).** Three models (mimo-v2-flash, grok-4.20, claude-opus-4.5) show slightly lower raw accuracy with thinking -- the thinking mode reduces hallucinations (improving penalized scores) but does not always increase recall.
- The largest beneficiary is **claude-opus-4** (+9.8 pp penalized, +10.2 pp raw), suggesting thinking helps most when the base model is mid-range.

### 1.3 Per-Tier Thinking Bonus

| Tier | Mean Delta | Median Delta | Min | Max | Positive/Total |
|---|---|---|---|---|---|
| T1 (Easy) | +0.006 | +0.008 | -0.015 | +0.015 | 8/12 |
| T2 | +0.010 | +0.008 | -0.008 | +0.038 | 9/12 |
| T3 | **+0.050** | +0.031 | +0.003 | +0.140 | **12/12** |
| T4 | **+0.067** | +0.049 | -0.020 | +0.195 | 10/12 |
| T5 | +0.057 | +0.019 | -0.035 | +0.240 | 8/12 |
| T6 | +0.020 | +0.003 | -0.063 | +0.103 | 6/12 |
| T7 (Hardest) | +0.000 | +0.000 | +0.000 | +0.005 | 1/12 |

**Key finding:** Thinking mode provides the largest benefit on **T3--T4** (medium-hard tiers), where the absolute improvement is 5--7 percentage points on average. The benefit is **not proportional** to tier difficulty -- it peaks in the middle tiers and drops off for both easy tiers (where the base model already performs well) and the hardest tiers (where the knowledge simply is not present in the model's parameters). At T3, the thinking bonus is positive for all 12 pairs.

### 1.4 Per-Pair Tier Deltas (Think - Base)

| Base Model | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| hunyuan-a13b | +0.007 | -0.008 | +0.057 | +0.000 | +0.000 | +0.000 | +0.000 |
| gpt-5-nano | +0.007 | +0.020 | +0.010 | +0.028 | +0.018 | +0.000 | +0.000 |
| claude-haiku-4.5 | -0.015 | +0.015 | +0.113 | +0.068 | +0.043 | +0.005 | +0.000 |
| gemini-2.5-flash-lite | +0.012 | +0.007 | +0.030 | +0.132 | -0.018 | +0.000 | +0.000 |
| claude-opus-4 | +0.007 | +0.002 | **+0.140** | **+0.195** | **+0.240** | +0.102 | +0.000 |
| claude-sonnet-4 | +0.000 | +0.007 | +0.030 | +0.145 | +0.183 | +0.048 | +0.005 |
| gemini-2.5-flash | +0.012 | +0.008 | +0.068 | +0.027 | +0.225 | -0.062 | +0.000 |
| mimo-v2-flash | +0.015 | +0.037 | +0.032 | -0.020 | -0.035 | +0.000 | +0.000 |
| grok-4.20 | +0.000 | +0.025 | +0.088 | +0.125 | +0.020 | +0.085 | +0.000 |
| deepseek-v3.2 | +0.015 | -0.007 | +0.002 | +0.052 | +0.020 | +0.012 | +0.000 |
| claude-opus-4.5 | +0.010 | +0.015 | +0.020 | +0.045 | -0.033 | -0.030 | +0.000 |
| gemini-3-flash | +0.000 | -0.007 | +0.007 | +0.007 | +0.015 | +0.080 | +0.000 |

Claude Opus 4 shows the largest per-tier gains: +14.0 pp at T3, +19.5 pp at T4, and +24.0 pp at T5. This suggests that thinking mode can unlock latent knowledge that the base model has encoded but fails to retrieve in a single forward pass.

---

## 2. MoE vs Dense at Similar Effective Sizes

### 2.1 Correlation Analysis

For 17 MoE models with known parameter counts (excluding nemotron-ultra-253b with 0 accuracy due to evaluation failure):

| Predictor | Pearson R with accuracy |
|---|---|
| log(total_params) | **0.80** |
| log(active_params) | 0.66 |

**Total parameter count is a better predictor of IKP accuracy than active parameter count for MoE models.** This makes intuitive sense: knowledge storage is distributed across all expert weights, even if only a subset is activated per token.

### 2.2 Same Active Params, Different Total Params

**Llama-4 (both 17B active):**

| Model | Total | Active | Penalized | Raw |
|---|---|---|---|---|
| llama-4-scout | 109B | 17B | 0.3911 | 0.4857 |
| llama-4-maverick | 402B | 17B | **0.5839** | **0.6321** |

With identical active params, Maverick (4x more total params) scores **+19.3 pp** higher. This is the clearest evidence that total parameter count, not active count, determines knowledge capacity in MoE models.

**Qwen3 (both 3B active):**

| Model | Total | Active | Penalized | Raw |
|---|---|---|---|---|
| qwen3-30b-a3b-think | 30B | 3B | 0.3864 | 0.4457 |
| qwen3-next-80b-a3b | 80B | 3B | **0.4968** | **0.5557** |

The 80B total model scores **+11.0 pp** higher despite the same 3B active params.

### 2.3 MoE vs Dense Head-to-Head Comparisons

**llama-4-scout (109B total / 17B active MoE) vs similar-sized dense models:**

| Model | Params | Penalized | T3 | T4 | T5 |
|---|---|---|---|---|---|
| qwen-2.5-7b (dense) | 7.6B | 0.2757 | 0.000 | 0.000 | 0.000 |
| llama-3.1-8b (dense) | 8B | 0.3654 | 0.438 | 0.195 | 0.010 |
| **llama-4-scout (MoE)** | **109B/17B** | **0.3911** | **0.620** | **0.250** | **0.000** |
| mistral-small-24b (dense) | 24B | 0.4414 | 0.738 | 0.383 | 0.000 |

Scout (17B active) performs between 8B and 24B dense models, closer to 24B -- suggesting its effective knowledge capacity is roughly 1.5--2x its active parameter count.

**deepseek-v3 (671B total / 37B active MoE) vs dense models:**

| Model | Params | Penalized | T3 | T4 | T5 |
|---|---|---|---|---|---|
| qwen-2.5-72b (dense) | 72.7B | 0.4379 | 0.730 | 0.323 | 0.028 |
| llama-3.3-70b (dense) | 70.6B | 0.5196 | 0.795 | 0.640 | 0.228 |
| mistral-large (dense) | 123B | 0.5054 | 0.860 | 0.603 | 0.130 |
| **deepseek-v3 (MoE)** | **671B/37B** | **0.6229** | **0.945** | **0.853** | **0.527** |

DeepSeek-V3 (37B active) substantially outperforms all dense models up to 123B, and its performance on T4--T5 suggests effective knowledge equivalent to a dense model well above 123B. This is consistent with MoE total params (671B) being the better predictor.

**gemma-4-26b-a4b (26B total / 4B active MoE) within Gemma family:**

| Model | Params | Penalized | T3 | T4 |
|---|---|---|---|---|
| gemma-3-4b (dense) | 4B | 0.2364 | 0.000 | 0.000 |
| gemma-3-12b (dense) | 12B | 0.2982 | 0.217 | 0.000 |
| **gemma-4-26b-a4b (MoE)** | **26B/4B** | **0.3561** | **0.438** | **0.090** |
| gemma-4-31b (dense) | 31B | 0.3661 | 0.522 | 0.060 |

The 4B-active MoE model performs comparably to the 31B dense model, far above the 4B and 12B dense models. Its knowledge capacity tracks total params (26B) rather than active params (4B).

### 2.4 Summary

For factual knowledge benchmarks, **total parameter count is the primary determinant of MoE knowledge capacity**. Active parameter count provides a floor, but the additional expert parameters store substantially more knowledge. The empirical "effective knowledge size" of an MoE model appears to be somewhere between active and total, but much closer to total -- roughly 50--80% of total parameter count in knowledge-equivalent dense model size.

---

## 3. Cross-Generation Comparisons Within Families

### 3.1 Gemma (2 vs 3 vs 4)

| Model | Params | Arch | Penalized | Raw | T3 | T4 | T5 |
|---|---|---|---|---|---|---|---|
| gemma-2-2b | 2.6B | dense | 0.2132 | 0.3214 | 0.000 | 0.000 | 0.000 |
| gemma-3-1b | 1B | dense | 0.1371 | 0.2064 | 0.000 | 0.000 | 0.000 |
| gemma-3-4b | 4B | dense | 0.2364 | 0.3314 | 0.000 | 0.000 | 0.000 |
| gemma-3n-e4b | 4B | dense | 0.2589 | 0.3421 | 0.015 | 0.000 | 0.000 |
| gemma-3-12b | 12B | dense | 0.2982 | 0.3971 | 0.217 | 0.000 | 0.000 |
| gemma-3-27b | 27B | dense | 0.3479 | 0.4607 | 0.458 | 0.025 | 0.000 |
| gemma-4-26b-a4b | 26B/4B | MoE | 0.3561 | 0.4486 | 0.438 | 0.090 | 0.000 |
| gemma-4-31b | 31B | dense | 0.3661 | 0.4686 | 0.522 | 0.060 | 0.000 |

At the ~27--31B size class, Gemma 4 (0.3661) modestly exceeds Gemma 3 (0.3479), a **+1.8 pp** improvement. Both generations plateau at T4, unable to break into T5. The Gemma family generally shows moderate knowledge for open-weight models of their size.

### 3.2 Llama (3 vs 3.1 vs 3.3 vs 4)

| Model | Params | Penalized | Raw | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|---|
| llama-3-8b | 8.0B | 0.3704 | 0.4371 | 0.477 | 0.203 | 0.000 | 0.000 |
| llama-3.1-8b | 8.0B | 0.3654 | 0.3893 | 0.438 | 0.195 | 0.010 | 0.000 |
| llama-3-70b | 70.6B | **0.5311** | 0.5650 | 0.828 | 0.655 | 0.263 | 0.007 |
| llama-3.1-70b | 70.6B | 0.4732 | 0.4900 | 0.780 | 0.507 | 0.177 | 0.005 |
| llama-3.3-70b | 70.6B | 0.5196 | 0.5564 | 0.795 | 0.640 | 0.228 | 0.013 |
| hermes-3-405b | 405B | 0.4382 | 0.4464 | 0.672 | 0.370 | 0.052 | 0.010 |
| llama-4-scout | 109B/17B MoE | 0.3911 | 0.4857 | 0.620 | 0.250 | 0.000 | 0.000 |
| llama-4-maverick | 402B/17B MoE | 0.5839 | 0.6321 | 0.868 | 0.745 | 0.410 | 0.085 |

**Surprising finding: Llama 3 outperforms 3.1 at both 8B and 70B on penalized accuracy.** At 70B, Llama 3 (0.5311) beats 3.1 (0.4732) by 5.8 pp. The 3.1 models appear to be more conservative (higher refusal rate), which hurts raw accuracy more than the knowledge improvement helps. Llama 3.3-70b (0.5196) recovers most of the ground lost in 3.1 but still trails 3.0.

Hermes-3-405b (Llama 3.1 405B fine-tune) at 0.4382 underperforms Llama 3-70b (0.5311), suggesting the NousResearch fine-tuning may have degraded factual knowledge relative to the base Llama instruct models.

Llama-4-maverick (0.5839) is the family's strongest model, outperforming all 70B dense variants by a significant margin despite having only 17B active params.

### 3.3 Qwen (2.5 vs 3 vs 3.5)

**Dense models at ~7--9B:**

| Model | Gen | Params | Penalized | Raw | T3 | T4 |
|---|---|---|---|---|---|---|
| qwen-2.5-7b | 2.5 | 7.6B | 0.2757 | 0.3407 | 0.000 | 0.000 |
| qwen3-8b-think | 3 | 8B | 0.3175 | 0.4171 | 0.338 | 0.000 |
| qwen3.5-9b-think | 3.5 | 9B | 0.3486 | 0.4357 | 0.520 | 0.000 |

Clear generational improvement: +4.2 pp from 2.5 to 3, +3.1 pp from 3 to 3.5, totaling **+7.3 pp** across three generations at this size class. T3 accuracy goes from 0% (2.5) to 52% (3.5).

**MoE models at 3B active:**

| Model | Gen | Total/Active | Penalized | Raw |
|---|---|---|---|---|
| qwen3-30b-a3b-think | 3 | 30B/3B | 0.3864 | 0.4457 |
| qwen3.5-35b-a3b-think | 3.5 | 35B/3B | 0.4500 | 0.5407 |

**+6.4 pp** improvement from Qwen3 to 3.5 at the same active parameter count.

**Dense 32B class:**

| Model | Gen | Params | Penalized | Raw | T3 | T4 |
|---|---|---|---|---|---|---|
| qwq-32b-think | 2.5 | 32B | 0.3939 | 0.4729 | 0.645 | 0.165 |
| qwen3-32b-think | 3 | 32B | 0.3925 | 0.4650 | 0.807 | 0.000 |

The 32B models are surprisingly close; Qwen3-32b has higher T3 (0.807 vs 0.645) but zero T4 (vs 0.165 for QwQ), possibly reflecting different calibration/refusal thresholds.

### 3.4 GPT (3.5 vs 4 vs 4o vs 4.1 vs 5)

| Model | Penalized | Raw | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| gpt-3.5-turbo | 0.5454 | 0.6050 | 0.915 | 0.640 | 0.285 | 0.000 |
| gpt-4 | 0.6379 | 0.6764 | 0.950 | 0.782 | 0.610 | 0.145 |
| gpt-4o | 0.6036 | 0.6314 | 0.877 | 0.757 | 0.502 | 0.122 |
| gpt-4o-mini | 0.4661 | 0.5450 | 0.818 | 0.490 | 0.003 | 0.000 |
| gpt-4.1-nano | 0.4275 | 0.5271 | 0.708 | 0.338 | 0.000 | 0.000 |
| gpt-4.1-mini | 0.5550 | 0.6529 | 0.902 | 0.730 | 0.275 | 0.000 |
| gpt-4.1 | **0.6718** | **0.7514** | 0.985 | 0.887 | 0.688 | 0.165 |
| gpt-5-nano-think | 0.4443 | 0.4579 | 0.715 | 0.375 | 0.028 | 0.000 |
| gpt-5-mini-think | 0.5339 | 0.5457 | 0.892 | 0.608 | 0.215 | 0.035 |
| gpt-5-think | **0.7032** | 0.7429 | 0.973 | 0.890 | 0.772 | 0.300 |
| gpt-5.4-mini | 0.5761 | 0.6293 | 0.873 | 0.765 | 0.410 | 0.000 |
| gpt-5.4 | 0.6461 | 0.7057 | 0.935 | 0.873 | 0.640 | 0.098 |

**Generation deltas (flagship models):**

| Transition | Penalized Delta |
|---|---|
| GPT-3.5 -> GPT-4 | **+9.3 pp** |
| GPT-4 -> GPT-4o | **-3.4 pp** |
| GPT-4o -> GPT-4.1 | **+6.8 pp** |
| GPT-4.1 -> GPT-5-think | **+3.1 pp** |

GPT-4 outperforms GPT-4o on IKP by 3.4 pp, suggesting that the 4o distillation/optimization may have sacrificed some factual knowledge. GPT-4.1 shows the best non-thinking performance in the family (0.6718). GPT-5-think (0.7032) is the overall GPT family best, particularly strong at T5 (0.772) and T6 (0.300).

The **nano/mini** tiers show that model size matters enormously within a generation: gpt-4.1-nano (0.4275) vs gpt-4.1 (0.6718) is a **24.4 pp** gap.

**Open-weight GPT-OSS models:**

| Model | Params | Penalized | T3 | T4 |
|---|---|---|---|---|
| gpt-oss-20b-think | 20B | 0.3396 | 0.453 | 0.000 |
| gpt-oss-120b-think | 120B | 0.4536 | 0.760 | 0.425 |

The open-weight models substantially lag behind their proprietary counterparts, suggesting different (smaller) training data or intentional knowledge limitation.

### 3.5 Claude (3 Haiku vs 3.5 Haiku vs 4.5 Haiku; Sonnet 4 vs 4.6; Opus 4 vs 4.5 vs 4.6)

**Haiku line:**

| Model | Penalized | Raw | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| claude-3-haiku | 0.3868 | 0.4014 | 0.537 | 0.203 | 0.025 | 0.010 |
| claude-3.5-haiku | **0.5114** | 0.5421 | 0.755 | 0.568 | 0.237 | 0.040 |
| claude-haiku-4.5 | 0.4400 | 0.4679 | 0.688 | 0.395 | 0.028 | 0.000 |
| claude-haiku-4.5-think | 0.4725 | 0.5043 | 0.800 | 0.463 | 0.070 | 0.005 |

Surprisingly, **claude-3.5-haiku (0.5114) outperforms claude-haiku-4.5 (0.4400)** by 7.1 pp, across all middle tiers. Even with thinking enabled, haiku-4.5 (0.4725) trails 3.5-haiku. This suggests the 4.5 Haiku model may have been more aggressively distilled or optimized for efficiency at the cost of factual knowledge.

**Sonnet line:**

| Model | Penalized | Raw | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| claude-sonnet-4 | 0.4996 | 0.5079 | 0.840 | 0.522 | 0.147 | 0.000 |
| claude-sonnet-4-think | 0.5593 | 0.5764 | 0.870 | 0.667 | 0.330 | 0.048 |
| claude-sonnet-4.6 | **0.6375** | **0.6900** | 0.950 | 0.900 | 0.588 | 0.025 |

Sonnet 4.6 is a substantial **+13.8 pp** improvement over Sonnet 4, with dramatic gains at T4 (+37.8 pp) and T5 (+44.1 pp). Sonnet 4.6 without thinking outperforms Sonnet 4 with thinking by 7.8 pp.

**Opus line:**

| Model | Penalized | Raw | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| claude-opus-4 | 0.4893 | 0.4971 | 0.820 | 0.517 | 0.107 | 0.000 |
| claude-opus-4-think | 0.5875 | 0.5993 | 0.960 | 0.713 | 0.347 | 0.102 |
| claude-opus-4.5 | 0.6682 | 0.7050 | 0.963 | 0.863 | 0.680 | 0.198 |
| claude-opus-4.5-think | 0.6721 | 0.6936 | 0.983 | 0.907 | 0.647 | 0.168 |
| claude-opus-4.6 | **0.6879** | **0.7250** | 0.980 | 0.935 | 0.715 | 0.190 |

Opus 4 -> 4.5 is the largest single-generation jump: **+17.9 pp**. Opus 4.5 -> 4.6 adds another **+2.0 pp**. The Opus line shows a clear upward trajectory across all tiers. Notably, Opus 4.6 (no thinking) outperforms Opus 4.5-think by 1.6 pp, suggesting the knowledge improvement in 4.6 outweighs the benefit of thinking on 4.5.

### 3.6 GLM (4 vs 4.5 vs 4.6 vs 4.7 vs 5 vs 5.1)

| Model | Penalized | Raw | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| glm-4-32b | 0.4236 | 0.5021 | 0.698 | 0.330 | 0.000 | 0.000 |
| glm-4.5-air-think | 0.5189 | 0.5757 | 0.853 | 0.593 | 0.210 | 0.000 |
| glm-4.5-think | 0.6154 | 0.6600 | 0.945 | 0.743 | 0.535 | 0.098 |
| glm-4.6-think | 0.6221 | 0.7136 | 0.927 | 0.890 | 0.565 | 0.000 |
| glm-4.7-flash-think | 0.4450 | 0.5321 | 0.828 | 0.318 | 0.000 | 0.000 |
| glm-4.7-think | 0.6339 | 0.7136 | 0.955 | 0.880 | 0.562 | 0.060 |
| glm-5-turbo-think | 0.6364 | 0.7021 | 0.963 | 0.828 | 0.625 | 0.052 |
| glm-5-think | 0.6446 | 0.7271 | 0.963 | 0.875 | 0.665 | 0.043 |
| glm-5.1-think | **0.6586** | 0.7143 | 0.975 | 0.912 | 0.695 | 0.105 |

The GLM family shows steady generational improvement from 4 to 5.1, with the largest jump at 4 -> 4.5 (+19.2 pp, though this also reflects switching from base to think mode). From 4.5 onward, each generation adds +0.7 to +1.4 pp, with consistent T5 improvement (from 0.535 to 0.695). GLM-4.7-flash is notably weaker than GLM-4.7, confirming that "flash" (smaller/faster) variants sacrifice knowledge.

---

## 4. Vendor Comparison at the Frontier

### 4.1 Best Model per Vendor

| Rank | Vendor | Best Model | Penalized | Raw | T5 | T6 |
|---|---|---|---|---|---|---|
| 1 | Google | gemini-3.1-pro | **0.8325** | 0.8450 | 0.963 | **0.902** |
| 2 | xAI | grok-4 | 0.7071 | 0.7500 | 0.835 | 0.268 |
| 3 | OpenAI | gpt-5-think | 0.7032 | 0.7429 | 0.772 | 0.300 |
| 4 | Moonshot | kimi-k2.5-think | 0.6896 | 0.7393 | 0.907 | 0.000 |
| 5 | Anthropic | claude-opus-4.6 | 0.6879 | 0.7250 | 0.715 | 0.190 |
| 6 | Zhipu | glm-5.1-think | 0.6586 | 0.7143 | 0.695 | 0.105 |
| 7 | DeepSeek | deepseek-r1-think | 0.6307 | 0.6886 | 0.608 | 0.077 |
| 8 | Xiaomi | mimo-v2-pro-think | 0.6004 | 0.6200 | 0.425 | 0.072 |
| 9 | Meta | llama-4-maverick | 0.5839 | 0.6321 | 0.410 | 0.085 |
| 10 | Alibaba | qwen3.5-397b-a17b-think | 0.5600 | 0.6443 | 0.325 | 0.000 |
| 11 | Mistral | mistral-medium-3.1 | 0.5593 | 0.6243 | 0.253 | 0.000 |
| 12 | Cohere | command-a | 0.5404 | 0.6036 | 0.212 | 0.000 |
| 13 | AI21 | jamba-large | 0.5400 | 0.5650 | 0.212 | 0.045 |
| 14 | NVIDIA | nemotron-70b | 0.5218 | 0.5393 | 0.198 | 0.005 |
| 15 | MiniMax | minimax-m2.7-think | 0.5018 | 0.5586 | 0.190 | 0.000 |
| 16 | Baidu | ernie-4.5-300b-a47b | 0.4946 | 0.5271 | 0.195 | 0.000 |
| 17 | AllenAI | olmo-3.1-32b | 0.3471 | 0.4143 | 0.000 | 0.000 |
| 18 | Microsoft | phi-4 | 0.3336 | 0.4293 | 0.000 | 0.000 |
| 19 | Amazon | nova-micro | 0.3161 | 0.3993 | 0.000 | 0.000 |
| 20 | Tencent | hunyuan-a13b-think | 0.2671 | 0.3686 | 0.000 | 0.000 |

**Key observations:**
- **Google leads by a wide margin.** Gemini-3.1-pro (0.8325) is 12.5 pp ahead of the second-place grok-4. Its T6 accuracy (0.902) is extraordinary -- 3x higher than any non-Google model.
- **xAI, OpenAI, Moonshot, and Anthropic** form a tight cluster at 0.69--0.71.
- **No model achieves significant T7 accuracy.** Only claude-sonnet-4-think scores T7 > 0 (0.005). T7 probes represent knowledge too obscure for any current model.
- T6 is the primary differentiator at the frontier. Google's dominance is largely driven by its exceptional T6 performance.

### 4.2 Vendor Top-3

| Vendor | #1 | #2 | #3 |
|---|---|---|---|
| Google | gemini-3.1-pro (0.833) | gemini-3-flash-think (0.803) | gemini-3-flash (0.788) |
| OpenAI | gpt-5-think (0.703) | gpt-4.1 (0.672) | gpt-5.4 (0.646) |
| Anthropic | claude-opus-4.6 (0.688) | claude-opus-4.5-think (0.672) | claude-opus-4.5 (0.668) |
| xAI | grok-4 (0.707) | grok-3 (0.697) | grok-4.20-think (0.621) |
| Zhipu | glm-5.1-think (0.659) | glm-5-think (0.645) | glm-5-turbo-think (0.636) |
| DeepSeek | deepseek-r1-think (0.631) | deepseek-v3 (0.623) | deepseek-v3.2-think (0.621) |
| Meta | llama-4-maverick (0.584) | llama-3-70b (0.531) | llama-3.3-70b (0.520) |
| Alibaba | qwen3.5-397b-a17b (0.560) | qwen3.5-122b-a10b (0.531) | qwen3-next-80b-a3b (0.497) |

---

## 5. Training Data Vintage Effect

### 5.1 Generation-over-Generation Summary

| Comparison | Penalized Delta | Raw Delta |
|---|---|---|
| **Positive (newer model knows more)** | | |
| Claude Opus 4 -> 4.5 | **+17.9 pp** | +20.8 pp |
| GLM 4 -> 4.5 (think) | **+19.2 pp** | +15.8 pp |
| Claude Sonnet 4 -> 4.6 | +13.8 pp | +18.2 pp |
| Claude 3 -> 3.5 Haiku | +12.5 pp | +14.1 pp |
| GPT 3.5 -> 4 | +9.3 pp | +7.1 pp |
| Qwen 2.5 -> 3.5 (~8B) | +7.3 pp | +9.5 pp |
| GPT 4o -> 4.1 | +6.8 pp | +12.0 pp |
| Qwen3 -> 3.5 (3B act MoE) | +6.4 pp | +9.5 pp |
| GPT 4.1 -> 5-think | +3.1 pp | -0.9 pp |
| Claude Opus 4.5 -> 4.6 | +2.0 pp | +2.0 pp |
| Gemma 3 -> 4 (~27-31B) | +1.8 pp | +0.8 pp |
| GLM 5 -> 5.1 | +1.4 pp | -1.3 pp |
| GLM 4.7 -> 5 | +1.1 pp | +1.4 pp |
| Grok 3 -> 4 | +1.0 pp | -0.3 pp |
| GLM 4.5 -> 4.6 | +0.7 pp | +5.4 pp |
| **Negative (newer model knows less)** | | |
| Claude 3.5 -> 4.5 Haiku | **-7.1 pp** | -7.4 pp |
| GPT 4 -> 4o | -3.4 pp | -4.5 pp |
| DeepSeek V3 -> V3.2 | -1.5 pp | -3.6 pp |
| Llama 3 -> 3.3 (70B) | -1.1 pp | -0.9 pp |
| Llama 3 -> 3.1 (8B) | -0.5 pp | -4.8 pp |

### 5.2 Analysis

**The general trend is positive:** 15 out of 20 within-family transitions show improvement in penalized accuracy. The median improvement is **+2.5 pp** per generation, with a mean of **+4.3 pp**.

However, five transitions show **knowledge regression** where newer models know less:

1. **Claude 3.5-haiku -> 4.5-haiku (-7.1 pp):** The largest regression. Likely reflects aggressive distillation or model compression in the "haiku" tier.

2. **GPT-4 -> GPT-4o (-3.4 pp):** GPT-4o was optimized for speed and multimodality; the distillation/efficiency optimization cost factual knowledge.

3. **DeepSeek V3 -> V3.2 (-1.5 pp):** A small regression, possibly due to alignment tuning or different emphasis.

4. **Llama 3 -> 3.1/3.3 (-0.5 to -1.1 pp):** Minor regressions, likely within measurement noise. The 3.1 models show increased refusal rates, which hurts raw accuracy substantially (-4.8 pp at 8B).

**Large jumps tend to occur during major model redesigns** (Claude Opus 4 -> 4.5, GLM 4 -> 4.5, GPT 3.5 -> 4), while minor version updates (GLM 4.6 -> 4.7, Claude 4.5 -> 4.6) produce smaller but still positive gains.

**Diminishing returns at the frontier:** The most advanced model families (GLM 5.x, Claude Opus 4.5+, GPT 5.x) show smaller per-generation improvements (+1--3 pp), suggesting approaching a ceiling around 0.7--0.85 on the current IKP benchmark (with T7 effectively uncrackable).

### 5.3 The Efficiency-Knowledge Tradeoff

Several transitions reveal a pattern where vendors sacrifice factual knowledge for efficiency:
- **GPT-4 -> GPT-4o:** Smaller, faster, but -3.4 pp knowledge
- **Claude 3.5-haiku -> 4.5-haiku:** Likely smaller model, -7.1 pp knowledge
- **GLM-4.7-flash vs GLM-4.7:** Flash variant scores 0.4450 vs full model's 0.6339 (-18.9 pp)
- **GPT-4.1-nano (0.4275) vs GPT-4.1 (0.6718):** -24.4 pp within same generation

This demonstrates that within a model generation, the size/efficiency tier is a far stronger determinant of factual knowledge than the generation number. A previous-generation large model often outperforms a current-generation small model.
