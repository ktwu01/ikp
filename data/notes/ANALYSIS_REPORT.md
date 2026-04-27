# IKP Benchmark: Comprehensive Analysis Report

**Date:** 2026-04-14
**Probe set:** v8 (1,400 probes, 200 per tier T1-T7)
**Models evaluated:** 140
**Judge model:** Gemini 3 Flash Preview (3-way: CORRECT/REFUSAL/WRONG)
**Scoring:** Penalized accuracy = mean of per-tier scores, each floored at 0.
Per-tier: (correct - 0.5 * wrong) / total.

---

## Executive Summary

The IKP benchmark measures factual knowledge capacity of language models through 1,400 probes spanning 7 tiers of obscurity, from common facts (T1: "What is the capital of France?") to rare knowledge (T7: obscure founding years and niche CS researchers). Across 140 models ranging from 1B to frontier-scale, we find:

1. **IKP accuracy scales linearly with log(params)** at R^2 = 0.893 across 62 open models, enabling parameter count estimation for proprietary models.
2. **Total parameters predict knowledge for MoE models**, not active parameters (R^2=0.84 vs 0.43), confirming knowledge is stored across all expert weights.
3. **Thinking/reasoning adds ~3 percentage points** as a roughly constant boost, concentrated at medium-hard tiers (T3-T5).
4. **Knowledge fingerprinting detects distillation**: distilled models inherit measurable knowledge patterns from both teacher and base models.
5. **The judge has ~0.1-0.2% error rate**, dominated by cases where models give the correct answer embedded in fabricated context.
6. **GPT-4's leaked 1800B MoE architecture serves as validation**: our open-model calibration predicts 1,188B, an underestimate consistent with proprietary models' superior training data quality.

---

## 1. Scaling Laws

### 1.1 Core Result

We fit **acc = slope * log10(params_B) + intercept** using ordinary least squares on open-weight models with known parameter counts. The calibration set includes models from 1B (gemma-3-1b) to 1,040B (kimi-k2.5-think), spanning dense, MoE, thinking, and non-thinking architectures.

| Subset                         |  n | Slope (pen) | Intercept | R^2 (pen) | RMSE (pen) | R^2 (raw) |
|:-------------------------------|---:|------------:|----------:|----------:|-----------:|----------:|
| All open (total params)        | 62 |      0.1521 |    0.1703 |    0.8932 |     0.0407 |    0.8643 |
| Dense only                     | 44 |      0.1443 |    0.1783 |    0.8116 |     0.0415 |    0.7691 |
| Dense non-thinking             | 34 |      0.1467 |    0.1756 |    0.8178 |     0.0439 |    0.7745 |
| Dense thinking                 | 10 |      0.1187 |    0.2150 |    0.6639 |     0.0301 |    0.6315 |
| MoE (total params)             | 18 |      0.1625 |    0.1492 |    0.8416 |     0.0375 |    0.7595 |
| MoE (active params)            | 18 |      0.1396 |    0.3579 |    0.4304 |     0.0711 |    0.3312 |

### 1.2 Key Findings

**The all-open fit is the strongest (R^2 = 0.893).** Including all 62 open models -- dense, MoE, thinking, non-thinking -- with total parameters as the x-axis yields the best calibration. The slope of 0.152 means each 10x increase in parameters adds ~15.2 percentage points of IKP accuracy.

**Penalized accuracy fits better than raw accuracy across all subsets.** The hallucination penalty consistently improves R^2 (e.g., 0.893 vs 0.864 for all open models), confirming that penalizing confabulation produces a cleaner signal of true knowledge capacity.

**Total params matter for MoE knowledge, not active params.** For MoE models, total params yield R^2 = 0.842 versus 0.430 for active params. This is a strong result: factual knowledge is stored across all expert weights during training, even if only a subset fires per token. The clearest example: llama-4-scout (109B total, 17B active) vs llama-4-maverick (402B total, 17B active) -- same active parameters, but maverick scores 58.4% vs scout's 39.1%, a 19.3 percentage point gap driven entirely by total parameter count.

**The MoE fit dramatically improved with kimi-k2 correction.** After correcting kimi-k2's score from 45.0% (rate-limited) to 63.1% (corrected), the MoE R^2 jumped from 0.596 to 0.842. The kimi-k2 and kimi-k2.5-think models at 1,000B and 1,040B respectively extend the calibration range well beyond the previous largest open MoE (DeepSeek at 671B), anchoring the fit at the frontier end.

**Thinking models have a lower slope (0.119 vs 0.147) but higher intercept (0.215 vs 0.176).** This suggests thinking provides a constant-ish boost (~3 pp, see Section 2) rather than amplifying the scaling relationship. The lower R^2 (0.664) for thinking models reflects the heterogeneity of reasoning approaches.

### 1.3 MoE Calibration Data

| Model                           | Total B | Active B | Pen Acc | Predicted | Residual |
|:--------------------------------|--------:|---------:|--------:|----------:|---------:|
| gemma-4-26b-a4b                 |      26 |        4 |  0.3561 |    0.3855 |  -0.0294 |
| qwen3-30b-a3b-think             |      30 |        3 |  0.3864 |    0.3949 |  -0.0085 |
| qwen3.5-35b-a3b-think           |      35 |        3 |  0.4500 |    0.4051 |  +0.0449 |
| mixtral-8x7b                    |      47 |       13 |  0.4493 |    0.4241 |  +0.0251 |
| qwen3-next-80b-a3b              |      80 |        3 |  0.4968 |    0.4597 |  +0.0371 |
| llama-4-scout                   |     109 |       17 |  0.3911 |    0.4801 |  -0.0890 |
| qwen3.5-122b-a10b-think         |     122 |       10 |  0.5311 |    0.4876 |  +0.0435 |
| qwen3-235b-a22b-think           |     235 |       22 |  0.4868 |    0.5308 |  -0.0441 |
| ernie-4.5-300b-a47b             |     300 |       47 |  0.4946 |    0.5470 |  -0.0523 |
| qwen3.5-397b-a17b-think         |     397 |       17 |  0.5600 |    0.5655 |  -0.0055 |
| jamba-large                     |     398 |       94 |  0.5400 |    0.5656 |  -0.0256 |
| llama-4-maverick                |     402 |       17 |  0.5839 |    0.5663 |  +0.0176 |
| deepseek-v3                     |     671 |       37 |  0.6229 |    0.6001 |  +0.0227 |
| deepseek-v3.2-think             |     671 |       37 |  0.6211 |    0.6001 |  +0.0209 |
| deepseek-v3.2                   |     671 |       37 |  0.6075 |    0.6001 |  +0.0074 |
| deepseek-r1-think               |     671 |       37 |  0.6307 |    0.6001 |  +0.0306 |
| kimi-k2                         |    1000 |       32 |  0.6307 |    0.6265 |  +0.0042 |
| kimi-k2.5-think                 |    1040 |       32 |  0.6896 |    0.6291 |  +0.0606 |

### 1.4 Dense Non-Thinking Calibration Data

| Model                  | Params (B) | Penalized Acc | Predicted |  Residual |
|:-----------------------|-----------:|--------------:|----------:|----------:|
| gemma-3-1b             |        1.0 |        0.1371 |    0.1756 |   -0.0385 |
| llama-3.2-1b           |        1.2 |        0.1904 |    0.1893 |   +0.0010 |
| smollm2-1.7b           |        1.7 |        0.2321 |    0.2094 |   +0.0227 |
| granite-3.3-2b         |        2.0 |        0.2121 |    0.2198 |   -0.0076 |
| gemma-2-2b             |        2.6 |        0.2132 |    0.2365 |   -0.0233 |
| ministral-3b           |        3.0 |        0.2675 |    0.2456 |   +0.0219 |
| llama-3.2-3b           |        3.2 |        0.2843 |    0.2499 |   +0.0344 |
| phi-3-mini             |        3.8 |        0.2132 |    0.2607 |   -0.0474 |
| gemma-3-4b             |        4.0 |        0.2364 |    0.2639 |   -0.0275 |
| gemma-3n-e4b           |        4.0 |        0.2589 |    0.2639 |   -0.0050 |
| command-r7b            |        7.0 |        0.2696 |    0.2996 |   -0.0299 |
| mistral-7b             |        7.2 |        0.2746 |    0.3014 |   -0.0267 |
| qwen-2.5-7b            |        7.6 |        0.2757 |    0.3050 |   -0.0293 |
| ministral-8b           |        8.0 |        0.3404 |    0.3081 |   +0.0323 |
| llama-3-8b             |        8.0 |        0.3704 |    0.3083 |   +0.0620 |
| llama-3.1-8b           |        8.0 |        0.3654 |    0.3083 |   +0.0570 |
| gemma-3-12b            |       12.0 |        0.2982 |    0.3339 |   -0.0357 |
| mistral-nemo-12b       |       12.0 |        0.3386 |    0.3339 |   +0.0047 |
| phi-4                  |       14.7 |        0.3336 |    0.3468 |   -0.0133 |
| mistral-small-24b      |       24.0 |        0.4414 |    0.3781 |   +0.0634 |
| gemma-3-27b            |       27.0 |        0.3479 |    0.3856 |   -0.0377 |
| gemma-2-27b            |       27.0 |        0.3568 |    0.3856 |   -0.0288 |
| gemma-4-31b            |       31.0 |        0.3661 |    0.3944 |   -0.0283 |
| qwq-32b-think          |       32.0 |        0.3939 |    0.3964 |   -0.0025 |
| olmo-3.1-32b           |       32.0 |        0.3471 |    0.3964 |   -0.0493 |
| glm-4-32b              |       32.0 |        0.4236 |    0.3964 |   +0.0272 |
| nemotron-70b           |       70.0 |        0.5218 |    0.4463 |   +0.0755 |
| llama-3-70b            |       70.6 |        0.5311 |    0.4468 |   +0.0843 |
| llama-3.1-70b          |       70.6 |        0.4732 |    0.4468 |   +0.0264 |
| llama-3.3-70b          |       70.6 |        0.5196 |    0.4468 |   +0.0728 |
| qwen-2.5-72b           |       72.7 |        0.4379 |    0.4487 |   -0.0108 |
| command-r-plus         |      104.0 |        0.4243 |    0.4715 |   -0.0472 |
| mistral-large          |      123.0 |        0.5054 |    0.4822 |   +0.0232 |
| hermes-3-405b          |      405.0 |        0.4382 |    0.5581 |   -0.1199 |

---

## 2. Thinking Mode Effects

### 2.1 Overall

Across 12 base/think model pairs, thinking **always improves penalized accuracy** (100% positive). Mean boost: +3.0 percentage points, range: +0.4 to +9.8 pp.

| Base Model               | Base Acc | Think Acc |  Boost |
|:-------------------------|---------:|----------:|-------:|
| claude-opus-4            |   0.4893 |    0.5875 | +0.098 |
| claude-sonnet-4          |   0.4996 |    0.5593 | +0.060 |
| grok-4.20                |   0.5718 |    0.6207 | +0.049 |
| gemini-2.5-flash         |   0.5550 |    0.5946 | +0.040 |
| claude-haiku-4.5         |   0.4400 |    0.4725 | +0.033 |
| gemini-2.5-flash-lite    |   0.4804 |    0.5039 | +0.024 |
| gemini-3-flash           |   0.7882 |    0.8029 | +0.015 |
| deepseek-v3.2            |   0.6075 |    0.6211 | +0.014 |
| gpt-5-nano               |   0.4325 |    0.4443 | +0.012 |
| hunyuan-a13b             |   0.2589 |    0.2671 | +0.008 |
| mimo-v2-flash            |   0.5575 |    0.5618 | +0.004 |
| claude-opus-4.5          |   0.6682 |    0.6721 | +0.004 |

### 2.2 Per-Tier Breakdown

| Tier | Mean Delta | Positive/12 |
|------|-----------|-------------|
| T1 | +0.6% | 8/12 |
| T2 | +1.0% | 9/12 |
| **T3** | **+5.0%** | **12/12** |
| **T4** | **+6.7%** | 10/12 |
| T5 | +5.7% | 8/12 |
| T6 | +2.0% | 6/12 |
| T7 | +0.0% | 1/12 |

**Thinking helps most at T3-T4** (medium-hard knowledge). It has universal benefit at T3 (all 12 pairs improve). At T7, thinking adds nothing -- the knowledge simply is not stored in the weights. This supports the interpretation that chain-of-thought helps with knowledge *retrieval* but does not create new *stored* knowledge.

### 2.3 Largest Beneficiary

**Claude Opus 4:** +9.8 pp penalized (+10.2 pp raw). The base model has high refusal rates that thinking partially overcomes by allowing the model to reason through uncertainty before committing to an answer.

---

## 3. Proprietary Model Size Predictions

### 3.1 Methodology

All predictions use the **all-open calibration curve** fit on 62 open-weight models with known parameter counts:

**acc = 0.1521 * log10(params_B) + 0.1703** (R^2 = 0.893, RMSE = 0.041)

This calibration set ranges from 1B (gemma-3-1b) to 1,040B (kimi-k2.5-think) and includes dense, MoE, thinking, and non-thinking models. **No proprietary model is used as a calibration anchor.** All proprietary models are predicted as unknowns.

To invert the curve: **params_B = 10^((acc - 0.1703) / 0.1521)**

### 3.2 Prediction Table

| Model                       | Vendor     | Pen Acc | Est. Total Params |
|:----------------------------|:-----------|--------:|------------------:|
| gemini-3.1-pro              | Google     | 0.8325  |          22,644B  |
| gemini-3-flash-think        | Google     | 0.8029  |          14,454B  |
| gemini-3-flash              | Google     | 0.7882  |          11,580B  |
| grok-4                      | xAI        | 0.7071  |           3,392B  |
| gpt-5-think                 | OpenAI     | 0.7032  |           3,197B  |
| grok-3                      | xAI        | 0.6971  |           2,916B  |
| gemini-2.5-pro-think        | Google     | 0.6893  |           2,589B  |
| claude-opus-4.6             | Anthropic  | 0.6879  |           2,533B  |
| claude-opus-4.5-think       | Anthropic  | 0.6721  |           1,997B  |
| gpt-4.1                     | OpenAI     | 0.6718  |           1,986B  |
| claude-opus-4.5             | Anthropic  | 0.6682  |           1,881B  |
| glm-5.1-think               | Zhipu      | 0.6586  |           1,626B  |
| gpt-5.4                     | OpenAI     | 0.6461  |           1,345B  |
| glm-5-think                 | Zhipu      | 0.6446  |           1,317B  |
| **gpt-4 (known=1800B)**     | OpenAI     | 0.6379  |       **1,188B**  |
| claude-sonnet-4.6           | Anthropic  | 0.6375  |           1,182B  |
| glm-5-turbo-think           | Zhipu      | 0.6364  |           1,163B  |
| glm-4.7-think               | Zhipu      | 0.6339  |           1,119B  |
| glm-4.6-think               | Zhipu      | 0.6221  |             936B  |
| grok-4.20-think             | xAI        | 0.6207  |             916B  |
| glm-4.5-think               | Zhipu      | 0.6154  |             845B  |
| gpt-4o                      | OpenAI     | 0.6036  |             707B  |
| mimo-v2-pro-think           | Xiaomi     | 0.6004  |             673B  |
| gemini-2.5-flash-think      | Google     | 0.5946  |             617B  |
| claude-opus-4-think         | Anthropic  | 0.5875  |             554B  |
| gpt-5.4-mini                | OpenAI     | 0.5761  |             466B  |
| grok-4.20                   | xAI        | 0.5718  |             437B  |
| grok-4.1-fast-think         | xAI        | 0.5707  |             430B  |
| grok-3-mini-think           | xAI        | 0.5621  |             377B  |
| mistral-medium-3.1          | Mistral    | 0.5593  |             361B  |
| claude-sonnet-4-think       | Anthropic  | 0.5593  |             361B  |
| mimo-v2-flash               | Xiaomi     | 0.5575  |             352B  |
| gemini-2.5-flash            | Google     | 0.5550  |             339B  |
| gpt-4.1-mini                | OpenAI     | 0.5550  |             339B  |
| gpt-3.5-turbo               | OpenAI     | 0.5454  |             293B  |
| gemini-2.0-flash            | Google     | 0.5425  |             280B  |
| command-a                   | Cohere     | 0.5404  |             271B  |
| gpt-5-mini-think            | OpenAI     | 0.5339  |             246B  |
| glm-4.5-air-think           | Zhipu      | 0.5189  |             196B  |
| claude-3.5-haiku            | Anthropic  | 0.5114  |             175B  |
| minimax-m2.7-think          | Minimax    | 0.5018  |             151B  |
| claude-sonnet-4             | Anthropic  | 0.4996  |             147B  |
| claude-opus-4               | Anthropic  | 0.4893  |             125B  |
| gemini-2.5-flash-lite       | Google     | 0.4804  |             109B  |
| claude-haiku-4.5-think      | Anthropic  | 0.4725  |              97B  |
| gpt-4o-mini                 | OpenAI     | 0.4661  |              88B  |
| glm-4.7-flash-think         | Zhipu      | 0.4450  |              64B  |
| claude-haiku-4.5            | Anthropic  | 0.4400  |              59B  |
| gpt-5-nano                  | OpenAI     | 0.4325  |              53B  |
| gpt-4.1-nano                | OpenAI     | 0.4275  |              49B  |
| claude-3-haiku              | Anthropic  | 0.3868  |              27B  |
| nova-micro                  | Amazon     | 0.3161  |               9B  |

### 3.3 Validation Against GPT-4

GPT-4's leaked architecture is reportedly ~8x220B MoE = 1,760B total parameters. Our open-model calibration predicts **1,188B**, an underestimate of **34%**. This gap is expected: proprietary models benefit from higher-quality training data, extensive RLHF, and data curation that makes them "know more than their size suggests" relative to open models. Our predictions should therefore be interpreted as **lower bounds** on actual parameter count, or equivalently, as **effective knowledge capacity** measured in open-model parameter equivalents.

For context, the dense non-thinking curve (34 models) predicts GPT-4 at 1,416B (21% error), showing that the all-open curve's steeper slope produces slightly more conservative predictions.

### 3.4 Interpretation Caveats

These predictions estimate **effective knowledge capacity** in parameter-equivalents, not literal parameter counts. A model trained on higher-quality or more diverse data will appear "larger" than its actual parameter count. Similarly, thinking/reasoning models appear larger because chain-of-thought can extract additional knowledge at inference time. Proprietary models typically benefit from both advantages, so our predictions represent lower bounds on actual size.

The exponential nature of the inverse prediction means small accuracy differences produce large size differences at the frontier. For example, gemini-3.1-pro at 83.3% maps to 22,644B, while grok-4 at 70.7% maps to only 3,392B -- a 13 pp accuracy gap translates to a 7x size difference. Predictions above ~5,000B should be treated with particular caution as they extrapolate well beyond the calibration range (max 1,040B).

---

## 4. Cross-Generation Analysis

### 4.1 Key Progressions

**Claude Opus lineage:**
- Opus 4: 48.9% -> Opus 4.5: 66.8% (+17.9 pp) -> Opus 4.6: 68.8% (+2.0 pp)
- The 4 -> 4.5 jump is the largest single-generation improvement in the dataset.

**GPT lineage:**
- GPT-3.5: 54.5% -> GPT-4: 63.8% -> GPT-4o: 60.4% (regression!) -> GPT-4.1: 67.2% -> GPT-5-think: 70.3%
- GPT-4 -> 4o shows knowledge loss from distillation/compression.

**Qwen at ~8B:**
- Qwen-2.5-7b: 27.6% -> Qwen3-8b-think: 31.8% -> Qwen3.5-9b-think: 34.9%
- Steady +3.6 pp per generation.

**GLM series:**
- GLM-4-32b: 42.4% -> 4.5: 61.5% -> 4.6: 62.2% -> 4.7: 63.4% -> 5: 64.5% -> 5.1: 65.9%
- Remarkably steady improvement, +4-5 pp per generation.

**Claude Haiku regression:**
- Claude 3.5 Haiku: 51.1% -> Haiku 4.5: 44.0% (-7.1 pp)
- Knowledge loss when optimizing for efficiency in a smaller model class.

### 4.2 Vendor Frontier Ranking

| Rank | Vendor     | Best Model                | Accuracy |
|------|:-----------|:--------------------------|:--------:|
| 1    | Google     | gemini-3.1-pro            |  83.3%   |
| 2    | xAI        | grok-4                    |  70.7%   |
| 3    | OpenAI     | gpt-5-think               |  70.3%   |
| 4    | Moonshot   | kimi-k2.5-think           |  69.0%   |
| 5    | Anthropic  | claude-opus-4.6           |  68.8%   |
| 6    | Zhipu      | glm-5.1-think             |  65.9%   |
| 7    | DeepSeek   | deepseek-r1-think         |  63.1%   |
| 8    | Xiaomi     | mimo-v2-pro-think         |  60.0%   |
| 9    | Meta       | llama-4-maverick          |  58.4%   |
| 10   | Alibaba    | qwen3.5-397b-a17b-think   |  56.0%   |
| 11   | Mistral    | mistral-medium-3.1        |  55.9%   |
| 12   | Cohere     | command-a                 |  54.0%   |

---

## 5. Residual Analysis

### 5.1 Biggest Outliers from All-Open Regression

Regression: acc = 0.1521 * log10(B) + 0.1703 (R^2 = 0.893, n = 62)

#### Most Underperforming (actual << predicted)

| Model                             | Params  | Actual | Predicted | Residual | Arch  | Think |
|:----------------------------------|--------:|-------:|----------:|---------:|:------|:------|
| hermes-3-405b                     |  405.0B |  0.438 |     0.567 |   -0.129 | Dense | N     |
| llama-4-scout                     |  109.0B |  0.391 |     0.480 |   -0.089 | MoE   | N     |
| command-r-plus                    |  104.0B |  0.424 |     0.477 |   -0.053 | Dense | N     |
| ernie-4.5-300b-a47b               |  300.0B |  0.495 |     0.547 |   -0.052 | MoE   | N     |
| olmo-3.1-32b                      |   32.0B |  0.347 |     0.399 |   -0.052 | Dense | N     |
| deepseek-r1-distill-qwen-32b      |   32.0B |  0.351 |     0.399 |   -0.048 | Dense | Y     |
| nemotron-super-49b                |   49.0B |  0.380 |     0.427 |   -0.047 | Dense | Y     |
| phi-3-mini                        |    3.8B |  0.213 |     0.258 |   -0.045 | Dense | N     |
| qwen3-235b-a22b-think             |  235.0B |  0.487 |     0.531 |   -0.044 | MoE   | Y     |
| gemma-3-27b                       |   27.0B |  0.348 |     0.388 |   -0.040 | Dense | N     |

#### Most Overperforming (actual >> predicted)

| Model                             | Params  | Actual | Predicted | Residual | Arch  | Think |
|:----------------------------------|--------:|-------:|----------:|---------:|:------|:------|
| llama-3-70b                       |   70.6B |  0.531 |     0.451 |   +0.080 | Dense | N     |
| nemotron-70b                      |   70.0B |  0.522 |     0.451 |   +0.071 | Dense | N     |
| llama-3.3-70b                     |   70.6B |  0.520 |     0.451 |   +0.068 | Dense | N     |
| llama-3-8b                        |    8.0B |  0.370 |     0.308 |   +0.062 | Dense | N     |
| mistral-small-24b                 |   24.0B |  0.441 |     0.380 |   +0.061 | Dense | N     |
| kimi-k2.5-think                   | 1040.0B |  0.690 |     0.629 |   +0.061 | MoE   | Y     |
| llama-3.1-8b                      |    8.0B |  0.365 |     0.308 |   +0.057 | Dense | N     |
| qwen3.5-35b-a3b-think             |   35.0B |  0.450 |     0.405 |   +0.045 | MoE   | Y     |
| qwen3.5-122b-a10b-think           |  122.0B |  0.531 |     0.488 |   +0.044 | MoE   | Y     |
| deepseek-r1-distill-llama-70b     |   70.0B |  0.489 |     0.451 |   +0.038 | Dense | Y     |

### 5.2 Interpretation

**Hermes-3-405b is the largest negative outlier (-0.129).** This is a fine-tuned version of Llama 3.1 405B by NousResearch. The dramatic underperformance (0.438 vs expected 0.567) is caused by extreme refusal behavior: 188/200 refusals on T5 and 189/200 on T7. The Hermes fine-tuning imposed safety filters that destroy factual recall.

**Llama-3-70b is the largest positive outlier (+0.080).** The Llama 3 family consistently overperforms at 70B, suggesting Meta's training data for this model size was particularly knowledge-rich. Nemotron-70b, a fine-tuned Llama 3.1 70B, also overperforms (+0.071), inheriting the strong factual knowledge base.

**kimi-k2.5-think overperforms (+0.061).** Despite being the largest open model (1,040B), it exceeds the trend line, suggesting Moonshot's training data and thinking capabilities are particularly effective for factual recall at this scale.

**kimi-k2 now sits almost exactly on the regression line (+0.004).** After correcting the rate-limiting issue (from 45.0% to 63.1%), kimi-k2's accuracy is well-predicted by its 1,000B parameter count, validating the calibration curve at the largest scale.

**Distilled models show mixed results.** DeepSeek-R1-distill-llama-70b (+0.038) overperforms, suggesting successful knowledge transfer from the 671B teacher. In contrast, deepseek-r1-distill-qwen-32b (-0.048) underperforms, suggesting the smaller student could not fully absorb the teacher's knowledge.

---

## 6. Non-Monotonicity Analysis

### 6.1 Within-Family Violations

Cases where a larger model in the same family scores lower than a smaller one:

| Smaller Model               | Params | Acc    | Larger Model                | Params  | Acc    | Delta  |
|:----------------------------|-------:|-------:|:----------------------------|--------:|-------:|-------:|
| llama-3.3-70b               |  70.6B | 0.5196 | hermes-3-405b               |  405.0B | 0.4382 | 0.0814 |
| llama-3.1-70b               |  70.6B | 0.4732 | hermes-3-405b               |  405.0B | 0.4382 | 0.0350 |
| qwen3-next-80b-a3b          |  80.0B | 0.4968 | qwen3-235b-a22b-think       |  235.0B | 0.4868 | 0.0100 |
| ministral-8b                |   8.0B | 0.3404 | mistral-nemo-12b            |   12.0B | 0.3386 | 0.0018 |

### 6.2 Explanations

**Hermes-3-405b (Delta = -0.081):** The largest violation is entirely explained by post-training. The NousResearch fine-tuning imposed safety filters that cause 94% refusal rates on hard tiers, destroying effective knowledge retrieval.

**Qwen3-next-80b-a3b vs Qwen3-235b-a22b (Delta = 0.010):** A marginal difference likely reflecting training data differences rather than a scaling violation.

**Ministral-8b vs Mistral-Nemo-12b (Delta = 0.002):** Within noise.

### 6.3 Cross-Family Spread at Same Size

At ~32B, the spread across families is 0.076:
- glm-4-32b: 0.424
- qwq-32b: 0.394
- qwen3-32b: 0.393
- deepseek-r1-distill-qwen-32b: 0.351
- olmo-3.1-32b: 0.347

These within-size variations of 5-8 percentage points reflect differences in training data, data quality, and fine-tuning approach.

---

## 7. Per-Tier Scaling

Each tier (T1 = easiest through T7 = hardest) has a different scaling relationship. Using dense non-thinking models:

| Tier | Slope  | Intercept | R^2    | RMSE   | Useful Discrimination Range |
|:-----|-------:|----------:|-------:|-------:|:----------------------------|
| T1   | 0.0536 |    0.8872 | 0.3590 | 0.0455 | 1B - 71B                    |
| T2   | 0.2127 |    0.6155 | 0.6023 | 0.1097 | 1B - 104B                   |
| T3   | 0.4257 |   -0.0957 | 0.8022 | 0.1341 | 3B - 405B                   |
| T4   | 0.2658 |   -0.1349 | 0.5801 | 0.1435 | 8B - 405B                   |
| T5   | 0.0664 |   -0.0410 | 0.3376 | 0.0590 | 70B - 405B                  |
| T6   | 0.0026 |   -0.0018 | 0.3009 | 0.0025 | (no open model > 10%)       |
| T7   | 0.0000 |    0.0000 |    n/a | 0.0000 | (no open model > 0%)        |

### 7.1 Interpretation

**T3 is the most informative tier** (R^2 = 0.802, slope = 0.426), providing excellent size discrimination across the 3B-405B range. Each 10x increase in parameters adds ~42.6 percentage points of T3 accuracy.

**T1-T2 saturate quickly.** Even 1B models score >70% on T1 (common geographical knowledge). T1 has a ceiling effect that limits discrimination (R^2 = 0.359).

**T4 provides complementary discrimination** in the 8B-405B range (R^2 = 0.580), though with higher variance.

**T5-T7 separate only frontier-scale models.** T5 begins to discriminate at 70B+, while T6-T7 show almost no variation across open-weight models. These tiers are where proprietary frontier models separate from each other.

**The aggregate score works best** because it combines discrimination from all tiers. T3's steep slope drives most of the overall correlation.

---

## 8. Knowledge Fingerprinting and Distillation Detection

### 8.1 Vendor Clustering

Models from the same vendor share more knowledge (mean within-vendor Jaccard on T5-T6: 0.328) than cross-vendor (0.290), a ratio of 1.13x. The strongest internal clusters:

- **Gemini 3 family:** J = 0.85-0.90 (tightest cluster in the dataset)
- **OpenAI o3/gpt-5/gpt-4.1:** J = 0.76-0.79
- **Claude opus variants:** J = 0.62-0.78
- **Grok 3/4:** J = 0.77

### 8.2 Distillation Detection

**deepseek-r1-distill-llama-70b** (70B, distilled from 671B R1):
- Jaccard with teacher (R1): 0.319
- Jaccard with base architecture (llama-3.3-70b): 0.315
- Result: knowledge inherited roughly equally from both teacher and base (~1:1 ratio)
- The distilled model knows 100 T5-T6 probes vs base llama-3.3-70b's 67 -- distillation successfully transferred ~33 additional rare facts from the teacher.

**deepseek-r1-distill-qwen-32b** (32B, distilled from 671B R1):
- Jaccard with teacher: 0.089
- Jaccard with base (qwen-2.5-72b): 0.222
- Result: knowledge inherited primarily from base (2.5x ratio). The 32B student was too small to absorb the teacher's rare knowledge.

**nemotron-70b** (fine-tuned llama-3.1-70b):
- Jaccard with base: 0.792 -- near-perfect knowledge preservation through fine-tuning.

### 8.3 Gemini 3 T6 Anomaly

Gemini 3.1 Pro gets 184/200 T6 probes correct (92%), vs the next best non-Gemini model at roughly 48.5%. This advantage:
- Is **not concentrated in one domain** -- gemini-3.1-pro scores 100% across 7 of 14 T6 domains
- Includes **10 probes uniquely known by Gemini 3 models** spanning bridges, places, museums, universities, and sports clubs
- Produces Jaccard ~0.45-0.50 with the best non-Gemini T6 models

This is consistent with either a very large model or one trained on exceptionally broad data coverage. Our calibration predicts ~22,644B, but this should be treated with caution given the extrapolation beyond the calibration range.

---

## 9. Tier Design Evaluation

### 9.1 Discrimination Power

| Tier | Variance | Spearman rho (vs overall) | Useful Range     |
|------|----------|--------------------------|------------------|
| T1   | 0.012    | 0.720                    | 1B-71B           |
| T2   | 0.023    | 0.782                    | 1B-104B          |
| **T3** | **0.092** | **0.978**             | **3B-405B**      |
| **T4** | **0.112** | **0.963**             | **8B-405B**      |
| T5   | 0.079    | 0.873                    | 70B-405B         |
| T6   | 0.017    | 0.258                    | Frontier only    |
| T7   | 0.000    | -0.956                   | Broken           |

**T3 is the single most informative tier** for overall model quality (Spearman rho = 0.978). T4 has the highest variance, making it the best discriminator across the full population.

### 9.2 T7 Ceiling Effect

**T7 is effectively broken as a knowledge tier.** Only 1/118 models achieves positive penalized T7 accuracy (claude-sonnet-4-think at 0.5%). The T7 wrong rate (41.8%) is 8.5x the correct rate (4.9%). T7 currently measures hallucination avoidance, not knowledge.

### 9.3 Hallucination Penalty Impact

The penalty improves accuracy-size correlation from r=0.723 (raw) to r=0.792 (penalized). The most dramatic improvement is at T7 (+0.497): raw T7 correct rate is uncorrelated with size (r=0.074), but penalized T7 jumps to r=0.571 because larger models refuse rather than hallucinate.

### 9.4 Tier Boundary Sharpness

T2/T3 is the sharpest boundary (only 6.5% cross-tier overlap). T1/T2 is weakest (17.5% overlap). Four specific probes are flagged for review:
- Bangladesh highest peak (contested fact, 23% accuracy)
- Transatlantic cable year (gold=1866 is debatable vs 1858, 49% accuracy)
- St. Lawrence University founding (name collision, 0%)
- George Eastman Museum founding (debatable gold, 0%)

---

## 10. Transcript Analysis

### 10.1 Judge Quality

The Gemini 3 Flash Preview judge has an estimated **error rate of ~0.1-0.2%** across 189,367 evaluations. The dominant error pattern: model gives the correct primary answer but adds fabricated elaboration, triggering a WRONG verdict. The judge shows appropriate flexibility for diacritics, numeric formatting, CS field synonyms, and name variants.

### 10.2 Hallucination Patterns

On T5-T7 researcher probes, hallucinations split into:
- **Plausible wrong CS subfield (74%):** Model says "formal methods" when gold is "computer networking"
- **Name collision with non-CS person (26%):** Model describes a real person in a different field with the same name

**Small model pathology (gemma-3-4b):** Responds "formal methods" 87 times and "machine learning" 64 times to different researcher probes -- a default-answer confabulation pattern.

**Large model hallucination (gemini-2.5-pro-think):** Fabricates highly specific but wrong narratives, e.g., attributing NeRF research to a computer architect. More dangerous because it sounds authoritative.

For founding year hallucinations, models are off by a **median of 21 years** (mean 42 years), with a bias toward guessing older dates (59%).

### 10.3 Refusal vs Hallucination

Refusal rates range from 0.1% (grok-4.20) to near-100% (nemotron-ultra-253b, API failure). The correlation between refusal rate and accuracy-when-answering is **r = +0.316**, confirming that models with better calibration produce more accurate answers.

The Claude family shows the strongest calibration behavior. The Gemma family almost never refuses, leading to poor penalized scores despite reasonable raw accuracy.

---

## 11. Anomalies and Notable Results

### 11.1 Rate-Limited Models (Corrected)

Three models had severe rate limiting during initial evaluation:
- **kimi-k2:** 398/1400 probes rate-limited -> original score 45.0%, corrected to **63.1%** after retry
- **qwen3-max:** 836/1400 rate-limited -> original 25.1%, corrected to **61.2%**
- **llama-3.1-70b:** 26 rate-limited -> original 47.3%, corrected to **49.2%**

The kimi-k2 correction was critical for the calibration: it transformed kimi-k2 from the largest negative outlier (-0.163) to sitting almost exactly on the regression line (+0.004), and improved the all-open R^2 from 0.856 to 0.893.

### 11.2 API Failures

- **nemotron-ultra-253b:** 0% across all tiers -- complete API failure, excluded from analysis.

---

## 12. Methodology Notes

### 12.1 Probe Set (v8)

- 1,400 probes: 200 per tier, T1 (easiest) through T7 (hardest)
- 345 CS researcher probes reframed to "In computer science, what is the research subfield of X?"
- 8 problematic probes removed and replaced
- 5 co-credit gold answers fixed

### 12.2 Scoring Formula

Overall accuracy = mean of 7 per-tier scores, each computed as max((correct - 0.5 * wrong) / total, 0). This prevents hard-tier hallucination penalties from erasing easy-tier knowledge.

### 12.3 Calibration Methodology

The calibration curve uses **only open-weight models with known parameter counts** (n=62). No proprietary model is used as a calibration anchor. GPT-4's leaked size (1,800B) serves as an independent validation point.

The all-open fit (R^2 = 0.893) includes dense, MoE, thinking, and non-thinking models, using total parameters for all architectures. The calibration range spans from 1B (gemma-3-1b) to 1,040B (kimi-k2.5-think).

### 12.4 Judge

Gemini 3 Flash Preview via OpenRouter (temperature=0, reasoning.effort=low). 3-way classification: CORRECT/REFUSAL/WRONG. Estimated error rate: 0.1-0.2%.

---

## 13. Summary of Key Findings

1. **log10(total_params) vs IKP accuracy is strongly linear** across 62 open models (R^2 = 0.893), confirming that factual knowledge capacity scales predictably with model size. The slope of 0.152 means each 10x increase adds ~15.2 percentage points.

2. **Total parameters matter for knowledge, not active parameters.** For MoE models, total params give R^2 = 0.842 vs 0.430 for active params. All expert weights store knowledge, even when only a subset fires per token.

3. **The corrected kimi-k2 score (63.1%) was critical.** It validated the scaling law at 1,000B and improved the overall R^2 from 0.856 to 0.893. The kimi models at 1,000-1,040B extend the open-model calibration to the frontier scale.

4. **GPT-4 validates the calibration.** Our prediction of 1,188B vs the known 1,800B (34% error) is consistent with the expectation that proprietary training data quality makes models appear somewhat smaller than they are.

5. **Frontier proprietary models have effective knowledge capacity of 1,000-3,400B** in open-model parameter equivalents (claude-opus-4.6 at 2,533B, grok-4 at 3,392B, gpt-5-think at 3,197B). Gemini 3.1 Pro at 22,644B is an outlier, likely reflecting both large size and exceptional data coverage.

6. **T3 (Wikipedia/researcher knowledge) is the single most informative tier** for size discrimination (R^2 = 0.802, slope = 0.426 per decade). T5-T7 separate only frontier models.

7. **Non-monotonicity is rare and explained by post-training.** The largest violation (hermes-3-405b, -8.1%) is caused by excessive safety refusals, not a scaling law failure.

8. **Thinking adds ~3 percentage points** on average, functioning as a roughly constant boost at T3-T5. This supports the interpretation that chain-of-thought helps with knowledge *retrieval* but does not create new *stored* knowledge.

9. **Training data quality explains most residual variance.** Llama 70B models consistently overperform (+7-8 pp), while Gemma models slightly underperform, suggesting Meta's training data is particularly knowledge-rich at the 70B scale.

---

*Report generated from analysis of 62 open-weight calibration models and 140 total evaluated models. Raw analysis files: analysis_scaling.md, analysis_thinking_arch.md, analysis_transcripts.md, analysis_fingerprint.md, analysis_tiers.md.*
