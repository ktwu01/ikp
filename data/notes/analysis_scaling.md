# IKP Scaling Law Analysis

Statistical analysis of IKP (Implicit Knowledge Probing) benchmark results across 140 models.
Penalized accuracy uses a -0.5 hallucination penalty for wrong answers; raw accuracy counts only correct answers.
Calibration uses **only open-weight models with known parameter counts**. No proprietary model is used as a calibration anchor.

---

## 1. Scaling Law Fits: log10(total_params) vs Accuracy

We fit **acc = slope * log10(params_B) + intercept** for different model subsets.

### 1.1 Summary Table

| Subset                         |  n | Slope (pen) | Intercept | R^2 (pen) | RMSE (pen) | R^2 (raw) |
|:-------------------------------|---:|------------:|----------:|----------:|-----------:|----------:|
| All open (total params)        | 62 |      0.1521 |    0.1703 |    0.8932 |     0.0407 |    0.8643 |
| Dense only                     | 44 |      0.1443 |    0.1783 |    0.8116 |     0.0415 |    0.7691 |
| Dense non-thinking             | 34 |      0.1467 |    0.1756 |    0.8178 |     0.0439 |    0.7745 |
| Dense thinking                 | 10 |      0.1187 |    0.2150 |    0.6639 |     0.0301 |    0.6315 |
| MoE (total params)             | 18 |      0.1625 |    0.1492 |    0.8416 |     0.0375 |    0.7595 |
| MoE (active params)            | 18 |      0.1396 |    0.3579 |    0.4304 |     0.0711 |    0.3312 |

### 1.2 Key Findings

**Total params is far better than active params for MoE models.** Using total parameters for MoE models yields R^2 = 0.842, while active parameters gives only R^2 = 0.430. This is a striking result: for factual knowledge capacity, all parameters in the model contribute, not just those activated at inference time. This makes physical sense -- knowledge is stored across all expert weights during training, even if only a subset fires per token.

**The kimi-k2 correction transformed the MoE fit.** Before correcting kimi-k2's rate-limited score (45.0% -> 63.1%), the MoE R^2 was only 0.596. The corrected score at 1,000B sits almost exactly on the regression line (residual +0.004), validating the scaling law at the largest open-model scale.

**Penalized accuracy fits better than raw accuracy across all subsets.** The hallucination penalty consistently improves R^2 (e.g., 0.893 vs 0.864 for all open models), confirming that penalizing confabulation produces a cleaner signal of true knowledge capacity.

**The all-open fit gives the cleanest calibration curve** (R^2 = 0.893), outperforming dense-only (0.812) and dense-non-thinking (0.818) subsets. Including MoE models and thinking models in the calibration set improves the fit because it extends the calibration range to 1,040B via kimi-k2.5-think. The slope is 0.1521, meaning each 10x increase in parameters adds ~15.2 percentage points of IKP accuracy.

**Thinking models have a lower slope (0.119 vs 0.147)** but a higher intercept (0.215 vs 0.176). This suggests thinking provides a constant-ish boost (~3-4 points, see Section 6) rather than amplifying the scaling relationship. The lower R^2 (0.664) for thinking models reflects the heterogeneity of reasoning approaches.

### 1.3 Dense Non-Thinking Calibration Data

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

## 2. Parameter Prediction for Proprietary Models

### 2.1 Calibration Curve

We use the all-open fit as the primary calibration curve:

**acc = 0.1521 * log10(B) + 0.1703** (R^2 = 0.893, n = 62)

This curve is built entirely from open-weight models with known parameter counts, ranging from 1B (gemma-3-1b) to 1,040B (kimi-k2.5-think). No proprietary model (including GPT-4) is used as a calibration anchor. GPT-4's leaked size serves only as an independent validation point.

### 2.2 Prediction Table

| Model                       | Vendor     | Pen Acc  | Predicted Size |
|:----------------------------|:-----------|:--------:|:--------------:|
| gemini-3.1-pro              | Google     |  0.8325  |       22,644B  |
| gemini-3-flash-think        | Google     |  0.8029  |       14,454B  |
| gemini-3-flash              | Google     |  0.7882  |       11,580B  |
| grok-4                      | xAI        |  0.7071  |        3,392B  |
| gpt-5-think                 | OpenAI     |  0.7032  |        3,197B  |
| grok-3                      | xAI        |  0.6971  |        2,916B  |
| gemini-2.5-pro-think        | Google     |  0.6893  |        2,589B  |
| claude-opus-4.6             | Anthropic  |  0.6879  |        2,533B  |
| claude-opus-4.5-think       | Anthropic  |  0.6721  |        1,997B  |
| gpt-4.1                     | OpenAI     |  0.6718  |        1,986B  |
| claude-opus-4.5             | Anthropic  |  0.6682  |        1,881B  |
| glm-5.1-think               | Zhipu      |  0.6586  |        1,626B  |
| gpt-5.4                     | OpenAI     |  0.6461  |        1,345B  |
| glm-5-think                 | Zhipu      |  0.6446  |        1,317B  |
| gpt-4 (known=1800B)         | OpenAI     |  0.6379  |        1,188B  |
| claude-sonnet-4.6           | Anthropic  |  0.6375  |        1,182B  |
| glm-5-turbo-think           | Zhipu      |  0.6364  |        1,163B  |
| glm-4.7-think               | Zhipu      |  0.6339  |        1,119B  |
| glm-4.6-think               | Zhipu      |  0.6221  |          936B  |
| grok-4.20-think             | xAI        |  0.6207  |          916B  |
| glm-4.5-think               | Zhipu      |  0.6154  |          845B  |
| gpt-4o                      | OpenAI     |  0.6036  |          707B  |
| mimo-v2-pro-think           | Xiaomi     |  0.6004  |          673B  |
| gemini-2.5-flash-think      | Google     |  0.5946  |          617B  |
| claude-opus-4-think         | Anthropic  |  0.5875  |          554B  |
| gpt-5.4-mini                | OpenAI     |  0.5761  |          466B  |
| grok-4.20                   | xAI        |  0.5718  |          437B  |
| grok-4.1-fast-think         | xAI        |  0.5707  |          430B  |
| grok-3-mini-think           | xAI        |  0.5621  |          377B  |
| mistral-medium-3.1          | Mistral    |  0.5593  |          361B  |
| claude-sonnet-4-think       | Anthropic  |  0.5593  |          361B  |
| mimo-v2-flash               | Xiaomi     |  0.5575  |          352B  |
| gemini-2.5-flash            | Google     |  0.5550  |          339B  |
| gpt-4.1-mini                | OpenAI     |  0.5550  |          339B  |
| gpt-3.5-turbo               | OpenAI     |  0.5454  |          293B  |
| gemini-2.0-flash            | Google     |  0.5425  |          280B  |
| command-a                   | Cohere     |  0.5404  |          271B  |
| gpt-5-mini-think            | OpenAI     |  0.5339  |          246B  |
| glm-4.5-air-think           | Zhipu      |  0.5189  |          196B  |
| claude-3.5-haiku            | Anthropic  |  0.5114  |          175B  |
| minimax-m2.7-think          | Minimax    |  0.5018  |          151B  |
| claude-sonnet-4             | Anthropic  |  0.4996  |          147B  |
| claude-opus-4               | Anthropic  |  0.4893  |          125B  |
| gemini-2.5-flash-lite       | Google     |  0.4804  |          109B  |
| claude-haiku-4.5-think      | Anthropic  |  0.4725  |           97B  |
| gpt-4o-mini                 | OpenAI     |  0.4661  |           88B  |
| glm-4.7-flash-think         | Zhipu      |  0.4450  |           64B  |
| claude-haiku-4.5            | Anthropic  |  0.4400  |           59B  |
| gpt-5-nano                  | OpenAI     |  0.4325  |           53B  |
| gpt-4.1-nano                | OpenAI     |  0.4275  |           49B  |
| claude-3-haiku              | Anthropic  |  0.3868  |           27B  |
| nova-micro                  | Amazon     |  0.3161  |            9B  |

### 2.3 Validation Against GPT-4

**GPT-4:** Leaked architecture is reportedly ~8x220B MoE = 1,760B total. Our open-model prediction is **1,188B**, an underestimate of **34%**. This gap is expected and informative: proprietary models benefit from higher-quality training data, extensive RLHF, and data curation that makes them "know more than their size suggests" relative to open models. The systematic underestimate suggests our predictions are **lower bounds** on actual parameter count.

**GPT-4o:** The Epoch AI estimate puts GPT-4o at ~200B total. Our prediction of 707B suggests GPT-4o was trained on significantly better data than the open-model average, making it appear ~3.5x larger than it actually is.

### 2.4 Comparison with Previous Calibration

The previous report used an "anchored" calibration that included GPT-4 (1800B) and Gemini 3.1 Pro (assumed 5000B) as calibration points. The new approach uses only open models:

| Model                  | Pen Acc | Old Pred (anchored) | New Pred (open-only) | Change |
|:-----------------------|--------:|:-------------------:|:--------------------:|:------:|
| gemini-3.1-pro         | 0.8325  |          17,535B    |          22,644B     |  +29%  |
| grok-4                 | 0.7071  |           2,781B    |           3,392B     |  +22%  |
| gpt-5-think            | 0.7032  |           2,625B    |           3,197B     |  +22%  |
| claude-opus-4.6        | 0.6879  |           2,095B    |           2,533B     |  +21%  |
| gpt-4.1                | 0.6718  |           1,654B    |           1,986B     |  +20%  |
| gpt-4 (known=1800B)    | 0.6379  |           1,005B    |           1,188B     |  +18%  |
| gpt-4o                 | 0.6036  |             607B    |             707B     |  +17%  |
| claude-3-haiku          | 0.3868  |              25B    |              27B     |   +8%  |

The new open-only calibration produces **systematically higher predictions** (17-29%) because it does not anchor on proprietary models that overperform for their size. The GPT-4 error (34% vs 44% in old anchored) is directionally better, as the old anchored curve was artificially dragged toward GPT-4's data point.

### 2.5 Interpretation Caveats

These predictions estimate **effective knowledge capacity** in parameter-equivalents, not literal parameter counts. A model trained on higher-quality or more diverse data will appear "larger" than its actual parameter count. Similarly, thinking/reasoning models appear larger because chain-of-thought can extract additional knowledge at inference time. Proprietary models typically benefit from both advantages (better data, RLHF, etc.), so our predictions should be interpreted as lower bounds on actual parameter count.

The exponential nature of the inverse prediction means small accuracy differences produce large size differences at the frontier. Predictions above ~5,000B extrapolate well beyond the calibration range (max 1,040B) and should be treated with particular caution.

---

## 3. Non-Monotonicity Analysis

### 3.1 Within-Family Violations

These are cases where a larger model in the same family scores *lower* than a smaller one:

| Smaller Model               | Params | Acc    | Larger Model                | Params  | Acc    | Delta  |
|:----------------------------|-------:|-------:|:----------------------------|--------:|-------:|-------:|
| llama-3.3-70b               |  70.6B | 0.5196 | hermes-3-405b               |  405.0B | 0.4382 | 0.0814 |
| llama-3.1-70b               |  70.6B | 0.4732 | hermes-3-405b               |  405.0B | 0.4382 | 0.0350 |
| qwen3-next-80b-a3b          |  80.0B | 0.4968 | qwen3-235b-a22b-think       |  235.0B | 0.4868 | 0.0100 |
| ministral-8b                |   8.0B | 0.3404 | mistral-nemo-12b            |   12.0B | 0.3386 | 0.0018 |

### 3.2 Explanations

**Hermes-3-405b (Delta = -0.081):** This is the largest violation. Hermes-3-405b is a fine-tuned version of Llama 3.1 405B by NousResearch. The dramatic underperformance (0.438 vs 0.520-0.531 for 70B Llamas) is almost certainly due to the model's extreme refusal behavior: it has by far the highest refusal rates in the dataset. The Hermes fine-tuning imposed safety filters that cause it to decline to answer many factual questions, destroying its effective knowledge retrieval.

**Qwen3-next-80b-a3b vs Qwen3-235b-a22b (Delta = 0.010):** A small MoE (80B total, 3B active) slightly edges out a much larger MoE (235B total, 22B active). Both are thinking models. The difference is marginal and may reflect dataset/training differences rather than a true scaling violation.

**Ministral-8b vs Mistral-Nemo-12b (Delta = 0.002):** Within noise.

### 3.3 Cross-Family Spread at Same Size

At ~32B, the spread across families is 0.076:
- glm-4-32b: 0.424
- qwq-32b: 0.394
- qwen3-32b: 0.393
- deepseek-r1-distill-qwen-32b: 0.351
- olmo-3.1-32b: 0.347

At ~70B, the spread is 0.058:
- llama-3-70b: 0.531
- llama-3.3-70b: 0.520
- llama-3.1-70b: 0.473

These within-size variations of 5-8 percentage points reflect differences in training data, data quality, and fine-tuning approach. The spread is comparable to the ~3% boost from thinking (Section 6), suggesting training data diversity matters as much as or more than inference-time reasoning.

---

## 4. Per-Tier Scaling

Each tier (T1=most common knowledge through T7=rarest) has a different scaling relationship. Using dense non-thinking models:

| Tier | Slope  | Intercept | R^2    | RMSE   | Useful Discrimination Range |
|:-----|-------:|----------:|-------:|-------:|:----------------------------|
| T1   | 0.0536 |    0.8872 | 0.3590 | 0.0455 | 1B - 71B                    |
| T2   | 0.2127 |    0.6155 | 0.6023 | 0.1097 | 1B - 104B                   |
| T3   | 0.4257 |   -0.0957 | 0.8022 | 0.1341 | 3B - 405B                   |
| T4   | 0.2658 |   -0.1349 | 0.5801 | 0.1435 | 8B - 405B                   |
| T5   | 0.0664 |   -0.0410 | 0.3376 | 0.0590 | 70B - 405B                  |
| T6   | 0.0026 |   -0.0018 | 0.3009 | 0.0025 | (no open model > 10%)       |
| T7   | 0.0000 |    0.0000 |    n/a | 0.0000 | (no open model > 0%)        |

### 4.1 Interpretation

**T3 is the most informative tier** (R^2 = 0.802, slope = 0.426), providing excellent size discrimination across nearly the full range (3B-405B). T3 probes (Wikipedia/researcher knowledge) show the steepest scaling: each 10x increase in parameters adds ~42.6 percentage points.

**T1-T2 saturate quickly.** Even 1B models score >70% on T1 (common geographical knowledge). T1 has a ceiling effect that limits its discrimination power (low R^2 = 0.359). T2 is somewhat better, useful for distinguishing models in the 1B-100B range.

**T4 provides complementary discrimination** in the 8B-405B range, though with higher variance (R^2 = 0.580). T4 probes cover less-well-known researchers and entities.

**T5-T7 are only useful for frontier-scale models.** T5 begins to discriminate at 70B+, while T6-T7 show almost no variation across open-weight models. These tiers are where proprietary frontier models separate from each other (see Section 2: the top models reach T5 scores of 70-97%).

**The aggregate score (T1-T7 average) works best** because it combines the discrimination power of all tiers. The T3-heavy weighting in the aggregate score explains why it correlates so well with size.

---

## 5. Residual Analysis

### 5.1 Biggest Outliers from All-Open Regression

Regression: acc = 0.1521 * log10(B) + 0.1703 (R^2 = 0.893)

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

**Hermes-3-405b (-0.129)** is the largest outlier due to excessive refusal behavior imposed by NousResearch fine-tuning. It has 188/200 refusals on T5 and 189/200 on T7.

**kimi-k2 now sits on the line (+0.004).** After correcting the rate-limiting issue (from 45.0% to 63.1%), kimi-k2's accuracy is well-predicted by its 1,000B parameter count. In the previous analysis, kimi-k2 was the largest negative outlier (-0.157); this correction was the single most impactful data fix in the analysis.

**Llama-3-70b (+0.080)** is the largest positive outlier. The Llama 3 family consistently overperforms at 70B, suggesting Meta's training data for this model size was particularly high-quality for factual knowledge. Nemotron-70b, a fine-tuned Llama 3.1 70B, also overperforms (+0.071), inheriting the strong factual knowledge base.

**Mistral-small-24b (+0.061)** significantly outperforms its size class, likely due to Mistral's focus on data quality and knowledge-dense training mixtures.

**kimi-k2.5-think (+0.061)** overperforms at 1,040B, the largest in our calibration set. This positive residual is partly explained by thinking mode (see Section 6: thinking adds ~3 pp on average, and the all-open fit includes both thinking and non-thinking models).

**Distilled models show mixed results.** DeepSeek-R1-distill-llama-70b (+0.038) overperforms, suggesting successful knowledge distillation from the much larger R1 teacher. In contrast, deepseek-r1-distill-qwen-32b (-0.048) underperforms, suggesting the smaller student model could not fully absorb the teacher's knowledge.

### 5.3 Dense Non-Thinking Residual Pattern

When restricting to dense non-thinking models only:

- **Llama family models at 70B consistently overperform** (~+0.07), suggesting training data advantage
- **Gemma models tend to slightly underperform** (gemma-3-12b: -0.036, gemma-3-27b: -0.038, gemma-4-31b: -0.028)
- **Hermes-3-405b is a dramatic outlier** (-0.120), the only 405B dense model, likely degraded by fine-tuning

---

## 6. Thinking Model Boost Analysis

Across 12 base/thinking model pairs with both variants evaluated:

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

**Average thinking boost: +0.030** (3.0 percentage points)

The boost is always positive but varies significantly (0.4% to 9.8%). Larger boosts occur for models that are more "cautious" in non-thinking mode (e.g., Claude Opus 4 has high refusal rates that thinking partially overcomes). Models already near their ceiling show minimal boost (e.g., Claude Opus 4.5, Mimo v2 flash).

---

## 7. Summary of Key Findings

1. **log10(total_params) vs IKP accuracy is strongly linear** across 62 open models (R^2 = 0.893), confirming that factual knowledge capacity scales predictably with model size.

2. **Total parameters matter for knowledge, not active parameters.** For MoE models, total params give R^2 = 0.842 vs 0.430 for active params. All expert weights store knowledge, even when only a subset fires per token.

3. **The all-open calibration** (R^2 = 0.893, slope = 0.152) provides the cleanest prediction baseline. Each 10x increase in parameters adds ~15.2 percentage points of penalized IKP accuracy.

4. **The calibration extends to 1,040B** via kimi-k2.5-think, significantly reducing extrapolation for frontier model predictions. The corrected kimi-k2 (63.1%) sits precisely on the trend line at 1,000B.

5. **GPT-4 validates the curve:** predicted 1,188B vs known 1,800B (34% underestimate), consistent with proprietary training data advantage.

6. **Frontier proprietary models score equivalently to 1,000-3,400B parameter open models.** Gemini 3.1 Pro at 22,644B is a dramatic outlier requiring extrapolation well beyond the calibration range.

7. **T3 (Wikipedia/researcher knowledge) is the single most informative tier** for size discrimination (R^2 = 0.802, slope = 0.426 per decade). T5-T7 separate only frontier models.

8. **Non-monotonicity is rare and explained by post-training.** The largest violation (Hermes-3-405b, -8.1%) is caused by excessive safety refusals destroying factual recall, not by a failure of the scaling law itself.

9. **Thinking adds ~3 percentage points** on average, functioning as a roughly constant boost rather than changing the scaling slope. This supports the interpretation that chain-of-thought reasoning helps with knowledge *retrieval* but does not create new *stored* knowledge.

10. **Training data quality explains most residual variance.** Llama 70B models consistently overperform (+7-8 pts), while Gemma models slightly underperform, suggesting Meta's training data is particularly knowledge-rich at the 70B scale point.
