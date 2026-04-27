# IKP Knowledge Fingerprint Analysis

Analysis of knowledge overlap patterns across 135 models using T5-T6 (rare knowledge) probes.
T5 probes: 200, T6 probes: 200, Total T5+T6: 400

## 0. Model T5-T6 Accuracy Overview

| Rank | Model | Vendor | T5 correct (/200) | T6 correct (/200) | T5+T6 total (/400) |
|------|-------|--------|-------------------|-------------------|---------------------|
| 1 | gemini-3.1-pro | google | 193 | 184 | 377 |
| 2 | gemini-3-flash-think | google | 196 | 164 | 360 |
| 3 | gemini-3-flash | google | 194 | 153 | 347 |
| 4 | grok-3 | xai | 170 | 90 | 260 |
| 5 | gemini-2.5-pro-think | google | 163 | 97 | 260 |
| 6 | grok-4 | xai | 173 | 79 | 252 |
| 7 | o3 | openai | 159 | 92 | 251 |
| 8 | gpt-4.1 | openai | 158 | 88 | 246 |
| 9 | gpt-5-think | openai | 158 | 86 | 244 |
| 10 | kimi-k2.5-think | moonshot | 186 | 44 | 230 |
| 11 | glm-5-think | zhipu | 154 | 65 | 219 |
| 12 | glm-5.1-think | zhipu | 156 | 61 | 217 |
| 13 | claude-opus-4.6 | anthropic | 149 | 64 | 213 |
| 14 | glm-4.7-think | zhipu | 139 | 68 | 207 |
| 15 | glm-5-turbo-think | zhipu | 143 | 60 | 203 |
| 16 | glm-4.6-think | zhipu | 141 | 60 | 201 |
| 17 | gpt-5.4 | openai | 140 | 61 | 201 |
| 18 | claude-opus-4.5 | anthropic | 144 | 55 | 199 |
| 19 | grok-4.20 | xai | 136 | 60 | 196 |
| 20 | gemini-2.5-flash-think | google | 130 | 61 | 191 |
| 21 | seed-2.0-lite-think | bytedance | 130 | 56 | 186 |
| 22 | deepseek-r1-think | deepseek | 136 | 49 | 185 |
| 23 | gpt-4 | openai | 130 | 53 | 183 |
| 24 | kimi-k2 | moonshot | 139 | 44 | 183 |
| 25 | claude-opus-4.5-think | anthropic | 135 | 43 | 178 |
| 26 | claude-sonnet-4.6 | anthropic | 133 | 43 | 176 |
| 27 | deepseek-v3.2-think | deepseek | 129 | 47 | 176 |
| 28 | qwen3-max | alibaba | 127 | 40 | 167 |
| 29 | deepseek-v3 | deepseek | 121 | 46 | 167 |
| 30 | glm-4.5-think | zhipu | 120 | 42 | 162 |
| 31 | grok-4.20-think | xai | 120 | 35 | 155 |
| 32 | step-3.5-flash-think | stepfun | 111 | 37 | 148 |
| 33 | gpt-4.1-mini | openai | 103 | 42 | 145 |
| 34 | gpt-4o | openai | 103 | 42 | 145 |
| 35 | deepseek-v3.2 | deepseek | 116 | 28 | 144 |
| 36 | mimo-v2-flash | xiaomi | 110 | 33 | 143 |
| 37 | qwen3.5-plus-think | alibaba | 107 | 35 | 142 |
| 38 | nova-premier | amazon | 93 | 48 | 141 |
| 39 | qwen3.5-397b-a17b-think | alibaba | 100 | 37 | 137 |
| 40 | grok-4.1-fast-think | xai | 104 | 31 | 135 |

## 1. Knowledge Overlap Matrix (T5-T6 Jaccard Similarity)

Models included: overall accuracy > 40% (from evaluation_summary).

### Full Jaccard Matrix (T5+T6 correct sets)

Values rounded to 2 decimal places. Models sorted by T5+T6 count descending.

Showing top 30 models (of 97 qualifying). Full data in supplementary materials.
```
Model                     | gemini-3.. | gemini-3.. | gemini-3.. |     grok-3 | gemini-2.. |     grok-4 |         o3 |    gpt-4.1 | gpt-5-th.. | kimi-k2... | glm-5-th.. | glm-5.1-.. | claude-o.. | glm-4.7-.. | glm-5-tu.. | glm-4.6-.. |    gpt-5.4 | claude-o.. |  grok-4.20 | gemini-2.. | seed-2.0.. | deepseek.. |      gpt-4 |    kimi-k2 | claude-o.. | claude-s.. | deepseek.. |  qwen3-max | deepseek.. | glm-4.5-..
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
gemini-3.1-pro            |    1.00    |    0.88    |    0.85    |    0.66    |    0.65    |    0.64    |    0.64    |    0.63    |    0.64    |    0.57    |    0.56    |    0.56    |    0.55    |    0.52    |    0.51    |    0.49    |    0.53    |    0.51    |    0.48    |    0.49    |    0.47    |    0.48    |    0.48    |    0.46    |    0.46    |    0.46    |    0.45    |    0.44    |    0.44    |    0.41   
gemini-3-flash-think      |    0.88    |    1.00    |    0.90    |    0.67    |    0.65    |    0.67    |    0.63    |    0.62    |    0.63    |    0.60    |    0.56    |    0.58    |    0.56    |    0.53    |    0.53    |    0.52    |    0.54    |    0.52    |    0.53    |    0.51    |    0.49    |    0.50    |    0.48    |    0.47    |    0.48    |    0.48    |    0.47    |    0.44    |    0.44    |    0.43   
gemini-3-flash            |    0.85    |    0.90    |    1.00    |    0.68    |    0.66    |    0.68    |    0.64    |    0.64    |    0.64    |    0.62    |    0.59    |    0.58    |    0.56    |    0.54    |    0.55    |    0.53    |    0.54    |    0.53    |    0.51    |    0.52    |    0.50    |    0.50    |    0.49    |    0.49    |    0.50    |    0.48    |    0.48    |    0.44    |    0.45    |    0.43   
grok-3                    |    0.66    |    0.67    |    0.68    |    1.00    |    0.65    |    0.77    |    0.63    |    0.67    |    0.63    |    0.68    |    0.63    |    0.61    |    0.61    |    0.57    |    0.57    |    0.61    |    0.60    |    0.63    |    0.57    |    0.54    |    0.55    |    0.55    |    0.58    |    0.58    |    0.59    |    0.53    |    0.52    |    0.52    |    0.50    |    0.52   
gemini-2.5-pro-think      |    0.65    |    0.65    |    0.66    |    0.65    |    1.00    |    0.62    |    0.66    |    0.66    |    0.67    |    0.57    |    0.56    |    0.57    |    0.57    |    0.55    |    0.53    |    0.51    |    0.60    |    0.54    |    0.44    |    0.60    |    0.50    |    0.51    |    0.55    |    0.51    |    0.49    |    0.49    |    0.50    |    0.49    |    0.50    |    0.51   
grok-4                    |    0.64    |    0.67    |    0.68    |    0.77    |    0.62    |    1.00    |    0.64    |    0.67    |    0.68    |    0.68    |    0.60    |    0.60    |    0.58    |    0.55    |    0.59    |    0.56    |    0.59    |    0.59    |    0.54    |    0.52    |    0.53    |    0.54    |    0.58    |    0.57    |    0.58    |    0.55    |    0.50    |    0.53    |    0.49    |    0.51   
o3                        |    0.64    |    0.63    |    0.64    |    0.63    |    0.66    |    0.64    |    1.00    |    0.78    |    0.79    |    0.57    |    0.58    |    0.59    |    0.56    |    0.58    |    0.58    |    0.52    |    0.60    |    0.54    |    0.41    |    0.53    |    0.53    |    0.49    |    0.60    |    0.55    |    0.54    |    0.51    |    0.49    |    0.51    |    0.51    |    0.53   
gpt-4.1                   |    0.63    |    0.62    |    0.64    |    0.67    |    0.66    |    0.67    |    0.78    |    1.00    |    0.76    |    0.59    |    0.60    |    0.59    |    0.59    |    0.58    |    0.60    |    0.54    |    0.61    |    0.57    |    0.43    |    0.57    |    0.55    |    0.54    |    0.63    |    0.57    |    0.57    |    0.52    |    0.50    |    0.56    |    0.53    |    0.53   
gpt-5-think               |    0.64    |    0.63    |    0.64    |    0.63    |    0.67    |    0.68    |    0.79    |    0.76    |    1.00    |    0.56    |    0.57    |    0.58    |    0.58    |    0.58    |    0.61    |    0.53    |    0.61    |    0.55    |    0.43    |    0.56    |    0.52    |    0.52    |    0.57    |    0.53    |    0.54    |    0.53    |    0.49    |    0.51    |    0.52    |    0.53   
kimi-k2.5-think           |    0.57    |    0.60    |    0.62    |    0.68    |    0.57    |    0.68    |    0.57    |    0.59    |    0.56    |    1.00    |    0.64    |    0.61    |    0.59    |    0.56    |    0.59    |    0.57    |    0.55    |    0.63    |    0.61    |    0.50    |    0.57    |    0.57    |    0.57    |    0.59    |    0.60    |    0.58    |    0.57    |    0.54    |    0.52    |    0.51   
glm-5-think               |    0.56    |    0.56    |    0.59    |    0.63    |    0.56    |    0.60    |    0.58    |    0.60    |    0.57    |    0.64    |    1.00    |    0.69    |    0.59    |    0.54    |    0.60    |    0.59    |    0.54    |    0.57    |    0.53    |    0.51    |    0.59    |    0.57    |    0.52    |    0.58    |    0.57    |    0.57    |    0.59    |    0.53    |    0.53    |    0.54   
glm-5.1-think             |    0.56    |    0.58    |    0.58    |    0.61    |    0.57    |    0.60    |    0.59    |    0.59    |    0.58    |    0.61    |    0.69    |    1.00    |    0.63    |    0.56    |    0.57    |    0.57    |    0.56    |    0.59    |    0.54    |    0.54    |    0.53    |    0.59    |    0.55    |    0.58    |    0.56    |    0.55    |    0.59    |    0.55    |    0.57    |    0.55   
claude-opus-4.6           |    0.55    |    0.56    |    0.56    |    0.61    |    0.57    |    0.58    |    0.56    |    0.59    |    0.58    |    0.59    |    0.59    |    0.63    |    1.00    |    0.54    |    0.54    |    0.54    |    0.54    |    0.73    |    0.51    |    0.55    |    0.49    |    0.61    |    0.57    |    0.57    |    0.74    |    0.66    |    0.56    |    0.55    |    0.57    |    0.54   
glm-4.7-think             |    0.52    |    0.53    |    0.54    |    0.57    |    0.55    |    0.55    |    0.58    |    0.58    |    0.58    |    0.56    |    0.54    |    0.56    |    0.54    |    1.00    |    0.63    |    0.69    |    0.56    |    0.56    |    0.43    |    0.50    |    0.50    |    0.56    |    0.53    |    0.54    |    0.55    |    0.50    |    0.52    |    0.47    |    0.51    |    0.62   
glm-5-turbo-think         |    0.51    |    0.53    |    0.55    |    0.57    |    0.53    |    0.59    |    0.58    |    0.60    |    0.61    |    0.59    |    0.60    |    0.57    |    0.54    |    0.63    |    1.00    |    0.65    |    0.57    |    0.55    |    0.44    |    0.49    |    0.53    |    0.53    |    0.54    |    0.56    |    0.56    |    0.51    |    0.50    |    0.52    |    0.50    |    0.60   
glm-4.6-think             |    0.49    |    0.52    |    0.53    |    0.61    |    0.51    |    0.56    |    0.52    |    0.54    |    0.53    |    0.57    |    0.59    |    0.57    |    0.54    |    0.69    |    0.65    |    1.00    |    0.54    |    0.57    |    0.46    |    0.50    |    0.54    |    0.53    |    0.47    |    0.55    |    0.57    |    0.51    |    0.54    |    0.48    |    0.51    |    0.62   
gpt-5.4                   |    0.53    |    0.54    |    0.54    |    0.60    |    0.60    |    0.59    |    0.60    |    0.61    |    0.61    |    0.55    |    0.54    |    0.56    |    0.54    |    0.56    |    0.57    |    0.54    |    1.00    |    0.53    |    0.41    |    0.61    |    0.51    |    0.53    |    0.52    |    0.54    |    0.53    |    0.53    |    0.50    |    0.52    |    0.51    |    0.53   
claude-opus-4.5           |    0.51    |    0.52    |    0.53    |    0.63    |    0.54    |    0.59    |    0.54    |    0.57    |    0.55    |    0.63    |    0.57    |    0.59    |    0.73    |    0.56    |    0.55    |    0.57    |    0.53    |    1.00    |    0.54    |    0.52    |    0.52    |    0.58    |    0.58    |    0.58    |    0.78    |    0.62    |    0.64    |    0.53    |    0.58    |    0.56   
grok-4.20                 |    0.48    |    0.53    |    0.51    |    0.57    |    0.44    |    0.54    |    0.41    |    0.43    |    0.43    |    0.61    |    0.53    |    0.54    |    0.51    |    0.43    |    0.44    |    0.46    |    0.41    |    0.54    |    1.00    |    0.39    |    0.44    |    0.49    |    0.40    |    0.45    |    0.47    |    0.51    |    0.52    |    0.41    |    0.43    |    0.46   
gemini-2.5-flash-think    |    0.49    |    0.51    |    0.52    |    0.54    |    0.60    |    0.52    |    0.53    |    0.57    |    0.56    |    0.50    |    0.51    |    0.54    |    0.55    |    0.50    |    0.49    |    0.50    |    0.61    |    0.52    |    0.39    |    1.00    |    0.50    |    0.47    |    0.49    |    0.52    |    0.54    |    0.52    |    0.49    |    0.55    |    0.52    |    0.51   
seed-2.0-lite-think       |    0.47    |    0.49    |    0.50    |    0.55    |    0.50    |    0.53    |    0.53    |    0.55    |    0.52    |    0.57    |    0.59    |    0.53    |    0.49    |    0.50    |    0.53    |    0.54    |    0.51    |    0.52    |    0.44    |    0.50    |    1.00    |    0.49    |    0.48    |    0.51    |    0.51    |    0.51    |    0.53    |    0.52    |    0.47    |    0.49   
deepseek-r1-think         |    0.48    |    0.50    |    0.50    |    0.55    |    0.51    |    0.54    |    0.49    |    0.54    |    0.52    |    0.57    |    0.57    |    0.59    |    0.61    |    0.56    |    0.53    |    0.53    |    0.53    |    0.58    |    0.49    |    0.47    |    0.49    |    1.00    |    0.50    |    0.58    |    0.55    |    0.59    |    0.72    |    0.51    |    0.59    |    0.50   
gpt-4                     |    0.48    |    0.48    |    0.49    |    0.58    |    0.55    |    0.58    |    0.60    |    0.63    |    0.57    |    0.57    |    0.52    |    0.55    |    0.57    |    0.53    |    0.54    |    0.47    |    0.52    |    0.58    |    0.40    |    0.49    |    0.48    |    0.50    |    1.00    |    0.56    |    0.57    |    0.50    |    0.47    |    0.55    |    0.52    |    0.52   
kimi-k2                   |    0.46    |    0.47    |    0.49    |    0.58    |    0.51    |    0.57    |    0.55    |    0.57    |    0.53    |    0.59    |    0.58    |    0.58    |    0.57    |    0.54    |    0.56    |    0.55    |    0.54    |    0.58    |    0.45    |    0.52    |    0.51    |    0.58    |    0.56    |    1.00    |    0.61    |    0.53    |    0.53    |    0.56    |    0.60    |    0.58   
claude-opus-4.5-think     |    0.46    |    0.48    |    0.50    |    0.59    |    0.49    |    0.58    |    0.54    |    0.57    |    0.54    |    0.60    |    0.57    |    0.56    |    0.74    |    0.55    |    0.56    |    0.57    |    0.53    |    0.78    |    0.47    |    0.54    |    0.51    |    0.55    |    0.57    |    0.61    |    1.00    |    0.65    |    0.57    |    0.57    |    0.58    |    0.57   
claude-sonnet-4.6         |    0.46    |    0.48    |    0.48    |    0.53    |    0.49    |    0.55    |    0.51    |    0.52    |    0.53    |    0.58    |    0.57    |    0.55    |    0.66    |    0.50    |    0.51    |    0.51    |    0.53    |    0.62    |    0.51    |    0.52    |    0.51    |    0.59    |    0.50    |    0.53    |    0.65    |    1.00    |    0.59    |    0.52    |    0.54    |    0.52   
deepseek-v3.2-think       |    0.45    |    0.47    |    0.48    |    0.52    |    0.50    |    0.50    |    0.49    |    0.50    |    0.49    |    0.57    |    0.59    |    0.59    |    0.56    |    0.52    |    0.50    |    0.54    |    0.50    |    0.64    |    0.52    |    0.49    |    0.53    |    0.72    |    0.47    |    0.53    |    0.57    |    0.59    |    1.00    |    0.50    |    0.60    |    0.52   
qwen3-max                 |    0.44    |    0.44    |    0.44    |    0.52    |    0.49    |    0.53    |    0.51    |    0.56    |    0.51    |    0.54    |    0.53    |    0.55    |    0.55    |    0.47    |    0.52    |    0.48    |    0.52    |    0.53    |    0.41    |    0.55    |    0.52    |    0.51    |    0.55    |    0.56    |    0.57    |    0.52    |    0.50    |    1.00    |    0.54    |    0.50   
deepseek-v3               |    0.44    |    0.44    |    0.45    |    0.50    |    0.50    |    0.49    |    0.51    |    0.53    |    0.52    |    0.52    |    0.53    |    0.57    |    0.57    |    0.51    |    0.50    |    0.51    |    0.51    |    0.58    |    0.43    |    0.52    |    0.47    |    0.59    |    0.52    |    0.60    |    0.58    |    0.54    |    0.60    |    0.54    |    1.00    |    0.54   
glm-4.5-think             |    0.41    |    0.43    |    0.43    |    0.52    |    0.51    |    0.51    |    0.53    |    0.53    |    0.53    |    0.51    |    0.54    |    0.55    |    0.54    |    0.62    |    0.60    |    0.62    |    0.53    |    0.56    |    0.46    |    0.51    |    0.49    |    0.50    |    0.52    |    0.58    |    0.57    |    0.52    |    0.52    |    0.50    |    0.54    |    1.00   
```

### Clusters (Jaccard > 0.50)

Pairs with Jaccard similarity > 0.50 on T5-T6 correct sets:

| Model A | Model B | Jaccard | |A|  | |B|  | |A∩B| |
|---------|---------|---------|------|------|-------|
| gemini-3-flash-think | gemini-3-flash | 0.901 | 360 | 347 | 335 |
| gemini-3.1-pro | gemini-3-flash-think | 0.880 | 377 | 360 | 345 |
| gemini-3.1-pro | gemini-3-flash | 0.852 | 377 | 347 | 333 |
| nemotron-70b | llama-3.1-70b | 0.792 | 50 | 45 | 42 |
| o3 | gpt-5-think | 0.787 | 251 | 244 | 218 |
| o3 | gpt-4.1 | 0.781 | 251 | 246 | 218 |
| claude-opus-4.5 | claude-opus-4.5-think | 0.778 | 199 | 178 | 165 |
| grok-3 | grok-4 | 0.772 | 260 | 252 | 223 |
| gpt-4.1 | gpt-5-think | 0.763 | 246 | 244 | 212 |
| claude-opus-4.6 | claude-opus-4.5-think | 0.738 | 213 | 178 | 166 |
| claude-opus-4.6 | claude-opus-4.5 | 0.731 | 213 | 199 | 174 |
| qwen3.5-plus-think | qwen3.5-397b-a17b-think | 0.722 | 142 | 137 | 117 |
| deepseek-r1-think | deepseek-v3.2-think | 0.719 | 185 | 176 | 151 |
| glm-5-think | glm-5.1-think | 0.690 | 219 | 217 | 178 |
| glm-4.7-think | glm-4.6-think | 0.686 | 207 | 201 | 166 |
| grok-4 | kimi-k2.5-think | 0.679 | 252 | 230 | 195 |
| grok-3 | kimi-k2.5-think | 0.678 | 260 | 230 | 198 |
| gemini-3-flash | grok-4 | 0.678 | 347 | 252 | 242 |
| gemini-3-flash | grok-3 | 0.677 | 347 | 260 | 245 |
| grok-4 | gpt-5-think | 0.676 | 252 | 244 | 200 |
| grok-3 | gpt-4.1 | 0.670 | 260 | 246 | 203 |
| gemini-2.5-pro-think | gpt-5-think | 0.669 | 260 | 244 | 202 |
| gemini-3-flash-think | grok-4 | 0.668 | 360 | 252 | 245 |
| gemini-3-flash-think | grok-3 | 0.667 | 360 | 260 | 248 |
| grok-4 | gpt-4.1 | 0.666 | 252 | 246 | 199 |
| gemini-2.5-pro-think | o3 | 0.664 | 260 | 251 | 204 |
| gemini-2.5-pro-think | gpt-4.1 | 0.659 | 260 | 246 | 201 |
| gemini-3.1-pro | grok-3 | 0.659 | 377 | 260 | 253 |
| gemini-3-flash | gemini-2.5-pro-think | 0.658 | 347 | 260 | 241 |
| mimo-v2-flash | mimo-v2-flash-think | 0.657 | 143 | 132 | 109 |
| claude-opus-4.6 | claude-sonnet-4.6 | 0.655 | 213 | 176 | 154 |
| gemini-3-flash-think | gemini-2.5-pro-think | 0.653 | 360 | 260 | 245 |
| gemini-3.1-pro | gemini-2.5-pro-think | 0.650 | 377 | 260 | 251 |
| glm-5-turbo-think | glm-4.6-think | 0.649 | 203 | 201 | 159 |
| claude-opus-4.5-think | claude-sonnet-4.6 | 0.647 | 178 | 176 | 139 |
| grok-3 | gemini-2.5-pro-think | 0.646 | 260 | 260 | 204 |
| gemini-3.1-pro | o3 | 0.644 | 377 | 251 | 246 |
| grok-4 | o3 | 0.644 | 252 | 251 | 197 |
| gemini-3-flash | o3 | 0.643 | 347 | 251 | 234 |
| gemini-3.1-pro | grok-4 | 0.642 | 377 | 252 | 246 |
| kimi-k2.5-think | glm-5-think | 0.639 | 230 | 219 | 175 |
| gemini-3.1-pro | gpt-5-think | 0.639 | 377 | 244 | 242 |
| gemini-3-flash | gpt-4.1 | 0.638 | 347 | 246 | 231 |
| claude-opus-4.5 | deepseek-v3.2-think | 0.638 | 199 | 176 | 146 |
| gemini-3-flash | gpt-5-think | 0.637 | 347 | 244 | 230 |
| gemini-3-flash-think | o3 | 0.634 | 360 | 251 | 237 |
| glm-4.7-think | glm-5-turbo-think | 0.633 | 207 | 203 | 159 |
| gpt-4.1 | gpt-4 | 0.631 | 246 | 183 | 166 |
| kimi-k2.5-think | claude-opus-4.5 | 0.631 | 230 | 199 | 166 |
| gemini-3.1-pro | gpt-4.1 | 0.631 | 377 | 246 | 241 |
| grok-3 | glm-5-think | 0.629 | 260 | 219 | 185 |
| glm-5.1-think | claude-opus-4.6 | 0.629 | 217 | 213 | 166 |
| deepseek-v3 | deepseek-v3.2 | 0.628 | 167 | 144 | 120 |
| gemini-3-flash-think | gpt-5-think | 0.628 | 360 | 244 | 233 |
| grok-3 | claude-opus-4.5 | 0.628 | 260 | 199 | 177 |
| grok-3 | o3 | 0.627 | 260 | 251 | 197 |
| grok-3 | gpt-5-think | 0.626 | 260 | 244 | 194 |
| gpt-4.1-mini | o4-mini-think | 0.625 | 145 | 115 | 100 |
| gemini-3-flash-think | gpt-4.1 | 0.625 | 360 | 246 | 233 |
| claude-opus-4.5 | claude-sonnet-4.6 | 0.623 | 199 | 176 | 144 |
| glm-4.6-think | glm-4.5-think | 0.621 | 201 | 162 | 139 |
| glm-4.7-think | glm-4.5-think | 0.618 | 207 | 162 | 141 |
| gemini-3-flash | kimi-k2.5-think | 0.616 | 347 | 230 | 220 |
| gemini-2.5-pro-think | grok-4 | 0.615 | 260 | 252 | 195 |
| gpt-4.1 | gpt-5.4 | 0.614 | 246 | 201 | 170 |
| gpt-5-think | glm-5-turbo-think | 0.614 | 244 | 203 | 170 |
| gpt-5-think | gpt-5.4 | 0.612 | 244 | 201 | 169 |
| kimi-k2 | claude-opus-4.5-think | 0.612 | 183 | 178 | 137 |
| grok-3 | glm-5.1-think | 0.611 | 260 | 217 | 181 |
| claude-opus-4.6 | deepseek-r1-think | 0.611 | 213 | 185 | 151 |
| grok-3 | claude-opus-4.6 | 0.609 | 260 | 213 | 179 |
| kimi-k2.5-think | glm-5.1-think | 0.608 | 230 | 217 | 169 |
| kimi-k2.5-think | grok-4.20 | 0.608 | 230 | 196 | 161 |
| gpt-5.4 | gemini-2.5-flash-think | 0.607 | 201 | 191 | 148 |
| grok-3 | glm-4.6-think | 0.606 | 260 | 201 | 174 |
| glm-5-think | glm-5-turbo-think | 0.605 | 219 | 203 | 159 |
| gpt-4.1 | glm-5-turbo-think | 0.604 | 246 | 203 | 169 |
| gemini-3-flash-think | kimi-k2.5-think | 0.603 | 360 | 230 | 222 |
| o3 | gpt-5.4 | 0.603 | 251 | 201 | 170 |
| deepseek-v3.2-think | deepseek-v3 | 0.603 | 176 | 167 | 129 |
| llama-3.3-70b | nemotron-70b | 0.603 | 67 | 50 | 44 |
| grok-4 | glm-5-think | 0.602 | 252 | 219 | 177 |
| glm-5-turbo-think | glm-4.5-think | 0.601 | 203 | 162 | 137 |
| grok-3 | gpt-5.4 | 0.601 | 260 | 201 | 173 |
| gemini-2.5-pro-think | gpt-5.4 | 0.601 | 260 | 201 | 173 |
| kimi-k2.5-think | claude-opus-4.5-think | 0.600 | 230 | 178 | 153 |
| gemini-2.5-pro-think | gemini-2.5-flash-think | 0.599 | 260 | 191 | 169 |
| kimi-k2 | deepseek-v3 | 0.598 | 183 | 167 | 131 |
| gpt-4.1 | glm-5-think | 0.598 | 246 | 219 | 174 |
| o3 | gpt-4 | 0.596 | 251 | 183 | 162 |
| grok-4 | glm-5.1-think | 0.595 | 252 | 217 | 175 |
| kimi-k2.5-think | kimi-k2 | 0.595 | 230 | 183 | 154 |
| glm-5-think | seed-2.0-lite-think | 0.594 | 219 | 186 | 151 |
| glm-5-think | claude-opus-4.6 | 0.594 | 219 | 213 | 161 |
| glm-5.1-think | claude-opus-4.5 | 0.594 | 217 | 199 | 155 |
| deepseek-r1-think | deepseek-v3 | 0.593 | 185 | 167 | 131 |
| glm-5-think | deepseek-v3.2-think | 0.593 | 219 | 176 | 147 |
| gpt-4.1 | kimi-k2.5-think | 0.592 | 246 | 230 | 177 |
| kimi-k2.5-think | glm-5-turbo-think | 0.592 | 230 | 203 | 161 |
| o3 | glm-5.1-think | 0.592 | 251 | 217 | 174 |
| mistral-large | nemotron-70b | 0.592 | 63 | 50 | 42 |
| glm-5.1-think | deepseek-v3.2-think | 0.591 | 217 | 176 | 146 |
| glm-5-think | glm-4.6-think | 0.591 | 219 | 201 | 156 |
| deepseek-r1-think | claude-sonnet-4.6 | 0.590 | 185 | 176 | 134 |
| gemini-3-flash | glm-5-think | 0.590 | 347 | 219 | 210 |
| grok-4 | gpt-5.4 | 0.589 | 252 | 201 | 168 |
| glm-5.1-think | deepseek-r1-think | 0.589 | 217 | 185 | 149 |
| gpt-4.1 | claude-opus-4.6 | 0.588 | 246 | 213 | 170 |
| grok-4 | claude-opus-4.5 | 0.588 | 252 | 199 | 167 |
| kimi-k2.5-think | claude-opus-4.6 | 0.588 | 230 | 213 | 164 |
| grok-3 | claude-opus-4.5-think | 0.587 | 260 | 178 | 162 |
| gpt-4.1 | glm-5.1-think | 0.586 | 246 | 217 | 171 |
| claude-sonnet-4.6 | deepseek-v3.2-think | 0.586 | 176 | 176 | 130 |
| grok-4 | glm-5-turbo-think | 0.585 | 252 | 203 | 168 |
| claude-opus-4.5 | deepseek-v3 | 0.584 | 199 | 167 | 135 |
| glm-5-think | kimi-k2 | 0.583 | 219 | 183 | 148 |
| claude-opus-4.5-think | deepseek-v3 | 0.583 | 178 | 167 | 127 |
| gpt-5-think | glm-4.7-think | 0.582 | 244 | 207 | 166 |
| grok-3 | gpt-4 | 0.582 | 260 | 183 | 163 |
| o3 | glm-5-turbo-think | 0.582 | 251 | 203 | 167 |
| grok-4 | claude-opus-4.6 | 0.582 | 252 | 213 | 171 |
| gpt-5-think | claude-opus-4.6 | 0.581 | 244 | 213 | 168 |
| glm-5.1-think | kimi-k2 | 0.581 | 217 | 183 | 147 |
| claude-opus-4.5 | deepseek-r1-think | 0.580 | 199 | 185 | 141 |
| kimi-k2.5-think | claude-sonnet-4.6 | 0.580 | 230 | 176 | 149 |
| qwen3.5-flash-think | qwen3.5-35b-a3b-think | 0.580 | 55 | 54 | 40 |
| deepseek-r1-think | kimi-k2 | 0.579 | 185 | 183 | 135 |
| o3 | glm-4.7-think | 0.579 | 251 | 207 | 168 |
| gpt-5-think | glm-5.1-think | 0.579 | 244 | 217 | 169 |
| claude-opus-4.5 | gpt-4 | 0.579 | 199 | 183 | 140 |
| claude-opus-4.5 | kimi-k2 | 0.579 | 199 | 183 | 140 |
| gpt-4.1 | glm-4.7-think | 0.578 | 246 | 207 | 166 |
| o3 | glm-5-think | 0.577 | 251 | 219 | 172 |
| grok-3 | kimi-k2 | 0.577 | 260 | 183 | 162 |
| gemini-3-flash-think | glm-5.1-think | 0.577 | 360 | 217 | 211 |
| grok-4 | gpt-4 | 0.576 | 252 | 183 | 159 |
| gemini-3-flash | glm-5.1-think | 0.575 | 347 | 217 | 206 |
| kimi-k2 | glm-4.5-think | 0.575 | 183 | 162 | 126 |
| grok-4 | claude-opus-4.5-think | 0.575 | 252 | 178 | 157 |
| gpt-5-think | glm-5-think | 0.575 | 244 | 219 | 169 |
| gemini-2.5-pro-think | glm-5.1-think | 0.574 | 260 | 217 | 174 |
| claude-opus-4.5-think | glm-4.5-think | 0.574 | 178 | 162 | 124 |
| glm-5.1-think | deepseek-v3 | 0.574 | 217 | 167 | 140 |
| kimi-k2.5-think | deepseek-v3.2-think | 0.574 | 230 | 176 | 148 |
| claude-opus-4.5-think | deepseek-v3.2-think | 0.573 | 178 | 176 | 129 |
| glm-5-think | deepseek-r1-think | 0.572 | 219 | 185 | 147 |
| glm-5-turbo-think | gpt-5.4 | 0.572 | 203 | 201 | 147 |
| gemini-2.5-pro-think | claude-opus-4.6 | 0.571 | 260 | 213 | 172 |
| glm-5-think | claude-opus-4.5 | 0.571 | 219 | 199 | 152 |
| glm-5.1-think | glm-4.6-think | 0.571 | 217 | 201 | 152 |
| claude-opus-4.6 | kimi-k2 | 0.571 | 213 | 183 | 144 |
| gemini-2.5-pro-think | kimi-k2.5-think | 0.571 | 260 | 230 | 178 |
| grok-4 | kimi-k2 | 0.570 | 252 | 183 | 158 |
| gpt-4.1 | claude-opus-4.5-think | 0.570 | 246 | 178 | 154 |
| kimi-k2.5-think | gpt-4 | 0.570 | 230 | 183 | 150 |
| claude-opus-4.6 | deepseek-v3 | 0.570 | 213 | 167 | 138 |
| gpt-5-think | gpt-4 | 0.570 | 244 | 183 | 155 |
| kimi-k2.5-think | seed-2.0-lite-think | 0.570 | 230 | 186 | 151 |
| gpt-4 | claude-opus-4.5-think | 0.570 | 183 | 178 | 131 |
| grok-3 | glm-5-turbo-think | 0.569 | 260 | 203 | 168 |
| glm-5-think | claude-opus-4.5-think | 0.569 | 219 | 178 | 144 |
| glm-4.6-think | claude-opus-4.5 | 0.569 | 201 | 199 | 145 |
| gemini-3.1-pro | kimi-k2.5-think | 0.568 | 377 | 230 | 220 |
| claude-opus-4.5-think | qwen3-max | 0.568 | 178 | 167 | 125 |
| glm-5-think | claude-sonnet-4.6 | 0.567 | 219 | 176 | 143 |
| kimi-k2.5-think | glm-4.6-think | 0.567 | 230 | 201 | 156 |
| glm-5.1-think | glm-5-turbo-think | 0.567 | 217 | 203 | 152 |
| grok-3 | glm-4.7-think | 0.567 | 260 | 207 | 169 |
| grok-3 | grok-4.20 | 0.567 | 260 | 196 | 165 |
| gpt-4.1 | claude-opus-4.5 | 0.567 | 246 | 199 | 161 |
| o3 | kimi-k2.5-think | 0.567 | 251 | 230 | 174 |
| gpt-4.1 | gemini-2.5-flash-think | 0.566 | 246 | 191 | 158 |
| glm-4.6-think | claude-opus-4.5-think | 0.566 | 201 | 178 | 137 |
| kimi-k2.5-think | deepseek-r1-think | 0.566 | 230 | 185 | 150 |
| gpt-4.1 | kimi-k2 | 0.566 | 246 | 183 | 155 |
| claude-opus-4.6 | gpt-4 | 0.565 | 213 | 183 | 143 |
| gemini-3-flash-think | glm-5-think | 0.565 | 360 | 219 | 209 |
| gpt-5-think | gemini-2.5-flash-think | 0.565 | 244 | 191 | 157 |
| glm-5.1-think | glm-4.7-think | 0.565 | 217 | 207 | 153 |
| gpt-5-think | kimi-k2.5-think | 0.564 | 244 | 230 | 171 |
| claude-opus-4.5 | glm-4.5-think | 0.563 | 199 | 162 | 130 |
| glm-5-turbo-think | kimi-k2 | 0.563 | 203 | 183 | 139 |
| claude-opus-4.6 | deepseek-v3.2-think | 0.562 | 213 | 176 | 140 |
| grok-4 | glm-4.6-think | 0.562 | 252 | 201 | 163 |
| glm-5-turbo-think | claude-opus-4.5-think | 0.561 | 203 | 178 | 137 |
| gemini-3-flash-think | claude-opus-4.6 | 0.561 | 360 | 213 | 206 |
| glm-5.1-think | claude-opus-4.5-think | 0.561 | 217 | 178 | 142 |
| deepseek-v3.2-think | deepseek-v3.2 | 0.561 | 176 | 144 | 115 |
| gemini-3.1-pro | glm-5-think | 0.560 | 377 | 219 | 214 |
| glm-5.1-think | gpt-5.4 | 0.560 | 217 | 201 | 150 |
| claude-opus-4.5 | grok-4.20-think | 0.559 | 199 | 155 | 127 |
| gemini-3.1-pro | glm-5.1-think | 0.559 | 377 | 217 | 213 |
| gpt-4.1 | qwen3-max | 0.558 | 246 | 167 | 148 |
| gpt-4 | kimi-k2 | 0.557 | 183 | 183 | 131 |
| glm-4.7-think | gpt-5.4 | 0.557 | 207 | 201 | 146 |
| o3 | claude-opus-4.6 | 0.557 | 251 | 213 | 166 |
| claude-opus-4.5-think | grok-4.20-think | 0.556 | 178 | 155 | 119 |
| gemini-3-flash | claude-opus-4.6 | 0.556 | 347 | 213 | 200 |
| glm-4.7-think | claude-opus-4.5 | 0.556 | 207 | 199 | 145 |
| glm-4.7-think | deepseek-r1-think | 0.556 | 207 | 185 | 140 |
| kimi-k2 | qwen3-max | 0.556 | 183 | 167 | 125 |
| gemini-2.5-pro-think | glm-5-think | 0.555 | 260 | 219 | 171 |
| kimi-k2.5-think | glm-4.7-think | 0.555 | 230 | 207 | 156 |
| glm-5.1-think | qwen3-max | 0.555 | 217 | 167 | 137 |
| glm-4.6-think | kimi-k2 | 0.555 | 201 | 183 | 137 |
| gpt-4 | gpt-4o | 0.555 | 183 | 145 | 117 |
| gemini-2.5-pro-think | gpt-4 | 0.554 | 260 | 183 | 158 |
| grok-3 | seed-2.0-lite-think | 0.554 | 260 | 186 | 159 |
| claude-opus-4.6 | gemini-2.5-flash-think | 0.554 | 213 | 191 | 144 |
| gemini-3-flash | glm-5-turbo-think | 0.554 | 347 | 203 | 196 |
| glm-5.1-think | claude-sonnet-4.6 | 0.553 | 217 | 176 | 140 |
| gemini-3.1-pro | claude-opus-4.6 | 0.553 | 377 | 213 | 210 |
| kimi-k2.5-think | grok-4.20-think | 0.552 | 230 | 155 | 137 |
| glm-5-turbo-think | claude-opus-4.5 | 0.552 | 203 | 199 | 143 |
| claude-opus-4.5 | deepseek-v3.2 | 0.552 | 199 | 144 | 122 |
| deepseek-r1-think | deepseek-v3.2 | 0.552 | 185 | 144 | 117 |
| deepseek-r1-think | claude-opus-4.5-think | 0.551 | 185 | 178 | 129 |
| claude-opus-4.6 | qwen3-max | 0.551 | 213 | 167 | 135 |
| grok-4 | glm-4.7-think | 0.551 | 252 | 207 | 163 |
| glm-5.1-think | gpt-4 | 0.550 | 217 | 183 | 142 |
| kimi-k2.5-think | gpt-5.4 | 0.550 | 230 | 201 | 153 |
| o3 | kimi-k2 | 0.550 | 251 | 183 | 154 |
| gemini-2.5-flash-think | qwen3-max | 0.550 | 191 | 167 | 127 |
| gpt-5-think | claude-opus-4.5 | 0.549 | 244 | 199 | 157 |
| gpt-4 | qwen3-max | 0.549 | 183 | 167 | 124 |
| gpt-4.1 | seed-2.0-lite-think | 0.548 | 246 | 186 | 153 |
| claude-opus-4.5-think | deepseek-v3.2 | 0.548 | 178 | 144 | 114 |
| glm-5.1-think | glm-4.5-think | 0.547 | 217 | 162 | 134 |
| gemini-2.5-pro-think | glm-4.7-think | 0.546 | 260 | 207 | 165 |
| grok-4.20 | grok-4.20-think | 0.546 | 196 | 155 | 124 |
| glm-4.7-think | claude-opus-4.5-think | 0.546 | 207 | 178 | 136 |
| claude-sonnet-4.6 | deepseek-v3.2 | 0.546 | 176 | 144 | 113 |
| grok-3 | deepseek-r1-think | 0.545 | 260 | 185 | 157 |
| grok-4 | claude-sonnet-4.6 | 0.545 | 252 | 176 | 151 |
| claude-opus-4.6 | glm-4.6-think | 0.545 | 213 | 201 | 146 |
| claude-opus-4.6 | gpt-5.4 | 0.545 | 213 | 201 | 146 |
| glm-5-think | gpt-5.4 | 0.544 | 219 | 201 | 148 |
| glm-5-turbo-think | gpt-4 | 0.544 | 203 | 183 | 136 |
| gemini-3-flash | gpt-5.4 | 0.544 | 347 | 201 | 193 |
| glm-5-think | glm-4.7-think | 0.543 | 219 | 207 | 150 |
| claude-opus-4.6 | glm-4.5-think | 0.543 | 213 | 162 | 132 |
| llama-3-70b | nemotron-70b | 0.543 | 75 | 50 | 44 |
| gemini-3-flash | glm-4.7-think | 0.543 | 347 | 207 | 195 |
| glm-5-think | glm-4.5-think | 0.543 | 219 | 162 | 134 |
| gpt-4.1 | glm-4.6-think | 0.541 | 246 | 201 | 157 |
| claude-sonnet-4.6 | mimo-v2-flash | 0.541 | 176 | 143 | 112 |
| deepseek-v3.2-think | mimo-v2-flash | 0.541 | 176 | 143 | 112 |
| claude-opus-4.6 | glm-5-turbo-think | 0.541 | 213 | 203 | 146 |
| glm-4.6-think | gpt-5.4 | 0.540 | 201 | 201 | 141 |
| gpt-5-think | claude-opus-4.5-think | 0.540 | 244 | 178 | 148 |
| glm-5.1-think | gemini-2.5-flash-think | 0.540 | 217 | 191 | 143 |
| grok-4 | grok-4.20 | 0.540 | 252 | 196 | 157 |
| gpt-4.1 | deepseek-r1-think | 0.539 | 246 | 185 | 151 |
| grok-3 | gemini-2.5-flash-think | 0.539 | 260 | 191 | 158 |
| qwen3-max | deepseek-v3 | 0.539 | 167 | 167 | 117 |
| glm-4.6-think | deepseek-v3.2-think | 0.539 | 201 | 176 | 132 |
| kimi-k2.5-think | qwen3-max | 0.539 | 230 | 167 | 139 |
| grok-4 | deepseek-r1-think | 0.539 | 252 | 185 | 153 |
| claude-opus-4.6 | glm-4.7-think | 0.538 | 213 | 207 | 147 |
| claude-sonnet-4.6 | deepseek-v3 | 0.538 | 176 | 167 | 120 |
| o3 | claude-opus-4.5-think | 0.538 | 251 | 178 | 150 |
| gemini-2.5-flash-think | claude-opus-4.5-think | 0.537 | 191 | 178 | 129 |
| deepseek-v3 | glm-4.5-think | 0.537 | 167 | 162 | 115 |
| gemini-3-flash-think | gpt-5.4 | 0.537 | 360 | 201 | 196 |
| claude-opus-4.5 | grok-4.20 | 0.537 | 199 | 196 | 138 |
| gpt-5.4 | kimi-k2 | 0.536 | 201 | 183 | 134 |
| o3 | claude-opus-4.5 | 0.536 | 251 | 199 | 157 |
| glm-4.6-think | seed-2.0-lite-think | 0.536 | 201 | 186 | 135 |
| glm-4.7-think | kimi-k2 | 0.535 | 207 | 183 | 136 |
| glm-5.1-think | grok-4.20 | 0.535 | 217 | 196 | 144 |
| kimi-k2 | deepseek-v3.2 | 0.535 | 183 | 144 | 114 |
| gemini-2.5-pro-think | claude-opus-4.5 | 0.535 | 260 | 199 | 160 |
| o3 | gemini-2.5-flash-think | 0.535 | 251 | 191 | 154 |
| gpt-5-think | glm-4.6-think | 0.534 | 244 | 201 | 155 |
| llama-3.3-70b | llama-3.1-70b | 0.534 | 67 | 45 | 39 |
| kimi-k2 | deepseek-v3.2-think | 0.534 | 183 | 176 | 125 |
| gemini-3-flash-think | glm-5-turbo-think | 0.534 | 360 | 203 | 196 |
| glm-5-turbo-think | deepseek-r1-think | 0.534 | 203 | 185 | 135 |
| o3 | seed-2.0-lite-think | 0.533 | 251 | 186 | 152 |
| llama-3-70b | mistral-large | 0.533 | 75 | 63 | 48 |
| gemini-2.5-pro-think | glm-5-turbo-think | 0.533 | 260 | 203 | 161 |
| qwen3.5-plus-think | qwen3.6-plus-think | 0.533 | 142 | 114 | 89 |
| gpt-5-think | claude-sonnet-4.6 | 0.533 | 244 | 176 | 146 |
| glm-5.1-think | seed-2.0-lite-think | 0.532 | 217 | 186 | 140 |
| glm-5-think | qwen3-max | 0.532 | 219 | 167 | 134 |
| glm-5-think | deepseek-v3 | 0.532 | 219 | 167 | 134 |
| gpt-5.4 | deepseek-r1-think | 0.532 | 201 | 185 | 134 |
| gpt-5.4 | glm-4.5-think | 0.532 | 201 | 162 | 126 |
| glm-5-turbo-think | seed-2.0-lite-think | 0.531 | 203 | 186 | 135 |
| grok-4 | seed-2.0-lite-think | 0.531 | 252 | 186 | 152 |
| claude-opus-4.5 | qwen3-max | 0.531 | 199 | 167 | 127 |
| glm-5-think | grok-4.20 | 0.531 | 219 | 196 | 144 |
| gpt-5-think | kimi-k2 | 0.530 | 244 | 183 | 148 |
| glm-4.5-think | deepseek-v3.2 | 0.530 | 162 | 144 | 106 |
| grok-3 | claude-sonnet-4.6 | 0.530 | 260 | 176 | 151 |
| o3 | glm-4.5-think | 0.530 | 251 | 162 | 143 |
| gpt-4.1 | deepseek-v3 | 0.530 | 246 | 167 | 143 |
| gemini-3-flash | claude-opus-4.5 | 0.529 | 347 | 199 | 189 |
| glm-4.7-think | gpt-4 | 0.529 | 207 | 183 | 135 |
| grok-4 | qwen3-max | 0.529 | 252 | 167 | 145 |
| gemini-3.1-pro | gpt-5.4 | 0.529 | 377 | 201 | 200 |
| gemini-3-flash-think | glm-4.7-think | 0.528 | 360 | 207 | 196 |
| gpt-5.4 | claude-opus-4.5-think | 0.528 | 201 | 178 | 131 |
| gpt-4.1 | glm-4.5-think | 0.528 | 246 | 162 | 141 |
| kimi-k2 | claude-sonnet-4.6 | 0.528 | 183 | 176 | 124 |
| gemini-3-flash-think | grok-4.20 | 0.527 | 360 | 196 | 192 |
| seed-2.0-lite-think | deepseek-v3.2-think | 0.527 | 186 | 176 | 125 |
| gpt-4.1 | gpt-4o | 0.527 | 246 | 145 | 135 |
| llama-3-70b | llama-3.3-70b | 0.527 | 75 | 67 | 49 |
| gpt-5.4 | claude-opus-4.5 | 0.527 | 201 | 199 | 138 |
| gemini-3-flash | glm-4.6-think | 0.526 | 347 | 201 | 189 |
| gpt-5-think | glm-4.5-think | 0.526 | 244 | 162 | 140 |
| gpt-5.4 | claude-sonnet-4.6 | 0.526 | 201 | 176 | 130 |
| grok-4.20-think | grok-4.1-fast-think | 0.526 | 155 | 135 | 100 |
| grok-3 | grok-4.20-think | 0.526 | 260 | 155 | 143 |
| glm-4.6-think | deepseek-r1-think | 0.526 | 201 | 185 | 133 |
| claude-opus-4.6 | deepseek-v3.2 | 0.526 | 213 | 144 | 123 |
| gpt-5-think | seed-2.0-lite-think | 0.525 | 244 | 186 | 148 |
| gemini-3.1-pro | glm-4.7-think | 0.525 | 377 | 207 | 201 |
| grok-4.20 | deepseek-v3.2-think | 0.525 | 196 | 176 | 128 |
| claude-opus-4.5-think | llama-4-maverick | 0.525 | 178 | 133 | 107 |
| grok-3 | deepseek-v3.2-think | 0.524 | 260 | 176 | 150 |
| gemini-3-flash-think | glm-4.6-think | 0.524 | 360 | 201 | 193 |
| gemini-3-flash | gemini-2.5-flash-think | 0.524 | 347 | 191 | 185 |
| claude-opus-4.5-think | qwen3.5-plus-think | 0.524 | 178 | 142 | 110 |
| gpt-4.1 | claude-sonnet-4.6 | 0.523 | 246 | 176 | 145 |
| claude-opus-4.5 | gemini-2.5-flash-think | 0.523 | 199 | 191 | 134 |
| gemini-3-flash-think | claude-opus-4.5 | 0.523 | 360 | 199 | 192 |
| qwen3-max | llama-4-maverick | 0.523 | 167 | 133 | 103 |
| glm-5-think | gpt-4 | 0.523 | 219 | 183 | 138 |
| claude-sonnet-4.6 | glm-4.5-think | 0.523 | 176 | 162 | 116 |
| deepseek-v3.2-think | glm-4.5-think | 0.523 | 176 | 162 | 116 |
| deepseek-v3.2 | gpt-5.4-mini | 0.522 | 144 | 127 | 93 |
| grok-4 | gemini-2.5-flash-think | 0.522 | 252 | 191 | 152 |
| gpt-5-think | deepseek-v3 | 0.522 | 244 | 167 | 141 |
| o3 | glm-4.6-think | 0.522 | 251 | 201 | 155 |
| claude-opus-4.5 | seed-2.0-lite-think | 0.522 | 199 | 186 | 132 |
| gpt-4 | deepseek-v3 | 0.522 | 183 | 167 | 120 |
| qwen3-max | step-3.5-flash-think | 0.522 | 167 | 148 | 108 |
| seed-2.0-lite-think | qwen3-max | 0.522 | 186 | 167 | 121 |
| mistral-large | llama-3.1-70b | 0.521 | 63 | 45 | 37 |
| kimi-k2.5-think | deepseek-v3 | 0.521 | 230 | 167 | 136 |
| gpt-5.4 | qwen3-max | 0.521 | 201 | 167 | 126 |
| gemini-2.5-flash-think | kimi-k2 | 0.520 | 191 | 183 | 128 |
| glm-4.7-think | deepseek-v3.2-think | 0.520 | 207 | 176 | 131 |
| gpt-4 | glm-4.5-think | 0.520 | 183 | 162 | 118 |
| kimi-k2 | gpt-5.4-mini | 0.520 | 183 | 127 | 106 |
| grok-3 | qwen3-max | 0.520 | 260 | 167 | 146 |
| kimi-k2 | llama-4-maverick | 0.519 | 183 | 133 | 108 |
| gpt-4 | gpt-4.1-mini | 0.519 | 183 | 145 | 112 |
| grok-3 | glm-4.5-think | 0.518 | 260 | 162 | 144 |
| gpt-5.4 | gpt-4 | 0.518 | 201 | 183 | 131 |
| claude-sonnet-4.6 | qwen3-max | 0.518 | 176 | 167 | 117 |
| claude-opus-4.5-think | gpt-5.4-mini | 0.517 | 178 | 127 | 104 |
| gemini-2.5-flash-think | deepseek-v3 | 0.517 | 191 | 167 | 122 |
| gemini-2.5-flash-think | claude-sonnet-4.6 | 0.517 | 191 | 176 | 125 |
| glm-5-turbo-think | qwen3-max | 0.516 | 203 | 167 | 126 |
| kimi-k2.5-think | mimo-v2-flash | 0.516 | 230 | 143 | 127 |
| gpt-5-think | deepseek-r1-think | 0.516 | 244 | 185 | 146 |
| o3 | qwen3-max | 0.514 | 251 | 167 | 142 |
| o3 | deepseek-v3 | 0.514 | 251 | 167 | 142 |
| gemini-3.1-pro | glm-5-turbo-think | 0.514 | 377 | 203 | 197 |
| glm-5-think | grok-4.20-think | 0.514 | 219 | 155 | 127 |
| glm-4.7-think | deepseek-v3 | 0.514 | 207 | 167 | 127 |
| kimi-k2.5-think | glm-4.5-think | 0.514 | 230 | 162 | 133 |
| claude-opus-4.5 | mimo-v2-flash | 0.513 | 199 | 143 | 116 |
| glm-4.5-think | gpt-5.4-mini | 0.513 | 162 | 127 | 98 |
| grok-4 | grok-4.20-think | 0.513 | 252 | 155 | 138 |
| claude-opus-4.5-think | qwen3.6-plus-think | 0.513 | 178 | 114 | 99 |
| gemini-2.5-pro-think | glm-4.5-think | 0.513 | 260 | 162 | 143 |
| seed-2.0-lite-think | kimi-k2 | 0.512 | 186 | 183 | 125 |
| glm-5.1-think | grok-4.20-think | 0.512 | 217 | 155 | 126 |
| gemini-2.5-pro-think | kimi-k2 | 0.512 | 260 | 183 | 150 |
| gemini-3.1-pro | claude-opus-4.5 | 0.512 | 377 | 199 | 195 |
| llama-3.3-70b | mistral-large | 0.512 | 67 | 63 | 44 |
| gpt-5-think | qwen3-max | 0.511 | 244 | 167 | 139 |
| deepseek-r1-think | qwen3-max | 0.511 | 185 | 167 | 119 |
| qwen3-max | qwen-plus | 0.511 | 167 | 117 | 96 |
| seed-2.0-lite-think | claude-opus-4.5-think | 0.510 | 186 | 178 | 123 |
| glm-5-turbo-think | claude-sonnet-4.6 | 0.510 | 203 | 176 | 128 |
| qwen-plus | qwen3.6-plus-think | 0.510 | 117 | 114 | 78 |
| claude-opus-4.5 | gpt-5.4-mini | 0.509 | 199 | 127 | 110 |
| claude-opus-4.6 | grok-4.20 | 0.509 | 213 | 196 | 138 |
| claude-opus-4.5 | llama-4-maverick | 0.509 | 199 | 133 | 112 |
| o3 | claude-sonnet-4.6 | 0.509 | 251 | 176 | 144 |
| deepseek-v3.2 | mimo-v2-pro-think | 0.509 | 144 | 114 | 87 |
| gemini-2.5-flash-think | glm-4.5-think | 0.509 | 191 | 162 | 119 |
| gemini-2.5-pro-think | deepseek-r1-think | 0.508 | 260 | 185 | 150 |
| gemini-3-flash | grok-4.20 | 0.508 | 347 | 196 | 183 |
| seed-2.0-lite-think | claude-sonnet-4.6 | 0.508 | 186 | 176 | 122 |
| claude-opus-4.6 | grok-4.20-think | 0.508 | 213 | 155 | 124 |
| glm-4.6-think | deepseek-v3 | 0.508 | 201 | 167 | 124 |
| gpt-5.4 | deepseek-v3 | 0.508 | 201 | 167 | 124 |
| kimi-k2.5-think | deepseek-v3.2 | 0.508 | 230 | 144 | 126 |
| glm-4.6-think | claude-sonnet-4.6 | 0.508 | 201 | 176 | 127 |
| deepseek-v3 | gpt-5.4-mini | 0.508 | 167 | 127 | 99 |
| glm-5-think | gemini-2.5-flash-think | 0.507 | 219 | 191 | 138 |
| claude-sonnet-4.6 | llama-4-maverick | 0.507 | 176 | 133 | 104 |
| gemini-2.5-pro-think | glm-4.6-think | 0.507 | 260 | 201 | 155 |
| gpt-5.4-mini | mimo-v2-pro-think | 0.506 | 127 | 114 | 81 |
| glm-5-think | deepseek-v3.2 | 0.506 | 219 | 144 | 122 |
| grok-4.20 | claude-sonnet-4.6 | 0.506 | 196 | 176 | 125 |
| gpt-5.4 | seed-2.0-lite-think | 0.506 | 201 | 186 | 130 |
| gemini-3-flash-think | gemini-2.5-flash-think | 0.505 | 360 | 191 | 185 |
| grok-4 | glm-4.5-think | 0.505 | 252 | 162 | 139 |
| claude-opus-4.5-think | mimo-v2-pro-think | 0.505 | 178 | 114 | 98 |
| deepseek-r1-think | mimo-v2-flash | 0.505 | 185 | 143 | 110 |
| claude-sonnet-4.6 | grok-4.20-think | 0.505 | 176 | 155 | 111 |
| deepseek-v3.2-think | grok-4.20-think | 0.505 | 176 | 155 | 111 |
| deepseek-v3.2-think | qwen3-max | 0.504 | 176 | 167 | 115 |
| glm-5.1-think | deepseek-v3.2 | 0.504 | 217 | 144 | 121 |
| glm-5-turbo-think | deepseek-v3.2-think | 0.504 | 203 | 176 | 127 |
| grok-3 | deepseek-v3 | 0.504 | 260 | 167 | 143 |
| gemini-2.5-pro-think | deepseek-v3.2-think | 0.503 | 260 | 176 | 146 |
| deepseek-v3 | qwen3.6-plus-think | 0.503 | 167 | 114 | 94 |
| gpt-4.1-mini | gpt-4o | 0.503 | 145 | 145 | 97 |
| deepseek-v3.2-think | mimo-v2-flash-think | 0.502 | 176 | 132 | 103 |
| deepseek-r1-think | glm-4.5-think | 0.502 | 185 | 162 | 116 |
| gpt-4.1 | deepseek-v3.2-think | 0.502 | 246 | 176 | 141 |
| grok-4 | deepseek-v3.2-think | 0.502 | 252 | 176 | 143 |
| gemini-2.5-pro-think | seed-2.0-lite-think | 0.502 | 260 | 186 | 149 |
| gemini-3-flash | seed-2.0-lite-think | 0.501 | 347 | 186 | 178 |

Total pairs with Jaccard > 0.50: 422

### Cluster groups (connected components at Jaccard > 0.50)

**Cluster 1** (46 models, vendors: alibaba, anthropic, bytedance, deepseek, google, meta, moonshot, openai, stepfun, xai, xiaomi, zhipu):
  - gemini-3.1-pro (google, T5+T6=377)
  - gemini-3-flash-think (google, T5+T6=360)
  - gemini-3-flash (google, T5+T6=347)
  - grok-3 (xai, T5+T6=260)
  - gemini-2.5-pro-think (google, T5+T6=260)
  - grok-4 (xai, T5+T6=252)
  - o3 (openai, T5+T6=251)
  - gpt-4.1 (openai, T5+T6=246)
  - gpt-5-think (openai, T5+T6=244)
  - kimi-k2.5-think (moonshot, T5+T6=230)
  - glm-5-think (zhipu, T5+T6=219)
  - glm-5.1-think (zhipu, T5+T6=217)
  - claude-opus-4.6 (anthropic, T5+T6=213)
  - glm-4.7-think (zhipu, T5+T6=207)
  - glm-5-turbo-think (zhipu, T5+T6=203)
  - gpt-5.4 (openai, T5+T6=201)
  - glm-4.6-think (zhipu, T5+T6=201)
  - claude-opus-4.5 (anthropic, T5+T6=199)
  - grok-4.20 (xai, T5+T6=196)
  - gemini-2.5-flash-think (google, T5+T6=191)
  - seed-2.0-lite-think (bytedance, T5+T6=186)
  - deepseek-r1-think (deepseek, T5+T6=185)
  - kimi-k2 (moonshot, T5+T6=183)
  - gpt-4 (openai, T5+T6=183)
  - claude-opus-4.5-think (anthropic, T5+T6=178)
  - claude-sonnet-4.6 (anthropic, T5+T6=176)
  - deepseek-v3.2-think (deepseek, T5+T6=176)
  - qwen3-max (alibaba, T5+T6=167)
  - deepseek-v3 (deepseek, T5+T6=167)
  - glm-4.5-think (zhipu, T5+T6=162)
  - grok-4.20-think (xai, T5+T6=155)
  - step-3.5-flash-think (stepfun, T5+T6=148)
  - gpt-4o (openai, T5+T6=145)
  - gpt-4.1-mini (openai, T5+T6=145)
  - deepseek-v3.2 (deepseek, T5+T6=144)
  - mimo-v2-flash (xiaomi, T5+T6=143)
  - qwen3.5-plus-think (alibaba, T5+T6=142)
  - qwen3.5-397b-a17b-think (alibaba, T5+T6=137)
  - grok-4.1-fast-think (xai, T5+T6=135)
  - llama-4-maverick (meta, T5+T6=133)
  - mimo-v2-flash-think (xiaomi, T5+T6=132)
  - gpt-5.4-mini (openai, T5+T6=127)
  - qwen-plus (alibaba, T5+T6=117)
  - o4-mini-think (openai, T5+T6=115)
  - qwen3.6-plus-think (alibaba, T5+T6=114)
  - mimo-v2-pro-think (xiaomi, T5+T6=114)

**Cluster 2** (5 models, vendors: meta, mistral, nvidia):
  - llama-3-70b (meta, T5+T6=75)
  - llama-3.3-70b (meta, T5+T6=67)
  - mistral-large (mistral, T5+T6=63)
  - nemotron-70b (nvidia, T5+T6=50)
  - llama-3.1-70b (meta, T5+T6=45)

**Cluster 3** (2 models, vendors: alibaba):
  - qwen3.5-flash-think (alibaba, T5+T6=55)
  - qwen3.5-35b-a3b-think (alibaba, T5+T6=54)

## 2. Distillation Detection

### DeepSeek R1 Distilled Models

**deepseek-r1-distill-qwen-32b-think**
- T5 correct: 15, T6 correct: 7, T5+T6: 22

  vs **teacher (deepseek-r1-think)**: T5+T6=185
    - Intersection: 17
    - Jaccard: 0.089
    - Overlap coefficient: 0.773
    - % of distilled model's knowledge from teacher: 77.3%
    - % of teacher's knowledge in distilled model: 9.2%
    - Distilled-only probes: 5
    - Teacher-only probes: 168

  vs **base (qwen-2.5-72b)**: T5+T6=33
    - Intersection: 10
    - Jaccard: 0.222
    - Overlap coefficient: 0.455
    - % of distilled model's knowledge from base: 45.5%
    - % of base's knowledge in distilled model: 30.3%
    - Distilled-only probes: 12
    - Base-only probes: 23

  vs **base (qwq-32b-think)**: T5+T6=32
    - Intersection: 13
    - Jaccard: 0.317
    - Overlap coefficient: 0.591
    - % of distilled model's knowledge from base: 59.1%
    - % of base's knowledge in distilled model: 40.6%
    - Distilled-only probes: 9
    - Base-only probes: 19

  vs **base (qwen3-32b-think)**: T5+T6=23
    - Intersection: 6
    - Jaccard: 0.154
    - Overlap coefficient: 0.273
    - % of distilled model's knowledge from base: 27.3%
    - % of base's knowledge in distilled model: 26.1%
    - Distilled-only probes: 16
    - Base-only probes: 17

  **Verdict for deepseek-r1-distill-qwen-32b-think**:
    - Jaccard with teacher (deepseek-r1-think): 0.089
    - Jaccard with base (qwen-2.5-72b): 0.222
    - **Knowledge inherited primarily from BASE** (Jaccard ratio: 2.48x)

  **Verdict for deepseek-r1-distill-qwen-32b-think**:
    - Jaccard with teacher (deepseek-r1-think): 0.089
    - Jaccard with base (qwq-32b-think): 0.317
    - **Knowledge inherited primarily from BASE** (Jaccard ratio: 3.54x)

  **Verdict for deepseek-r1-distill-qwen-32b-think**:
    - Jaccard with teacher (deepseek-r1-think): 0.089
    - Jaccard with base (qwen3-32b-think): 0.154
    - **Knowledge inherited primarily from BASE** (Jaccard ratio: 1.72x)

**deepseek-r1-distill-llama-70b-think**
- T5 correct: 74, T6 correct: 26, T5+T6: 100

  vs **teacher (deepseek-r1-think)**: T5+T6=185
    - Intersection: 69
    - Jaccard: 0.319
    - Overlap coefficient: 0.690
    - % of distilled model's knowledge from teacher: 69.0%
    - % of teacher's knowledge in distilled model: 37.3%
    - Distilled-only probes: 31
    - Teacher-only probes: 116

  vs **base (llama-3.1-70b)**: T5+T6=45
    - Intersection: 25
    - Jaccard: 0.208
    - Overlap coefficient: 0.556
    - % of distilled model's knowledge from base: 25.0%
    - % of base's knowledge in distilled model: 55.6%
    - Distilled-only probes: 75
    - Base-only probes: 20

  vs **base (llama-3.3-70b)**: T5+T6=67
    - Intersection: 40
    - Jaccard: 0.315
    - Overlap coefficient: 0.597
    - % of distilled model's knowledge from base: 40.0%
    - % of base's knowledge in distilled model: 59.7%
    - Distilled-only probes: 60
    - Base-only probes: 27

  **Verdict for deepseek-r1-distill-llama-70b-think**:
    - Jaccard with teacher (deepseek-r1-think): 0.319
    - Jaccard with base (llama-3.1-70b): 0.208
    - **Knowledge inherited primarily from TEACHER** (Jaccard ratio: 1.53x)

  **Verdict for deepseek-r1-distill-llama-70b-think**:
    - Jaccard with teacher (deepseek-r1-think): 0.319
    - Jaccard with base (llama-3.3-70b): 0.315
    - **Knowledge inherited primarily from TEACHER** (Jaccard ratio: 1.01x)

### Nemotron Models (pruned/distilled from Llama)

**nemotron-super-49b-think** (T5+T6=38):
  vs llama-3.1-70b (T5+T6=45): Jaccard=0.078, intersection=6
  vs llama-3.3-70b (T5+T6=67): Jaccard=0.167, intersection=15

**nemotron-70b** (T5+T6=50):
  vs llama-3.1-70b (T5+T6=45): Jaccard=0.792, intersection=42
  vs llama-3.3-70b (T5+T6=67): Jaccard=0.603, intersection=44

**nemotron-ultra-253b** (T5+T6=0):
  vs llama-3.1-70b (T5+T6=45): Jaccard=0.000, intersection=0
  vs llama-3.3-70b (T5+T6=67): Jaccard=0.000, intersection=0

## 3. Family Knowledge Inheritance

For each model family, we examine whether smaller models' T5-T6 knowledge
is a strict subset of larger models' knowledge.

### Family: claude

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| claude-haiku-4.5-think | ? | 26 | 6 | 32 |
| claude-3.5-haiku | ? | 60 | 15 | 75 |
| claude-opus-4.6 | ? | 149 | 64 | 213 |
| claude-sonnet-4.6 | ? | 133 | 43 | 176 |
| claude-opus-4-think | ? | 73 | 25 | 98 |
| claude-opus-4.5-think | ? | 135 | 43 | 178 |
| claude-opus-4 | ? | 22 | 3 | 25 |
| claude-sonnet-4-think | ? | 69 | 15 | 84 |
| claude-opus-4.5 | ? | 144 | 55 | 199 |
| claude-haiku-4.5 | ? | 16 | 3 | 19 |
| claude-sonnet-4 | ? | 30 | 3 | 33 |
| claude-3-haiku | ? | 6 | 2 | 8 |

Pairwise inheritance analysis:

- **claude-haiku-4.5-think** (32) vs **claude-3.5-haiku** (75):
  - Shared: 23, Small-only: 9, Large-only: 52
  - 71.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.274
  - NOT strictly hierarchical: 9 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-opus-4.6** (213):
  - Shared: 29, Small-only: 3, Large-only: 184
  - 90.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.134
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-sonnet-4.6** (176):
  - Shared: 30, Small-only: 2, Large-only: 146
  - 93.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.169
  - NOT strictly hierarchical: 2 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-opus-4-think** (98):
  - Shared: 26, Small-only: 6, Large-only: 72
  - 81.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.250
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-opus-4.5-think** (178):
  - Shared: 29, Small-only: 3, Large-only: 149
  - 90.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.160
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-opus-4** (25):
  - Shared: 12, Small-only: 20, Large-only: 13
  - 37.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.267
  - NOT strictly hierarchical: 20 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-sonnet-4-think** (84):
  - Shared: 24, Small-only: 8, Large-only: 60
  - 75.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.261
  - NOT strictly hierarchical: 8 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-opus-4.5** (199):
  - Shared: 30, Small-only: 2, Large-only: 169
  - 93.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.149
  - NOT strictly hierarchical: 2 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-haiku-4.5** (19):
  - Shared: 15, Small-only: 17, Large-only: 4
  - 46.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.417
  - NOT strictly hierarchical: 17 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-sonnet-4** (33):
  - Shared: 20, Small-only: 12, Large-only: 13
  - 62.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.444
  - NOT strictly hierarchical: 12 probes known by smaller but not larger
- **claude-haiku-4.5-think** (32) vs **claude-3-haiku** (8):
  - Shared: 6, Small-only: 26, Large-only: 2
  - 18.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.176
  - NOT strictly hierarchical: 26 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-opus-4.6** (213):
  - Shared: 71, Small-only: 4, Large-only: 142
  - 94.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.327
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-sonnet-4.6** (176):
  - Shared: 68, Small-only: 7, Large-only: 108
  - 90.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.372
  - NOT strictly hierarchical: 7 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-opus-4-think** (98):
  - Shared: 47, Small-only: 28, Large-only: 51
  - 62.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.373
  - NOT strictly hierarchical: 28 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-opus-4.5-think** (178):
  - Shared: 69, Small-only: 6, Large-only: 109
  - 92.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.375
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-opus-4** (25):
  - Shared: 13, Small-only: 62, Large-only: 12
  - 17.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.149
  - NOT strictly hierarchical: 62 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-sonnet-4-think** (84):
  - Shared: 49, Small-only: 26, Large-only: 35
  - 65.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.445
  - NOT strictly hierarchical: 26 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-opus-4.5** (199):
  - Shared: 70, Small-only: 5, Large-only: 129
  - 93.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.343
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-haiku-4.5** (19):
  - Shared: 14, Small-only: 61, Large-only: 5
  - 18.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.175
  - NOT strictly hierarchical: 61 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-sonnet-4** (33):
  - Shared: 24, Small-only: 51, Large-only: 9
  - 32.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.286
  - NOT strictly hierarchical: 51 probes known by smaller but not larger
- **claude-3.5-haiku** (75) vs **claude-3-haiku** (8):
  - Shared: 5, Small-only: 70, Large-only: 3
  - 6.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.064
  - NOT strictly hierarchical: 70 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-sonnet-4.6** (176):
  - Shared: 154, Small-only: 59, Large-only: 22
  - 72.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.655
  - NOT strictly hierarchical: 59 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-opus-4-think** (98):
  - Shared: 89, Small-only: 124, Large-only: 9
  - 41.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.401
  - NOT strictly hierarchical: 124 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-opus-4.5-think** (178):
  - Shared: 166, Small-only: 47, Large-only: 12
  - 77.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.738
  - NOT strictly hierarchical: 47 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-opus-4** (25):
  - Shared: 25, Small-only: 188, Large-only: 0
  - 11.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.117
  - NOT strictly hierarchical: 188 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-sonnet-4-think** (84):
  - Shared: 81, Small-only: 132, Large-only: 3
  - 38.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.375
  - NOT strictly hierarchical: 132 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-opus-4.5** (199):
  - Shared: 174, Small-only: 39, Large-only: 25
  - 81.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.731
  - NOT strictly hierarchical: 39 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-haiku-4.5** (19):
  - Shared: 18, Small-only: 195, Large-only: 1
  - 8.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.084
  - NOT strictly hierarchical: 195 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-sonnet-4** (33):
  - Shared: 33, Small-only: 180, Large-only: 0
  - 15.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.155
  - NOT strictly hierarchical: 180 probes known by smaller but not larger
- **claude-opus-4.6** (213) vs **claude-3-haiku** (8):
  - Shared: 8, Small-only: 205, Large-only: 0
  - 3.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.038
  - NOT strictly hierarchical: 205 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-opus-4-think** (98):
  - Shared: 79, Small-only: 97, Large-only: 19
  - 44.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.405
  - NOT strictly hierarchical: 97 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-opus-4.5-think** (178):
  - Shared: 139, Small-only: 37, Large-only: 39
  - 79.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.647
  - NOT strictly hierarchical: 37 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-opus-4** (25):
  - Shared: 22, Small-only: 154, Large-only: 3
  - 12.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.123
  - NOT strictly hierarchical: 154 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-sonnet-4-think** (84):
  - Shared: 73, Small-only: 103, Large-only: 11
  - 41.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.390
  - NOT strictly hierarchical: 103 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-opus-4.5** (199):
  - Shared: 144, Small-only: 32, Large-only: 55
  - 81.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.623
  - NOT strictly hierarchical: 32 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-haiku-4.5** (19):
  - Shared: 17, Small-only: 159, Large-only: 2
  - 9.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.096
  - NOT strictly hierarchical: 159 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-sonnet-4** (33):
  - Shared: 32, Small-only: 144, Large-only: 1
  - 18.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.181
  - NOT strictly hierarchical: 144 probes known by smaller but not larger
- **claude-sonnet-4.6** (176) vs **claude-3-haiku** (8):
  - Shared: 8, Small-only: 168, Large-only: 0
  - 4.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.045
  - NOT strictly hierarchical: 168 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-opus-4.5-think** (178):
  - Shared: 84, Small-only: 14, Large-only: 94
  - 85.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.438
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-opus-4** (25):
  - Shared: 24, Small-only: 74, Large-only: 1
  - 24.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.242
  - NOT strictly hierarchical: 74 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-sonnet-4-think** (84):
  - Shared: 46, Small-only: 52, Large-only: 38
  - 46.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.338
  - NOT strictly hierarchical: 52 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-opus-4.5** (199):
  - Shared: 85, Small-only: 13, Large-only: 114
  - 86.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.401
  - NOT strictly hierarchical: 13 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-haiku-4.5** (19):
  - Shared: 17, Small-only: 81, Large-only: 2
  - 17.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.170
  - NOT strictly hierarchical: 81 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-sonnet-4** (33):
  - Shared: 30, Small-only: 68, Large-only: 3
  - 30.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.297
  - NOT strictly hierarchical: 68 probes known by smaller but not larger
- **claude-opus-4-think** (98) vs **claude-3-haiku** (8):
  - Shared: 6, Small-only: 92, Large-only: 2
  - 6.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.060
  - NOT strictly hierarchical: 92 probes known by smaller but not larger
- **claude-opus-4.5-think** (178) vs **claude-opus-4** (25):
  - Shared: 25, Small-only: 153, Large-only: 0
  - 14.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.140
  - NOT strictly hierarchical: 153 probes known by smaller but not larger
- **claude-opus-4.5-think** (178) vs **claude-sonnet-4-think** (84):
  - Shared: 79, Small-only: 99, Large-only: 5
  - 44.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.432
  - NOT strictly hierarchical: 99 probes known by smaller but not larger
- **claude-opus-4.5-think** (178) vs **claude-opus-4.5** (199):
  - Shared: 165, Small-only: 13, Large-only: 34
  - 92.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.778
  - NOT strictly hierarchical: 13 probes known by smaller but not larger
- **claude-opus-4.5-think** (178) vs **claude-haiku-4.5** (19):
  - Shared: 17, Small-only: 161, Large-only: 2
  - 9.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.094
  - NOT strictly hierarchical: 161 probes known by smaller but not larger
- **claude-opus-4.5-think** (178) vs **claude-sonnet-4** (33):
  - Shared: 32, Small-only: 146, Large-only: 1
  - 18.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.179
  - NOT strictly hierarchical: 146 probes known by smaller but not larger
- **claude-opus-4.5-think** (178) vs **claude-3-haiku** (8):
  - Shared: 7, Small-only: 171, Large-only: 1
  - 3.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.039
  - NOT strictly hierarchical: 171 probes known by smaller but not larger
- **claude-opus-4** (25) vs **claude-sonnet-4-think** (84):
  - Shared: 18, Small-only: 7, Large-only: 66
  - 72.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.198
  - NOT strictly hierarchical: 7 probes known by smaller but not larger
- **claude-opus-4** (25) vs **claude-opus-4.5** (199):
  - Shared: 25, Small-only: 0, Large-only: 174
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.126
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **claude-opus-4** (25) vs **claude-haiku-4.5** (19):
  - Shared: 10, Small-only: 15, Large-only: 9
  - 40.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.294
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **claude-opus-4** (25) vs **claude-sonnet-4** (33):
  - Shared: 19, Small-only: 6, Large-only: 14
  - 76.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.487
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **claude-opus-4** (25) vs **claude-3-haiku** (8):
  - Shared: 5, Small-only: 20, Large-only: 3
  - 20.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.179
  - NOT strictly hierarchical: 20 probes known by smaller but not larger
- **claude-sonnet-4-think** (84) vs **claude-opus-4.5** (199):
  - Shared: 80, Small-only: 4, Large-only: 119
  - 95.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.394
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **claude-sonnet-4-think** (84) vs **claude-haiku-4.5** (19):
  - Shared: 15, Small-only: 69, Large-only: 4
  - 17.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.170
  - NOT strictly hierarchical: 69 probes known by smaller but not larger
- **claude-sonnet-4-think** (84) vs **claude-sonnet-4** (33):
  - Shared: 29, Small-only: 55, Large-only: 4
  - 34.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.330
  - NOT strictly hierarchical: 55 probes known by smaller but not larger
- **claude-sonnet-4-think** (84) vs **claude-3-haiku** (8):
  - Shared: 5, Small-only: 79, Large-only: 3
  - 6.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.057
  - NOT strictly hierarchical: 79 probes known by smaller but not larger
- **claude-opus-4.5** (199) vs **claude-haiku-4.5** (19):
  - Shared: 17, Small-only: 182, Large-only: 2
  - 8.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.085
  - NOT strictly hierarchical: 182 probes known by smaller but not larger
- **claude-opus-4.5** (199) vs **claude-sonnet-4** (33):
  - Shared: 32, Small-only: 167, Large-only: 1
  - 16.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.160
  - NOT strictly hierarchical: 167 probes known by smaller but not larger
- **claude-opus-4.5** (199) vs **claude-3-haiku** (8):
  - Shared: 7, Small-only: 192, Large-only: 1
  - 3.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.035
  - NOT strictly hierarchical: 192 probes known by smaller but not larger
- **claude-haiku-4.5** (19) vs **claude-sonnet-4** (33):
  - Shared: 14, Small-only: 5, Large-only: 19
  - 73.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.368
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **claude-haiku-4.5** (19) vs **claude-3-haiku** (8):
  - Shared: 5, Small-only: 14, Large-only: 3
  - 26.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.227
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **claude-sonnet-4** (33) vs **claude-3-haiku** (8):
  - Shared: 6, Small-only: 27, Large-only: 2
  - 18.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.171
  - NOT strictly hierarchical: 27 probes known by smaller but not larger

### Family: command

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| command-r7b | 7B | 6 | 5 | 11 |
| command-r-plus | 104B | 25 | 7 | 32 |
| command-a | ? | 73 | 28 | 101 |

Pairwise inheritance analysis:

- **command-r7b** (11) vs **command-r-plus** (32):
  - Shared: 1, Small-only: 10, Large-only: 31
  - 9.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.024
  - NOT strictly hierarchical: 10 probes known by smaller but not larger
- **command-r7b** (11) vs **command-a** (101):
  - Shared: 2, Small-only: 9, Large-only: 99
  - 18.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.018
  - NOT strictly hierarchical: 9 probes known by smaller but not larger
- **command-r-plus** (32) vs **command-a** (101):
  - Shared: 25, Small-only: 7, Large-only: 76
  - 78.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.231
  - NOT strictly hierarchical: 7 probes known by smaller but not larger

### Family: deepseek

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| deepseek-v3.2 | 671B | 116 | 28 | 144 |
| deepseek-r1-think | 671B | 136 | 49 | 185 |
| deepseek-v3 | 671B | 121 | 46 | 167 |
| deepseek-v3.2-think | 671B | 129 | 47 | 176 |

Pairwise inheritance analysis:

- **deepseek-v3.2** (144) vs **deepseek-r1-think** (185):
  - Shared: 117, Small-only: 27, Large-only: 68
  - 81.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.552
  - NOT strictly hierarchical: 27 probes known by smaller but not larger
- **deepseek-v3.2** (144) vs **deepseek-v3** (167):
  - Shared: 120, Small-only: 24, Large-only: 47
  - 83.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.628
  - NOT strictly hierarchical: 24 probes known by smaller but not larger
- **deepseek-v3.2** (144) vs **deepseek-v3.2-think** (176):
  - Shared: 115, Small-only: 29, Large-only: 61
  - 79.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.561
  - NOT strictly hierarchical: 29 probes known by smaller but not larger
- **deepseek-r1-think** (185) vs **deepseek-v3** (167):
  - Shared: 131, Small-only: 54, Large-only: 36
  - 70.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.593
  - NOT strictly hierarchical: 54 probes known by smaller but not larger
- **deepseek-r1-think** (185) vs **deepseek-v3.2-think** (176):
  - Shared: 151, Small-only: 34, Large-only: 25
  - 81.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.719
  - NOT strictly hierarchical: 34 probes known by smaller but not larger
- **deepseek-v3** (167) vs **deepseek-v3.2-think** (176):
  - Shared: 129, Small-only: 38, Large-only: 47
  - 77.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.603
  - NOT strictly hierarchical: 38 probes known by smaller but not larger

### Family: deepseek-distill

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| deepseek-r1-distill-qwen-32b-think | 32B | 15 | 7 | 22 |
| deepseek-r1-distill-llama-70b-think | 70B | 74 | 26 | 100 |

Pairwise inheritance analysis:

- **deepseek-r1-distill-qwen-32b-think** (22) vs **deepseek-r1-distill-llama-70b-think** (100):
  - Shared: 15, Small-only: 7, Large-only: 85
  - 68.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.140
  - NOT strictly hierarchical: 7 probes known by smaller but not larger

### Family: gemini

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| gemini-2.5-flash-lite-think | ? | 74 | 20 | 94 |
| gemini-2.5-pro-think | ? | 163 | 97 | 260 |
| gemini-2.5-flash-think | ? | 130 | 61 | 191 |
| gemini-2.0-flash | ? | 85 | 47 | 132 |
| gemini-3-flash | ? | 194 | 153 | 347 |
| gemini-3-flash-think | ? | 196 | 164 | 360 |
| gemini-2.5-flash | ? | 69 | 33 | 102 |
| gemini-3.1-pro | ? | 193 | 184 | 377 |
| gemini-2.5-flash-lite | ? | 54 | 13 | 67 |

Pairwise inheritance analysis:

- **gemini-2.5-flash-lite-think** (94) vs **gemini-2.5-pro-think** (260):
  - Shared: 85, Small-only: 9, Large-only: 175
  - 90.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.316
  - NOT strictly hierarchical: 9 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-2.5-flash-think** (191):
  - Shared: 68, Small-only: 26, Large-only: 123
  - 72.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.313
  - NOT strictly hierarchical: 26 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-2.0-flash** (132):
  - Shared: 51, Small-only: 43, Large-only: 81
  - 54.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.291
  - NOT strictly hierarchical: 43 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-3-flash** (347):
  - Shared: 87, Small-only: 7, Large-only: 260
  - 92.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.246
  - NOT strictly hierarchical: 7 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-3-flash-think** (360):
  - Shared: 88, Small-only: 6, Large-only: 272
  - 93.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.240
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-2.5-flash** (102):
  - Shared: 46, Small-only: 48, Large-only: 56
  - 48.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.307
  - NOT strictly hierarchical: 48 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-3.1-pro** (377):
  - Shared: 90, Small-only: 4, Large-only: 287
  - 95.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.236
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **gemini-2.5-flash-lite-think** (94) vs **gemini-2.5-flash-lite** (67):
  - Shared: 47, Small-only: 47, Large-only: 20
  - 50.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.412
  - NOT strictly hierarchical: 47 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-2.5-flash-think** (191):
  - Shared: 169, Small-only: 91, Large-only: 22
  - 65.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.599
  - NOT strictly hierarchical: 91 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-2.0-flash** (132):
  - Shared: 117, Small-only: 143, Large-only: 15
  - 45.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.425
  - NOT strictly hierarchical: 143 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-3-flash** (347):
  - Shared: 241, Small-only: 19, Large-only: 106
  - 92.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.658
  - NOT strictly hierarchical: 19 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-3-flash-think** (360):
  - Shared: 245, Small-only: 15, Large-only: 115
  - 94.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.653
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-2.5-flash** (102):
  - Shared: 98, Small-only: 162, Large-only: 4
  - 37.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.371
  - NOT strictly hierarchical: 162 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-3.1-pro** (377):
  - Shared: 251, Small-only: 9, Large-only: 126
  - 96.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.650
  - NOT strictly hierarchical: 9 probes known by smaller but not larger
- **gemini-2.5-pro-think** (260) vs **gemini-2.5-flash-lite** (67):
  - Shared: 61, Small-only: 199, Large-only: 6
  - 23.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.229
  - NOT strictly hierarchical: 199 probes known by smaller but not larger
- **gemini-2.5-flash-think** (191) vs **gemini-2.0-flash** (132):
  - Shared: 104, Small-only: 87, Large-only: 28
  - 54.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.475
  - NOT strictly hierarchical: 87 probes known by smaller but not larger
- **gemini-2.5-flash-think** (191) vs **gemini-3-flash** (347):
  - Shared: 185, Small-only: 6, Large-only: 162
  - 96.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.524
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **gemini-2.5-flash-think** (191) vs **gemini-3-flash-think** (360):
  - Shared: 185, Small-only: 6, Large-only: 175
  - 96.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.505
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **gemini-2.5-flash-think** (191) vs **gemini-2.5-flash** (102):
  - Shared: 93, Small-only: 98, Large-only: 9
  - 48.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.465
  - NOT strictly hierarchical: 98 probes known by smaller but not larger
- **gemini-2.5-flash-think** (191) vs **gemini-3.1-pro** (377):
  - Shared: 188, Small-only: 3, Large-only: 189
  - 98.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.495
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **gemini-2.5-flash-think** (191) vs **gemini-2.5-flash-lite** (67):
  - Shared: 57, Small-only: 134, Large-only: 10
  - 29.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.284
  - NOT strictly hierarchical: 134 probes known by smaller but not larger
- **gemini-2.0-flash** (132) vs **gemini-3-flash** (347):
  - Shared: 127, Small-only: 5, Large-only: 220
  - 96.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.361
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **gemini-2.0-flash** (132) vs **gemini-3-flash-think** (360):
  - Shared: 125, Small-only: 7, Large-only: 235
  - 94.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.341
  - NOT strictly hierarchical: 7 probes known by smaller but not larger
- **gemini-2.0-flash** (132) vs **gemini-2.5-flash** (102):
  - Shared: 77, Small-only: 55, Large-only: 25
  - 58.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.490
  - NOT strictly hierarchical: 55 probes known by smaller but not larger
- **gemini-2.0-flash** (132) vs **gemini-3.1-pro** (377):
  - Shared: 132, Small-only: 0, Large-only: 245
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.350
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **gemini-2.0-flash** (132) vs **gemini-2.5-flash-lite** (67):
  - Shared: 48, Small-only: 84, Large-only: 19
  - 36.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.318
  - NOT strictly hierarchical: 84 probes known by smaller but not larger
- **gemini-3-flash** (347) vs **gemini-3-flash-think** (360):
  - Shared: 335, Small-only: 12, Large-only: 25
  - 96.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.901
  - NOT strictly hierarchical: 12 probes known by smaller but not larger
- **gemini-3-flash** (347) vs **gemini-2.5-flash** (102):
  - Shared: 99, Small-only: 248, Large-only: 3
  - 28.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.283
  - NOT strictly hierarchical: 248 probes known by smaller but not larger
- **gemini-3-flash** (347) vs **gemini-3.1-pro** (377):
  - Shared: 333, Small-only: 14, Large-only: 44
  - 96.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.852
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **gemini-3-flash** (347) vs **gemini-2.5-flash-lite** (67):
  - Shared: 64, Small-only: 283, Large-only: 3
  - 18.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.183
  - NOT strictly hierarchical: 283 probes known by smaller but not larger
- **gemini-3-flash-think** (360) vs **gemini-2.5-flash** (102):
  - Shared: 97, Small-only: 263, Large-only: 5
  - 26.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.266
  - NOT strictly hierarchical: 263 probes known by smaller but not larger
- **gemini-3-flash-think** (360) vs **gemini-3.1-pro** (377):
  - Shared: 345, Small-only: 15, Large-only: 32
  - 95.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.880
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **gemini-3-flash-think** (360) vs **gemini-2.5-flash-lite** (67):
  - Shared: 64, Small-only: 296, Large-only: 3
  - 17.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.176
  - NOT strictly hierarchical: 296 probes known by smaller but not larger
- **gemini-2.5-flash** (102) vs **gemini-3.1-pro** (377):
  - Shared: 102, Small-only: 0, Large-only: 275
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.271
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **gemini-2.5-flash** (102) vs **gemini-2.5-flash-lite** (67):
  - Shared: 39, Small-only: 63, Large-only: 28
  - 38.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.300
  - NOT strictly hierarchical: 63 probes known by smaller but not larger
- **gemini-3.1-pro** (377) vs **gemini-2.5-flash-lite** (67):
  - Shared: 67, Small-only: 310, Large-only: 0
  - 17.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.178
  - NOT strictly hierarchical: 310 probes known by smaller but not larger

### Family: gemma3

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| gemma-3-1b | 1B | 4 | 1 | 5 |
| gemma-3-4b | 4B | 13 | 5 | 18 |
| gemma-3-12b | 12B | 14 | 9 | 23 |
| gemma-3-27b | 27B | 29 | 12 | 41 |

Pairwise inheritance analysis:

- **gemma-3-1b** (5) vs **gemma-3-4b** (18):
  - Shared: 0, Small-only: 5, Large-only: 18
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **gemma-3-1b** (5) vs **gemma-3-12b** (23):
  - Shared: 1, Small-only: 4, Large-only: 22
  - 20.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.037
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **gemma-3-1b** (5) vs **gemma-3-27b** (41):
  - Shared: 1, Small-only: 4, Large-only: 40
  - 20.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.022
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **gemma-3-4b** (18) vs **gemma-3-12b** (23):
  - Shared: 7, Small-only: 11, Large-only: 16
  - 38.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.206
  - NOT strictly hierarchical: 11 probes known by smaller but not larger
- **gemma-3-4b** (18) vs **gemma-3-27b** (41):
  - Shared: 3, Small-only: 15, Large-only: 38
  - 16.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.054
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **gemma-3-12b** (23) vs **gemma-3-27b** (41):
  - Shared: 2, Small-only: 21, Large-only: 39
  - 8.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.032
  - NOT strictly hierarchical: 21 probes known by smaller but not larger

### Family: gemma4

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| gemma-4-26b-a4b | 26B | 20 | 13 | 33 |
| gemma-4-31b | 31B | 26 | 16 | 42 |

Pairwise inheritance analysis:

- **gemma-4-26b-a4b** (33) vs **gemma-4-31b** (42):
  - Shared: 7, Small-only: 26, Large-only: 35
  - 21.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.103
  - NOT strictly hierarchical: 26 probes known by smaller but not larger

### Family: glm

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| glm-4-32b | 32B | 35 | 10 | 45 |
| glm-4.7-think | ? | 139 | 68 | 207 |
| glm-4.5-air-think | ? | 66 | 20 | 86 |
| glm-4.5-think | ? | 120 | 42 | 162 |
| glm-5-turbo-think | ? | 143 | 60 | 203 |
| glm-4.6-think | ? | 141 | 60 | 201 |
| glm-4.7-flash-think | ? | 42 | 10 | 52 |
| glm-5.1-think | ? | 156 | 61 | 217 |
| glm-5-think | ? | 154 | 65 | 219 |

Pairwise inheritance analysis:

- **glm-4-32b** (45) vs **glm-4.7-think** (207):
  - Shared: 38, Small-only: 7, Large-only: 169
  - 84.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.178
  - NOT strictly hierarchical: 7 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-4.5-air-think** (86):
  - Shared: 29, Small-only: 16, Large-only: 57
  - 64.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.284
  - NOT strictly hierarchical: 16 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-4.5-think** (162):
  - Shared: 34, Small-only: 11, Large-only: 128
  - 75.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.197
  - NOT strictly hierarchical: 11 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-5-turbo-think** (203):
  - Shared: 39, Small-only: 6, Large-only: 164
  - 86.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.187
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-4.6-think** (201):
  - Shared: 37, Small-only: 8, Large-only: 164
  - 82.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.177
  - NOT strictly hierarchical: 8 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-4.7-flash-think** (52):
  - Shared: 16, Small-only: 29, Large-only: 36
  - 35.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.198
  - NOT strictly hierarchical: 29 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-5.1-think** (217):
  - Shared: 34, Small-only: 11, Large-only: 183
  - 75.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.149
  - NOT strictly hierarchical: 11 probes known by smaller but not larger
- **glm-4-32b** (45) vs **glm-5-think** (219):
  - Shared: 37, Small-only: 8, Large-only: 182
  - 82.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.163
  - NOT strictly hierarchical: 8 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-4.5-air-think** (86):
  - Shared: 67, Small-only: 140, Large-only: 19
  - 32.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.296
  - NOT strictly hierarchical: 140 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-4.5-think** (162):
  - Shared: 141, Small-only: 66, Large-only: 21
  - 68.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.618
  - NOT strictly hierarchical: 66 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-5-turbo-think** (203):
  - Shared: 159, Small-only: 48, Large-only: 44
  - 76.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.633
  - NOT strictly hierarchical: 48 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-4.6-think** (201):
  - Shared: 166, Small-only: 41, Large-only: 35
  - 80.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.686
  - NOT strictly hierarchical: 41 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-4.7-flash-think** (52):
  - Shared: 41, Small-only: 166, Large-only: 11
  - 19.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.188
  - NOT strictly hierarchical: 166 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-5.1-think** (217):
  - Shared: 153, Small-only: 54, Large-only: 64
  - 73.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.565
  - NOT strictly hierarchical: 54 probes known by smaller but not larger
- **glm-4.7-think** (207) vs **glm-5-think** (219):
  - Shared: 150, Small-only: 57, Large-only: 69
  - 72.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.543
  - NOT strictly hierarchical: 57 probes known by smaller but not larger
- **glm-4.5-air-think** (86) vs **glm-4.5-think** (162):
  - Shared: 62, Small-only: 24, Large-only: 100
  - 72.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.333
  - NOT strictly hierarchical: 24 probes known by smaller but not larger
- **glm-4.5-air-think** (86) vs **glm-5-turbo-think** (203):
  - Shared: 67, Small-only: 19, Large-only: 136
  - 77.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.302
  - NOT strictly hierarchical: 19 probes known by smaller but not larger
- **glm-4.5-air-think** (86) vs **glm-4.6-think** (201):
  - Shared: 68, Small-only: 18, Large-only: 133
  - 79.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.311
  - NOT strictly hierarchical: 18 probes known by smaller but not larger
- **glm-4.5-air-think** (86) vs **glm-4.7-flash-think** (52):
  - Shared: 29, Small-only: 57, Large-only: 23
  - 33.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.266
  - NOT strictly hierarchical: 57 probes known by smaller but not larger
- **glm-4.5-air-think** (86) vs **glm-5.1-think** (217):
  - Shared: 68, Small-only: 18, Large-only: 149
  - 79.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.289
  - NOT strictly hierarchical: 18 probes known by smaller but not larger
- **glm-4.5-air-think** (86) vs **glm-5-think** (219):
  - Shared: 72, Small-only: 14, Large-only: 147
  - 83.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.309
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **glm-4.5-think** (162) vs **glm-5-turbo-think** (203):
  - Shared: 137, Small-only: 25, Large-only: 66
  - 84.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.601
  - NOT strictly hierarchical: 25 probes known by smaller but not larger
- **glm-4.5-think** (162) vs **glm-4.6-think** (201):
  - Shared: 139, Small-only: 23, Large-only: 62
  - 85.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.621
  - NOT strictly hierarchical: 23 probes known by smaller but not larger
- **glm-4.5-think** (162) vs **glm-4.7-flash-think** (52):
  - Shared: 38, Small-only: 124, Large-only: 14
  - 23.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.216
  - NOT strictly hierarchical: 124 probes known by smaller but not larger
- **glm-4.5-think** (162) vs **glm-5.1-think** (217):
  - Shared: 134, Small-only: 28, Large-only: 83
  - 82.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.547
  - NOT strictly hierarchical: 28 probes known by smaller but not larger
- **glm-4.5-think** (162) vs **glm-5-think** (219):
  - Shared: 134, Small-only: 28, Large-only: 85
  - 82.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.543
  - NOT strictly hierarchical: 28 probes known by smaller but not larger
- **glm-5-turbo-think** (203) vs **glm-4.6-think** (201):
  - Shared: 159, Small-only: 44, Large-only: 42
  - 78.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.649
  - NOT strictly hierarchical: 44 probes known by smaller but not larger
- **glm-5-turbo-think** (203) vs **glm-4.7-flash-think** (52):
  - Shared: 42, Small-only: 161, Large-only: 10
  - 20.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.197
  - NOT strictly hierarchical: 161 probes known by smaller but not larger
- **glm-5-turbo-think** (203) vs **glm-5.1-think** (217):
  - Shared: 152, Small-only: 51, Large-only: 65
  - 74.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.567
  - NOT strictly hierarchical: 51 probes known by smaller but not larger
- **glm-5-turbo-think** (203) vs **glm-5-think** (219):
  - Shared: 159, Small-only: 44, Large-only: 60
  - 78.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.605
  - NOT strictly hierarchical: 44 probes known by smaller but not larger
- **glm-4.6-think** (201) vs **glm-4.7-flash-think** (52):
  - Shared: 42, Small-only: 159, Large-only: 10
  - 20.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.199
  - NOT strictly hierarchical: 159 probes known by smaller but not larger
- **glm-4.6-think** (201) vs **glm-5.1-think** (217):
  - Shared: 152, Small-only: 49, Large-only: 65
  - 75.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.571
  - NOT strictly hierarchical: 49 probes known by smaller but not larger
- **glm-4.6-think** (201) vs **glm-5-think** (219):
  - Shared: 156, Small-only: 45, Large-only: 63
  - 77.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.591
  - NOT strictly hierarchical: 45 probes known by smaller but not larger
- **glm-4.7-flash-think** (52) vs **glm-5.1-think** (217):
  - Shared: 42, Small-only: 10, Large-only: 175
  - 80.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.185
  - NOT strictly hierarchical: 10 probes known by smaller but not larger
- **glm-4.7-flash-think** (52) vs **glm-5-think** (219):
  - Shared: 41, Small-only: 11, Large-only: 178
  - 78.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.178
  - NOT strictly hierarchical: 11 probes known by smaller but not larger
- **glm-5.1-think** (217) vs **glm-5-think** (219):
  - Shared: 178, Small-only: 39, Large-only: 41
  - 82.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.690
  - NOT strictly hierarchical: 39 probes known by smaller but not larger

### Family: gpt

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| gpt-4 | 1800B | 130 | 53 | 183 |
| gpt-4.1-mini | ? | 103 | 42 | 145 |
| o3-mini | ? | 20 | 7 | 27 |
| gpt-4o-mini | ? | 48 | 10 | 58 |
| gpt-4.1-nano | ? | 54 | 15 | 69 |
| gpt-4o | ? | 103 | 42 | 145 |
| gpt-3.5-turbo | ? | 86 | 21 | 107 |
| o4-mini-think | ? | 79 | 36 | 115 |
| o3 | ? | 159 | 92 | 251 |
| gpt-4.1 | ? | 158 | 88 | 246 |

Pairwise inheritance analysis:

- **gpt-4** (183) vs **gpt-4.1-mini** (145):
  - Shared: 112, Small-only: 71, Large-only: 33
  - 61.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.519
  - NOT strictly hierarchical: 71 probes known by smaller but not larger
- **gpt-4** (183) vs **o3-mini** (27):
  - Shared: 23, Small-only: 160, Large-only: 4
  - 12.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.123
  - NOT strictly hierarchical: 160 probes known by smaller but not larger
- **gpt-4** (183) vs **gpt-4o-mini** (58):
  - Shared: 48, Small-only: 135, Large-only: 10
  - 26.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.249
  - NOT strictly hierarchical: 135 probes known by smaller but not larger
- **gpt-4** (183) vs **gpt-4.1-nano** (69):
  - Shared: 53, Small-only: 130, Large-only: 16
  - 29.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.266
  - NOT strictly hierarchical: 130 probes known by smaller but not larger
- **gpt-4** (183) vs **gpt-4o** (145):
  - Shared: 117, Small-only: 66, Large-only: 28
  - 63.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.555
  - NOT strictly hierarchical: 66 probes known by smaller but not larger
- **gpt-4** (183) vs **gpt-3.5-turbo** (107):
  - Shared: 88, Small-only: 95, Large-only: 19
  - 48.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.436
  - NOT strictly hierarchical: 95 probes known by smaller but not larger
- **gpt-4** (183) vs **o4-mini-think** (115):
  - Shared: 90, Small-only: 93, Large-only: 25
  - 49.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.433
  - NOT strictly hierarchical: 93 probes known by smaller but not larger
- **gpt-4** (183) vs **o3** (251):
  - Shared: 162, Small-only: 21, Large-only: 89
  - 88.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.596
  - NOT strictly hierarchical: 21 probes known by smaller but not larger
- **gpt-4** (183) vs **gpt-4.1** (246):
  - Shared: 166, Small-only: 17, Large-only: 80
  - 90.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.631
  - NOT strictly hierarchical: 17 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **o3-mini** (27):
  - Shared: 23, Small-only: 122, Large-only: 4
  - 15.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.154
  - NOT strictly hierarchical: 122 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **gpt-4o-mini** (58):
  - Shared: 46, Small-only: 99, Large-only: 12
  - 31.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.293
  - NOT strictly hierarchical: 99 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **gpt-4.1-nano** (69):
  - Shared: 46, Small-only: 99, Large-only: 23
  - 31.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.274
  - NOT strictly hierarchical: 99 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **gpt-4o** (145):
  - Shared: 97, Small-only: 48, Large-only: 48
  - 66.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.503
  - NOT strictly hierarchical: 48 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **gpt-3.5-turbo** (107):
  - Shared: 77, Small-only: 68, Large-only: 30
  - 53.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.440
  - NOT strictly hierarchical: 68 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **o4-mini-think** (115):
  - Shared: 100, Small-only: 45, Large-only: 15
  - 69.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.625
  - NOT strictly hierarchical: 45 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **o3** (251):
  - Shared: 128, Small-only: 17, Large-only: 123
  - 88.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.478
  - NOT strictly hierarchical: 17 probes known by smaller but not larger
- **gpt-4.1-mini** (145) vs **gpt-4.1** (246):
  - Shared: 130, Small-only: 15, Large-only: 116
  - 89.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.498
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **o3-mini** (27) vs **gpt-4o-mini** (58):
  - Shared: 17, Small-only: 10, Large-only: 41
  - 63.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.250
  - NOT strictly hierarchical: 10 probes known by smaller but not larger
- **o3-mini** (27) vs **gpt-4.1-nano** (69):
  - Shared: 15, Small-only: 12, Large-only: 54
  - 55.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.185
  - NOT strictly hierarchical: 12 probes known by smaller but not larger
- **o3-mini** (27) vs **gpt-4o** (145):
  - Shared: 22, Small-only: 5, Large-only: 123
  - 81.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.147
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **o3-mini** (27) vs **gpt-3.5-turbo** (107):
  - Shared: 22, Small-only: 5, Large-only: 85
  - 81.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.196
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **o3-mini** (27) vs **o4-mini-think** (115):
  - Shared: 24, Small-only: 3, Large-only: 91
  - 88.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.203
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **o3-mini** (27) vs **o3** (251):
  - Shared: 25, Small-only: 2, Large-only: 226
  - 92.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.099
  - NOT strictly hierarchical: 2 probes known by smaller but not larger
- **o3-mini** (27) vs **gpt-4.1** (246):
  - Shared: 26, Small-only: 1, Large-only: 220
  - 96.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.105
  - NOT strictly hierarchical: 1 probes known by smaller but not larger
- **gpt-4o-mini** (58) vs **gpt-4.1-nano** (69):
  - Shared: 27, Small-only: 31, Large-only: 42
  - 46.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.270
  - NOT strictly hierarchical: 31 probes known by smaller but not larger
- **gpt-4o-mini** (58) vs **gpt-4o** (145):
  - Shared: 44, Small-only: 14, Large-only: 101
  - 75.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.277
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **gpt-4o-mini** (58) vs **gpt-3.5-turbo** (107):
  - Shared: 46, Small-only: 12, Large-only: 61
  - 79.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.387
  - NOT strictly hierarchical: 12 probes known by smaller but not larger
- **gpt-4o-mini** (58) vs **o4-mini-think** (115):
  - Shared: 40, Small-only: 18, Large-only: 75
  - 69.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.301
  - NOT strictly hierarchical: 18 probes known by smaller but not larger
- **gpt-4o-mini** (58) vs **o3** (251):
  - Shared: 52, Small-only: 6, Large-only: 199
  - 89.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.202
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **gpt-4o-mini** (58) vs **gpt-4.1** (246):
  - Shared: 55, Small-only: 3, Large-only: 191
  - 94.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.221
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **gpt-4.1-nano** (69) vs **gpt-4o** (145):
  - Shared: 38, Small-only: 31, Large-only: 107
  - 55.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.216
  - NOT strictly hierarchical: 31 probes known by smaller but not larger
- **gpt-4.1-nano** (69) vs **gpt-3.5-turbo** (107):
  - Shared: 40, Small-only: 29, Large-only: 67
  - 58.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.294
  - NOT strictly hierarchical: 29 probes known by smaller but not larger
- **gpt-4.1-nano** (69) vs **o4-mini-think** (115):
  - Shared: 39, Small-only: 30, Large-only: 76
  - 56.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.269
  - NOT strictly hierarchical: 30 probes known by smaller but not larger
- **gpt-4.1-nano** (69) vs **o3** (251):
  - Shared: 65, Small-only: 4, Large-only: 186
  - 94.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.255
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **gpt-4.1-nano** (69) vs **gpt-4.1** (246):
  - Shared: 63, Small-only: 6, Large-only: 183
  - 91.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.250
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **gpt-4o** (145) vs **gpt-3.5-turbo** (107):
  - Shared: 75, Small-only: 70, Large-only: 32
  - 51.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.424
  - NOT strictly hierarchical: 70 probes known by smaller but not larger
- **gpt-4o** (145) vs **o4-mini-think** (115):
  - Shared: 78, Small-only: 67, Large-only: 37
  - 53.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.429
  - NOT strictly hierarchical: 67 probes known by smaller but not larger
- **gpt-4o** (145) vs **o3** (251):
  - Shared: 131, Small-only: 14, Large-only: 120
  - 90.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.494
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **gpt-4o** (145) vs **gpt-4.1** (246):
  - Shared: 135, Small-only: 10, Large-only: 111
  - 93.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.527
  - NOT strictly hierarchical: 10 probes known by smaller but not larger
- **gpt-3.5-turbo** (107) vs **o4-mini-think** (115):
  - Shared: 66, Small-only: 41, Large-only: 49
  - 61.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.423
  - NOT strictly hierarchical: 41 probes known by smaller but not larger
- **gpt-3.5-turbo** (107) vs **o3** (251):
  - Shared: 97, Small-only: 10, Large-only: 154
  - 90.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.372
  - NOT strictly hierarchical: 10 probes known by smaller but not larger
- **gpt-3.5-turbo** (107) vs **gpt-4.1** (246):
  - Shared: 101, Small-only: 6, Large-only: 145
  - 94.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.401
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **o4-mini-think** (115) vs **o3** (251):
  - Shared: 101, Small-only: 14, Large-only: 150
  - 87.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.381
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **o4-mini-think** (115) vs **gpt-4.1** (246):
  - Shared: 105, Small-only: 10, Large-only: 141
  - 91.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.410
  - NOT strictly hierarchical: 10 probes known by smaller but not larger
- **o3** (251) vs **gpt-4.1** (246):
  - Shared: 218, Small-only: 33, Large-only: 28
  - 86.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.781
  - NOT strictly hierarchical: 33 probes known by smaller but not larger

### Family: gpt-oss

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| gpt-oss-20b-think | 20B | 15 | 15 | 30 |
| gpt-oss-120b-think | 120B | 59 | 23 | 82 |

Pairwise inheritance analysis:

- **gpt-oss-20b-think** (30) vs **gpt-oss-120b-think** (82):
  - Shared: 12, Small-only: 18, Large-only: 70
  - 40.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.120
  - NOT strictly hierarchical: 18 probes known by smaller but not larger

### Family: gpt5

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| gpt-5-nano-think | ? | 9 | 1 | 10 |
| gpt-5.4-mini | ? | 101 | 26 | 127 |
| gpt-5-mini-think | ? | 47 | 11 | 58 |
| gpt-5-nano | ? | 8 | 0 | 8 |
| gpt-5-think | ? | 158 | 86 | 244 |
| gpt-5.4 | ? | 140 | 61 | 201 |

Pairwise inheritance analysis:

- **gpt-5-nano-think** (10) vs **gpt-5.4-mini** (127):
  - Shared: 8, Small-only: 2, Large-only: 119
  - 80.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.062
  - NOT strictly hierarchical: 2 probes known by smaller but not larger
- **gpt-5-nano-think** (10) vs **gpt-5-mini-think** (58):
  - Shared: 8, Small-only: 2, Large-only: 50
  - 80.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.133
  - NOT strictly hierarchical: 2 probes known by smaller but not larger
- **gpt-5-nano-think** (10) vs **gpt-5-nano** (8):
  - Shared: 5, Small-only: 5, Large-only: 3
  - 50.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.385
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **gpt-5-nano-think** (10) vs **gpt-5-think** (244):
  - Shared: 10, Small-only: 0, Large-only: 234
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.041
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **gpt-5-nano-think** (10) vs **gpt-5.4** (201):
  - Shared: 10, Small-only: 0, Large-only: 191
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.050
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **gpt-5.4-mini** (127) vs **gpt-5-mini-think** (58):
  - Shared: 42, Small-only: 85, Large-only: 16
  - 33.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.294
  - NOT strictly hierarchical: 85 probes known by smaller but not larger
- **gpt-5.4-mini** (127) vs **gpt-5-nano** (8):
  - Shared: 7, Small-only: 120, Large-only: 1
  - 5.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.055
  - NOT strictly hierarchical: 120 probes known by smaller but not larger
- **gpt-5.4-mini** (127) vs **gpt-5-think** (244):
  - Shared: 119, Small-only: 8, Large-only: 125
  - 93.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.472
  - NOT strictly hierarchical: 8 probes known by smaller but not larger
- **gpt-5.4-mini** (127) vs **gpt-5.4** (201):
  - Shared: 109, Small-only: 18, Large-only: 92
  - 85.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.498
  - NOT strictly hierarchical: 18 probes known by smaller but not larger
- **gpt-5-mini-think** (58) vs **gpt-5-nano** (8):
  - Shared: 4, Small-only: 54, Large-only: 4
  - 6.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.065
  - NOT strictly hierarchical: 54 probes known by smaller but not larger
- **gpt-5-mini-think** (58) vs **gpt-5-think** (244):
  - Shared: 56, Small-only: 2, Large-only: 188
  - 96.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.228
  - NOT strictly hierarchical: 2 probes known by smaller but not larger
- **gpt-5-mini-think** (58) vs **gpt-5.4** (201):
  - Shared: 52, Small-only: 6, Large-only: 149
  - 89.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.251
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **gpt-5-nano** (8) vs **gpt-5-think** (244):
  - Shared: 8, Small-only: 0, Large-only: 236
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.033
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **gpt-5-nano** (8) vs **gpt-5.4** (201):
  - Shared: 8, Small-only: 0, Large-only: 193
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.040
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **gpt-5-think** (244) vs **gpt-5.4** (201):
  - Shared: 169, Small-only: 75, Large-only: 32
  - 69.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.612
  - NOT strictly hierarchical: 75 probes known by smaller but not larger

### Family: grok

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| grok-3 | ? | 170 | 90 | 260 |
| grok-4.20 | ? | 136 | 60 | 196 |
| grok-4 | ? | 173 | 79 | 252 |
| grok-4.20-think | ? | 120 | 35 | 155 |
| grok-4.1-fast-think | ? | 104 | 31 | 135 |
| grok-3-mini-think | ? | 86 | 22 | 108 |

Pairwise inheritance analysis:

- **grok-3** (260) vs **grok-4.20** (196):
  - Shared: 165, Small-only: 95, Large-only: 31
  - 63.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.567
  - NOT strictly hierarchical: 95 probes known by smaller but not larger
- **grok-3** (260) vs **grok-4** (252):
  - Shared: 223, Small-only: 37, Large-only: 29
  - 85.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.772
  - NOT strictly hierarchical: 37 probes known by smaller but not larger
- **grok-3** (260) vs **grok-4.20-think** (155):
  - Shared: 143, Small-only: 117, Large-only: 12
  - 55.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.526
  - NOT strictly hierarchical: 117 probes known by smaller but not larger
- **grok-3** (260) vs **grok-4.1-fast-think** (135):
  - Shared: 126, Small-only: 134, Large-only: 9
  - 48.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.468
  - NOT strictly hierarchical: 134 probes known by smaller but not larger
- **grok-3** (260) vs **grok-3-mini-think** (108):
  - Shared: 101, Small-only: 159, Large-only: 7
  - 38.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.378
  - NOT strictly hierarchical: 159 probes known by smaller but not larger
- **grok-4.20** (196) vs **grok-4** (252):
  - Shared: 157, Small-only: 39, Large-only: 95
  - 80.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.540
  - NOT strictly hierarchical: 39 probes known by smaller but not larger
- **grok-4.20** (196) vs **grok-4.20-think** (155):
  - Shared: 124, Small-only: 72, Large-only: 31
  - 63.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.546
  - NOT strictly hierarchical: 72 probes known by smaller but not larger
- **grok-4.20** (196) vs **grok-4.1-fast-think** (135):
  - Shared: 110, Small-only: 86, Large-only: 25
  - 56.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.498
  - NOT strictly hierarchical: 86 probes known by smaller but not larger
- **grok-4.20** (196) vs **grok-3-mini-think** (108):
  - Shared: 76, Small-only: 120, Large-only: 32
  - 38.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.333
  - NOT strictly hierarchical: 120 probes known by smaller but not larger
- **grok-4** (252) vs **grok-4.20-think** (155):
  - Shared: 138, Small-only: 114, Large-only: 17
  - 54.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.513
  - NOT strictly hierarchical: 114 probes known by smaller but not larger
- **grok-4** (252) vs **grok-4.1-fast-think** (135):
  - Shared: 124, Small-only: 128, Large-only: 11
  - 49.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.471
  - NOT strictly hierarchical: 128 probes known by smaller but not larger
- **grok-4** (252) vs **grok-3-mini-think** (108):
  - Shared: 98, Small-only: 154, Large-only: 10
  - 38.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.374
  - NOT strictly hierarchical: 154 probes known by smaller but not larger
- **grok-4.20-think** (155) vs **grok-4.1-fast-think** (135):
  - Shared: 100, Small-only: 55, Large-only: 35
  - 64.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.526
  - NOT strictly hierarchical: 55 probes known by smaller but not larger
- **grok-4.20-think** (155) vs **grok-3-mini-think** (108):
  - Shared: 84, Small-only: 71, Large-only: 24
  - 54.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.469
  - NOT strictly hierarchical: 71 probes known by smaller but not larger
- **grok-4.1-fast-think** (135) vs **grok-3-mini-think** (108):
  - Shared: 79, Small-only: 56, Large-only: 29
  - 58.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.482
  - NOT strictly hierarchical: 56 probes known by smaller but not larger

### Family: hunyuan

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| hunyuan-a13b | ? | 4 | 7 | 11 |
| hunyuan-a13b-think | ? | 13 | 10 | 23 |

Pairwise inheritance analysis:

- **hunyuan-a13b** (11) vs **hunyuan-a13b-think** (23):
  - Shared: 5, Small-only: 6, Large-only: 18
  - 45.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.172
  - NOT strictly hierarchical: 6 probes known by smaller but not larger

### Family: kimi

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| kimi-k2 | 1000B | 139 | 44 | 183 |
| kimi-k2.5-think | ? | 186 | 44 | 230 |

Pairwise inheritance analysis:

- **kimi-k2** (183) vs **kimi-k2.5-think** (230):
  - Shared: 154, Small-only: 29, Large-only: 76
  - 84.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.595
  - NOT strictly hierarchical: 29 probes known by smaller but not larger

### Family: llama

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| llama-3.2-1b | 1.24B | 0 | 0 | 0 |
| llama-3.2-3b | 3.21B | 3 | 0 | 3 |
| llama-3.1-8b | 8.03B | 6 | 1 | 7 |
| llama-3.3-70b | 70.6B | 54 | 13 | 67 |
| llama-3.1-70b | 70.6B | 39 | 6 | 45 |
| hermes-3-405b | 405B | 11 | 3 | 14 |

Pairwise inheritance analysis:

- **llama-3.2-1b** (0) vs **llama-3.2-3b** (3):
  - Shared: 0, Small-only: 0, Large-only: 3
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-1b** (0) vs **llama-3.1-8b** (7):
  - Shared: 0, Small-only: 0, Large-only: 7
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-1b** (0) vs **llama-3.3-70b** (67):
  - Shared: 0, Small-only: 0, Large-only: 67
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-1b** (0) vs **llama-3.1-70b** (45):
  - Shared: 0, Small-only: 0, Large-only: 45
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-1b** (0) vs **hermes-3-405b** (14):
  - Shared: 0, Small-only: 0, Large-only: 14
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-3b** (3) vs **llama-3.1-8b** (7):
  - Shared: 2, Small-only: 1, Large-only: 5
  - 66.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.250
  - NOT strictly hierarchical: 1 probes known by smaller but not larger
- **llama-3.2-3b** (3) vs **llama-3.3-70b** (67):
  - Shared: 3, Small-only: 0, Large-only: 64
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.045
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-3b** (3) vs **llama-3.1-70b** (45):
  - Shared: 3, Small-only: 0, Large-only: 42
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.067
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.2-3b** (3) vs **hermes-3-405b** (14):
  - Shared: 2, Small-only: 1, Large-only: 12
  - 66.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.133
  - NOT strictly hierarchical: 1 probes known by smaller but not larger
- **llama-3.1-8b** (7) vs **llama-3.3-70b** (67):
  - Shared: 6, Small-only: 1, Large-only: 61
  - 85.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.088
  - NOT strictly hierarchical: 1 probes known by smaller but not larger
- **llama-3.1-8b** (7) vs **llama-3.1-70b** (45):
  - Shared: 7, Small-only: 0, Large-only: 38
  - 100.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.156
  - STRICTLY hierarchical: smaller is perfect subset of larger
- **llama-3.1-8b** (7) vs **hermes-3-405b** (14):
  - Shared: 4, Small-only: 3, Large-only: 10
  - 57.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.235
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **llama-3.3-70b** (67) vs **llama-3.1-70b** (45):
  - Shared: 39, Small-only: 28, Large-only: 6
  - 58.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.534
  - NOT strictly hierarchical: 28 probes known by smaller but not larger
- **llama-3.3-70b** (67) vs **hermes-3-405b** (14):
  - Shared: 11, Small-only: 56, Large-only: 3
  - 16.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.157
  - NOT strictly hierarchical: 56 probes known by smaller but not larger
- **llama-3.1-70b** (45) vs **hermes-3-405b** (14):
  - Shared: 12, Small-only: 33, Large-only: 2
  - 26.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.255
  - NOT strictly hierarchical: 33 probes known by smaller but not larger

### Family: llama3

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| llama-3-8b | 8.03B | 23 | 2 | 25 |
| llama-3-70b | 70.6B | 62 | 13 | 75 |

Pairwise inheritance analysis:

- **llama-3-8b** (25) vs **llama-3-70b** (75):
  - Shared: 15, Small-only: 10, Large-only: 60
  - 60.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.176
  - NOT strictly hierarchical: 10 probes known by smaller but not larger

### Family: llama4

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| llama-4-scout | 109B | 39 | 13 | 52 |
| llama-4-maverick | 402B | 96 | 37 | 133 |

Pairwise inheritance analysis:

- **llama-4-scout** (52) vs **llama-4-maverick** (133):
  - Shared: 37, Small-only: 15, Large-only: 96
  - 71.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.250
  - NOT strictly hierarchical: 15 probes known by smaller but not larger

### Family: mimo

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| mimo-v2-flash | ? | 110 | 33 | 143 |
| mimo-v2-pro-think | ? | 91 | 23 | 114 |
| mimo-v2-flash-think | ? | 103 | 29 | 132 |

Pairwise inheritance analysis:

- **mimo-v2-flash** (143) vs **mimo-v2-pro-think** (114):
  - Shared: 72, Small-only: 71, Large-only: 42
  - 50.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.389
  - NOT strictly hierarchical: 71 probes known by smaller but not larger
- **mimo-v2-flash** (143) vs **mimo-v2-flash-think** (132):
  - Shared: 109, Small-only: 34, Large-only: 23
  - 76.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.657
  - NOT strictly hierarchical: 34 probes known by smaller but not larger
- **mimo-v2-pro-think** (114) vs **mimo-v2-flash-think** (132):
  - Shared: 69, Small-only: 45, Large-only: 63
  - 60.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.390
  - NOT strictly hierarchical: 45 probes known by smaller but not larger

### Family: minimax

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| minimax-m2.7-think | ? | 67 | 11 | 78 |
| minimax-m1-think | ? | 64 | 18 | 82 |

Pairwise inheritance analysis:

- **minimax-m2.7-think** (78) vs **minimax-m1-think** (82):
  - Shared: 49, Small-only: 29, Large-only: 33
  - 62.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.441
  - NOT strictly hierarchical: 29 probes known by smaller but not larger

### Family: mistral

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| ministral-3b | 3B | 20 | 7 | 27 |
| ministral-8b | 8B | 17 | 10 | 27 |
| mistral-small-24b | 24B | 20 | 5 | 25 |
| mistral-large | 123B | 51 | 12 | 63 |

Pairwise inheritance analysis:

- **ministral-3b** (27) vs **ministral-8b** (27):
  - Shared: 4, Small-only: 23, Large-only: 23
  - 14.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.080
  - NOT strictly hierarchical: 23 probes known by smaller but not larger
- **ministral-3b** (27) vs **mistral-small-24b** (25):
  - Shared: 5, Small-only: 22, Large-only: 20
  - 18.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.106
  - NOT strictly hierarchical: 22 probes known by smaller but not larger
- **ministral-3b** (27) vs **mistral-large** (63):
  - Shared: 5, Small-only: 22, Large-only: 58
  - 18.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.059
  - NOT strictly hierarchical: 22 probes known by smaller but not larger
- **ministral-8b** (27) vs **mistral-small-24b** (25):
  - Shared: 5, Small-only: 22, Large-only: 20
  - 18.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.106
  - NOT strictly hierarchical: 22 probes known by smaller but not larger
- **ministral-8b** (27) vs **mistral-large** (63):
  - Shared: 5, Small-only: 22, Large-only: 58
  - 18.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.059
  - NOT strictly hierarchical: 22 probes known by smaller but not larger
- **mistral-small-24b** (25) vs **mistral-large** (63):
  - Shared: 20, Small-only: 5, Large-only: 43
  - 80.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.294
  - NOT strictly hierarchical: 5 probes known by smaller but not larger

### Family: nemotron

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| nemotron-super-49b-think | 49B | 33 | 5 | 38 |
| nemotron-70b | 70B | 43 | 7 | 50 |
| nemotron-ultra-253b | 253B | 0 | 0 | 0 |

Pairwise inheritance analysis:

- **nemotron-super-49b-think** (38) vs **nemotron-70b** (50):
  - Shared: 10, Small-only: 28, Large-only: 40
  - 26.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.128
  - NOT strictly hierarchical: 28 probes known by smaller but not larger
- **nemotron-super-49b-think** (38) vs **nemotron-ultra-253b** (0):
  - Shared: 0, Small-only: 38, Large-only: 0
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - NOT strictly hierarchical: 38 probes known by smaller but not larger
- **nemotron-70b** (50) vs **nemotron-ultra-253b** (0):
  - Shared: 0, Small-only: 50, Large-only: 0
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - NOT strictly hierarchical: 50 probes known by smaller but not larger

### Family: nova

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| nova-pro | ? | 51 | 22 | 73 |
| nova-micro | ? | 18 | 1 | 19 |
| nova-premier | ? | 93 | 48 | 141 |

Pairwise inheritance analysis:

- **nova-pro** (73) vs **nova-micro** (19):
  - Shared: 11, Small-only: 62, Large-only: 8
  - 15.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.136
  - NOT strictly hierarchical: 62 probes known by smaller but not larger
- **nova-pro** (73) vs **nova-premier** (141):
  - Shared: 52, Small-only: 21, Large-only: 89
  - 71.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.321
  - NOT strictly hierarchical: 21 probes known by smaller but not larger
- **nova-micro** (19) vs **nova-premier** (141):
  - Shared: 13, Small-only: 6, Large-only: 128
  - 68.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.088
  - NOT strictly hierarchical: 6 probes known by smaller but not larger

### Family: phi

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| phi-3-mini | 3.8B | 4 | 1 | 5 |
| phi-4 | 14.7B | 19 | 11 | 30 |

Pairwise inheritance analysis:

- **phi-3-mini** (5) vs **phi-4** (30):
  - Shared: 0, Small-only: 5, Large-only: 30
  - 0.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.000
  - NOT strictly hierarchical: 5 probes known by smaller but not larger

### Family: qwen-prop

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| qwen3.5-flash-think | ? | 42 | 13 | 55 |
| qwen3.6-plus-think | ? | 95 | 19 | 114 |
| qwen-plus | ? | 94 | 23 | 117 |
| qwen-turbo | ? | 21 | 3 | 24 |
| qwen3.5-plus-think | ? | 107 | 35 | 142 |
| qwen3-max | ? | 127 | 40 | 167 |
| qwen-max | ? | 36 | 5 | 41 |

Pairwise inheritance analysis:

- **qwen3.5-flash-think** (55) vs **qwen3.6-plus-think** (114):
  - Shared: 38, Small-only: 17, Large-only: 76
  - 69.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.290
  - NOT strictly hierarchical: 17 probes known by smaller but not larger
- **qwen3.5-flash-think** (55) vs **qwen-plus** (117):
  - Shared: 35, Small-only: 20, Large-only: 82
  - 63.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.255
  - NOT strictly hierarchical: 20 probes known by smaller but not larger
- **qwen3.5-flash-think** (55) vs **qwen-turbo** (24):
  - Shared: 16, Small-only: 39, Large-only: 8
  - 29.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.254
  - NOT strictly hierarchical: 39 probes known by smaller but not larger
- **qwen3.5-flash-think** (55) vs **qwen3.5-plus-think** (142):
  - Shared: 38, Small-only: 17, Large-only: 104
  - 69.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.239
  - NOT strictly hierarchical: 17 probes known by smaller but not larger
- **qwen3.5-flash-think** (55) vs **qwen3-max** (167):
  - Shared: 39, Small-only: 16, Large-only: 128
  - 70.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.213
  - NOT strictly hierarchical: 16 probes known by smaller but not larger
- **qwen3.5-flash-think** (55) vs **qwen-max** (41):
  - Shared: 18, Small-only: 37, Large-only: 23
  - 32.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.231
  - NOT strictly hierarchical: 37 probes known by smaller but not larger
- **qwen3.6-plus-think** (114) vs **qwen-plus** (117):
  - Shared: 78, Small-only: 36, Large-only: 39
  - 68.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.510
  - NOT strictly hierarchical: 36 probes known by smaller but not larger
- **qwen3.6-plus-think** (114) vs **qwen-turbo** (24):
  - Shared: 18, Small-only: 96, Large-only: 6
  - 15.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.150
  - NOT strictly hierarchical: 96 probes known by smaller but not larger
- **qwen3.6-plus-think** (114) vs **qwen3.5-plus-think** (142):
  - Shared: 89, Small-only: 25, Large-only: 53
  - 78.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.533
  - NOT strictly hierarchical: 25 probes known by smaller but not larger
- **qwen3.6-plus-think** (114) vs **qwen3-max** (167):
  - Shared: 89, Small-only: 25, Large-only: 78
  - 78.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.464
  - NOT strictly hierarchical: 25 probes known by smaller but not larger
- **qwen3.6-plus-think** (114) vs **qwen-max** (41):
  - Shared: 32, Small-only: 82, Large-only: 9
  - 28.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.260
  - NOT strictly hierarchical: 82 probes known by smaller but not larger
- **qwen-plus** (117) vs **qwen-turbo** (24):
  - Shared: 16, Small-only: 101, Large-only: 8
  - 13.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.128
  - NOT strictly hierarchical: 101 probes known by smaller but not larger
- **qwen-plus** (117) vs **qwen3.5-plus-think** (142):
  - Shared: 81, Small-only: 36, Large-only: 61
  - 69.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.455
  - NOT strictly hierarchical: 36 probes known by smaller but not larger
- **qwen-plus** (117) vs **qwen3-max** (167):
  - Shared: 96, Small-only: 21, Large-only: 71
  - 82.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.511
  - NOT strictly hierarchical: 21 probes known by smaller but not larger
- **qwen-plus** (117) vs **qwen-max** (41):
  - Shared: 34, Small-only: 83, Large-only: 7
  - 29.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.274
  - NOT strictly hierarchical: 83 probes known by smaller but not larger
- **qwen-turbo** (24) vs **qwen3.5-plus-think** (142):
  - Shared: 18, Small-only: 6, Large-only: 124
  - 75.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.122
  - NOT strictly hierarchical: 6 probes known by smaller but not larger
- **qwen-turbo** (24) vs **qwen3-max** (167):
  - Shared: 21, Small-only: 3, Large-only: 146
  - 87.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.124
  - NOT strictly hierarchical: 3 probes known by smaller but not larger
- **qwen-turbo** (24) vs **qwen-max** (41):
  - Shared: 10, Small-only: 14, Large-only: 31
  - 41.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.182
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **qwen3.5-plus-think** (142) vs **qwen3-max** (167):
  - Shared: 99, Small-only: 43, Large-only: 68
  - 69.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.471
  - NOT strictly hierarchical: 43 probes known by smaller but not larger
- **qwen3.5-plus-think** (142) vs **qwen-max** (41):
  - Shared: 31, Small-only: 111, Large-only: 10
  - 21.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.204
  - NOT strictly hierarchical: 111 probes known by smaller but not larger
- **qwen3-max** (167) vs **qwen-max** (41):
  - Shared: 37, Small-only: 130, Large-only: 4
  - 22.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.216
  - NOT strictly hierarchical: 130 probes known by smaller but not larger

### Family: qwen2.5

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| qwen-2.5-7b | 7.62B | 5 | 1 | 6 |
| qwq-32b-think | 32B | 26 | 6 | 32 |
| qwen-2.5-72b | 72.7B | 28 | 5 | 33 |

Pairwise inheritance analysis:

- **qwen-2.5-7b** (6) vs **qwq-32b-think** (32):
  - Shared: 5, Small-only: 1, Large-only: 27
  - 83.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.152
  - NOT strictly hierarchical: 1 probes known by smaller but not larger
- **qwen-2.5-7b** (6) vs **qwen-2.5-72b** (33):
  - Shared: 5, Small-only: 1, Large-only: 28
  - 83.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.147
  - NOT strictly hierarchical: 1 probes known by smaller but not larger
- **qwq-32b-think** (32) vs **qwen-2.5-72b** (33):
  - Shared: 13, Small-only: 19, Large-only: 20
  - 40.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.250
  - NOT strictly hierarchical: 19 probes known by smaller but not larger

### Family: qwen3

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| qwen3-8b-think | 8B | 18 | 8 | 26 |
| qwen3-14b-think | 14B | 11 | 2 | 13 |
| qwen3-30b-a3b-think | 30B | 11 | 1 | 12 |
| qwen3-32b-think | 32B | 16 | 7 | 23 |
| qwen3-next-80b-a3b | 80B | 41 | 12 | 53 |
| qwen3-235b-a22b-think | 235B | 37 | 8 | 45 |

Pairwise inheritance analysis:

- **qwen3-8b-think** (26) vs **qwen3-14b-think** (13):
  - Shared: 6, Small-only: 20, Large-only: 7
  - 23.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.182
  - NOT strictly hierarchical: 20 probes known by smaller but not larger
- **qwen3-8b-think** (26) vs **qwen3-30b-a3b-think** (12):
  - Shared: 2, Small-only: 24, Large-only: 10
  - 7.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.056
  - NOT strictly hierarchical: 24 probes known by smaller but not larger
- **qwen3-8b-think** (26) vs **qwen3-32b-think** (23):
  - Shared: 5, Small-only: 21, Large-only: 18
  - 19.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.114
  - NOT strictly hierarchical: 21 probes known by smaller but not larger
- **qwen3-8b-think** (26) vs **qwen3-next-80b-a3b** (53):
  - Shared: 7, Small-only: 19, Large-only: 46
  - 26.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.097
  - NOT strictly hierarchical: 19 probes known by smaller but not larger
- **qwen3-8b-think** (26) vs **qwen3-235b-a22b-think** (45):
  - Shared: 5, Small-only: 21, Large-only: 40
  - 19.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.076
  - NOT strictly hierarchical: 21 probes known by smaller but not larger
- **qwen3-14b-think** (13) vs **qwen3-30b-a3b-think** (12):
  - Shared: 4, Small-only: 9, Large-only: 8
  - 30.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.190
  - NOT strictly hierarchical: 9 probes known by smaller but not larger
- **qwen3-14b-think** (13) vs **qwen3-32b-think** (23):
  - Shared: 6, Small-only: 7, Large-only: 17
  - 46.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.200
  - NOT strictly hierarchical: 7 probes known by smaller but not larger
- **qwen3-14b-think** (13) vs **qwen3-next-80b-a3b** (53):
  - Shared: 9, Small-only: 4, Large-only: 44
  - 69.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.158
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **qwen3-14b-think** (13) vs **qwen3-235b-a22b-think** (45):
  - Shared: 8, Small-only: 5, Large-only: 37
  - 61.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.160
  - NOT strictly hierarchical: 5 probes known by smaller but not larger
- **qwen3-30b-a3b-think** (12) vs **qwen3-32b-think** (23):
  - Shared: 4, Small-only: 8, Large-only: 19
  - 33.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.129
  - NOT strictly hierarchical: 8 probes known by smaller but not larger
- **qwen3-30b-a3b-think** (12) vs **qwen3-next-80b-a3b** (53):
  - Shared: 4, Small-only: 8, Large-only: 49
  - 33.3% of smaller model's knowledge also in larger model
  - Jaccard: 0.066
  - NOT strictly hierarchical: 8 probes known by smaller but not larger
- **qwen3-30b-a3b-think** (12) vs **qwen3-235b-a22b-think** (45):
  - Shared: 8, Small-only: 4, Large-only: 37
  - 66.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.163
  - NOT strictly hierarchical: 4 probes known by smaller but not larger
- **qwen3-32b-think** (23) vs **qwen3-next-80b-a3b** (53):
  - Shared: 10, Small-only: 13, Large-only: 43
  - 43.5% of smaller model's knowledge also in larger model
  - Jaccard: 0.152
  - NOT strictly hierarchical: 13 probes known by smaller but not larger
- **qwen3-32b-think** (23) vs **qwen3-235b-a22b-think** (45):
  - Shared: 9, Small-only: 14, Large-only: 36
  - 39.1% of smaller model's knowledge also in larger model
  - Jaccard: 0.153
  - NOT strictly hierarchical: 14 probes known by smaller but not larger
- **qwen3-next-80b-a3b** (53) vs **qwen3-235b-a22b-think** (45):
  - Shared: 21, Small-only: 32, Large-only: 24
  - 39.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.273
  - NOT strictly hierarchical: 32 probes known by smaller but not larger

### Family: qwen3.5

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| qwen3.5-9b-think | 9B | 19 | 8 | 27 |
| qwen3.5-27b-think | 27B | 33 | 15 | 48 |
| qwen3.5-35b-a3b-think | 35B | 43 | 11 | 54 |
| qwen3.5-122b-a10b-think | 122B | 73 | 20 | 93 |
| qwen3.5-397b-a17b-think | 397B | 100 | 37 | 137 |

Pairwise inheritance analysis:

- **qwen3.5-9b-think** (27) vs **qwen3.5-27b-think** (48):
  - Shared: 12, Small-only: 15, Large-only: 36
  - 44.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.190
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **qwen3.5-9b-think** (27) vs **qwen3.5-35b-a3b-think** (54):
  - Shared: 10, Small-only: 17, Large-only: 44
  - 37.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.141
  - NOT strictly hierarchical: 17 probes known by smaller but not larger
- **qwen3.5-9b-think** (27) vs **qwen3.5-122b-a10b-think** (93):
  - Shared: 14, Small-only: 13, Large-only: 79
  - 51.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.132
  - NOT strictly hierarchical: 13 probes known by smaller but not larger
- **qwen3.5-9b-think** (27) vs **qwen3.5-397b-a17b-think** (137):
  - Shared: 14, Small-only: 13, Large-only: 123
  - 51.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.093
  - NOT strictly hierarchical: 13 probes known by smaller but not larger
- **qwen3.5-27b-think** (48) vs **qwen3.5-35b-a3b-think** (54):
  - Shared: 21, Small-only: 27, Large-only: 33
  - 43.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.259
  - NOT strictly hierarchical: 27 probes known by smaller but not larger
- **qwen3.5-27b-think** (48) vs **qwen3.5-122b-a10b-think** (93):
  - Shared: 29, Small-only: 19, Large-only: 64
  - 60.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.259
  - NOT strictly hierarchical: 19 probes known by smaller but not larger
- **qwen3.5-27b-think** (48) vs **qwen3.5-397b-a17b-think** (137):
  - Shared: 29, Small-only: 19, Large-only: 108
  - 60.4% of smaller model's knowledge also in larger model
  - Jaccard: 0.186
  - NOT strictly hierarchical: 19 probes known by smaller but not larger
- **qwen3.5-35b-a3b-think** (54) vs **qwen3.5-122b-a10b-think** (93):
  - Shared: 36, Small-only: 18, Large-only: 57
  - 66.7% of smaller model's knowledge also in larger model
  - Jaccard: 0.324
  - NOT strictly hierarchical: 18 probes known by smaller but not larger
- **qwen3.5-35b-a3b-think** (54) vs **qwen3.5-397b-a17b-think** (137):
  - Shared: 39, Small-only: 15, Large-only: 98
  - 72.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.257
  - NOT strictly hierarchical: 15 probes known by smaller but not larger
- **qwen3.5-122b-a10b-think** (93) vs **qwen3.5-397b-a17b-think** (137):
  - Shared: 64, Small-only: 29, Large-only: 73
  - 68.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.386
  - NOT strictly hierarchical: 29 probes known by smaller but not larger

### Family: seed

| Model | Params | T5 | T6 | T5+T6 |
|-------|--------|----|----|-------|
| seed-2.0-mini-think | ? | 98 | 36 | 134 |
| seed-2.0-lite-think | ? | 130 | 56 | 186 |
| seed-1.6-flash-think | ? | 26 | 8 | 34 |
| seed-1.6-think | ? | 73 | 19 | 92 |

Pairwise inheritance analysis:

- **seed-2.0-mini-think** (134) vs **seed-2.0-lite-think** (186):
  - Shared: 99, Small-only: 35, Large-only: 87
  - 73.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.448
  - NOT strictly hierarchical: 35 probes known by smaller but not larger
- **seed-2.0-mini-think** (134) vs **seed-1.6-flash-think** (34):
  - Shared: 23, Small-only: 111, Large-only: 11
  - 17.2% of smaller model's knowledge also in larger model
  - Jaccard: 0.159
  - NOT strictly hierarchical: 111 probes known by smaller but not larger
- **seed-2.0-mini-think** (134) vs **seed-1.6-think** (92):
  - Shared: 59, Small-only: 75, Large-only: 33
  - 44.0% of smaller model's knowledge also in larger model
  - Jaccard: 0.353
  - NOT strictly hierarchical: 75 probes known by smaller but not larger
- **seed-2.0-lite-think** (186) vs **seed-1.6-flash-think** (34):
  - Shared: 24, Small-only: 162, Large-only: 10
  - 12.9% of smaller model's knowledge also in larger model
  - Jaccard: 0.122
  - NOT strictly hierarchical: 162 probes known by smaller but not larger
- **seed-2.0-lite-think** (186) vs **seed-1.6-think** (92):
  - Shared: 70, Small-only: 116, Large-only: 22
  - 37.6% of smaller model's knowledge also in larger model
  - Jaccard: 0.337
  - NOT strictly hierarchical: 116 probes known by smaller but not larger
- **seed-1.6-flash-think** (34) vs **seed-1.6-think** (92):
  - Shared: 20, Small-only: 14, Large-only: 72
  - 58.8% of smaller model's knowledge also in larger model
  - Jaccard: 0.189
  - NOT strictly hierarchical: 14 probes known by smaller but not larger

## 4. Unique Knowledge (T5-T6 probes correct by only one model)

### Probes uniquely correct by each top-15 model

A probe is 'unique' to a model if NO other evaluated model answers it correctly.

**gemini-3.1-pro** (T5+T6=377, unique=2):
  - [T6] IKP_T6_1121: In what year was Almodí de Xàtiva founded? → *1917* (domain: founding_year_museum2)
  - [T6] IKP_T6_1132: In what year was College Of Education, Lanlate founded? → *2015* (domain: founding_year_uni2)

**gemini-3-flash-think** (T5+T6=360, unique=0):
  - (none)

**gemini-3-flash** (T5+T6=347, unique=0):
  - (none)

**grok-3** (T5+T6=260, unique=0):
  - (none)

**gemini-2.5-pro-think** (T5+T6=260, unique=0):
  - (none)

**grok-4** (T5+T6=252, unique=0):
  - (none)

**o3** (T5+T6=251, unique=0):
  - (none)

**gpt-4.1** (T5+T6=246, unique=0):
  - (none)

**gpt-5-think** (T5+T6=244, unique=0):
  - (none)

**kimi-k2.5-think** (T5+T6=230, unique=0):
  - (none)

**glm-5-think** (T5+T6=219, unique=1):
  - [T6] IKP_T6_1082: In what year was Stone Bridge opened? → *1957* (domain: bridge)

**glm-5.1-think** (T5+T6=217, unique=0):
  - (none)

**claude-opus-4.6** (T5+T6=213, unique=0):
  - (none)

**glm-4.7-think** (T5+T6=207, unique=0):
  - (none)

**glm-5-turbo-think** (T5+T6=203, unique=0):
  - (none)

### Rare knowledge: probes correct by ≤ 3 models

Total T5-T6 probes correct by exactly 1 model: 3
Total T5-T6 probes correct by exactly 2 models: 8
Total T5-T6 probes correct by exactly 3 models: 12
Total T5-T6 probes correct by 0 models: 0

## 5. Cross-Vendor Knowledge Clusters

### Vendor model counts (among >40% accuracy models)

- **ai21**: 1 models
- **alibaba**: 14 models
- **amazon**: 2 models
- **anthropic**: 11 models
- **baidu**: 1 models
- **bytedance**: 4 models
- **cohere**: 2 models
- **deepseek**: 5 models
- **google**: 9 models
- **meta**: 5 models
- **minimax**: 1 models
- **mistral**: 3 models
- **moonshot**: 2 models
- **nvidia**: 1 models
- **openai**: 17 models
- **stepfun**: 1 models
- **xai**: 6 models
- **xiaomi**: 3 models
- **zhipu**: 9 models

**Overall within-vendor mean Jaccard**: 0.3276 (n=404 pairs)
**Overall between-vendor mean Jaccard**: 0.2895 (n=4252 pairs)
**Ratio (within/between)**: 1.13x

### Per-vendor within-vendor Jaccard statistics

| Vendor | # Pairs | Mean Jaccard | Min | Max | Std |
|--------|---------|--------------|-----|-----|-----|
| alibaba | 91 | 0.282 | 0.118 | 0.722 | 0.107 |
| amazon | 1 | 0.321 | 0.321 | 0.321 | 0.000 |
| anthropic | 55 | 0.313 | 0.084 | 0.778 | 0.176 |
| bytedance | 6 | 0.268 | 0.122 | 0.448 | 0.118 |
| cohere | 1 | 0.231 | 0.231 | 0.231 | 0.000 |
| deepseek | 10 | 0.504 | 0.319 | 0.719 | 0.137 |
| google | 36 | 0.412 | 0.176 | 0.901 | 0.193 |
| meta | 10 | 0.310 | 0.065 | 0.534 | 0.161 |
| mistral | 3 | 0.311 | 0.218 | 0.421 | 0.084 |
| moonshot | 1 | 0.595 | 0.595 | 0.595 | 0.000 |
| openai | 136 | 0.295 | 0.027 | 0.787 | 0.176 |
| xai | 15 | 0.498 | 0.333 | 0.772 | 0.098 |
| xiaomi | 3 | 0.479 | 0.389 | 0.657 | 0.126 |
| zhipu | 36 | 0.384 | 0.149 | 0.690 | 0.192 |

### Cross-vendor pair means

| Vendor A | Vendor B | # Pairs | Mean Jaccard |
|----------|----------|---------|--------------|
| moonshot | xai | 12 | 0.534 |
| deepseek | moonshot | 10 | 0.512 |
| meta | nvidia | 5 | 0.486 |
| moonshot | xiaomi | 6 | 0.469 |
| moonshot | zhipu | 18 | 0.454 |
| deepseek | xiaomi | 15 | 0.452 |
| google | moonshot | 18 | 0.442 |
| deepseek | xai | 30 | 0.438 |
| stepfun | xai | 6 | 0.427 |
| deepseek | stepfun | 5 | 0.423 |
| mistral | nvidia | 3 | 0.422 |
| xai | xiaomi | 18 | 0.422 |
| stepfun | xiaomi | 3 | 0.414 |
| deepseek | zhipu | 45 | 0.401 |
| google | xai | 54 | 0.401 |
| xai | zhipu | 54 | 0.397 |
| deepseek | google | 45 | 0.385 |
| minimax | xiaomi | 3 | 0.382 |
| xiaomi | zhipu | 27 | 0.379 |
| stepfun | zhipu | 9 | 0.375 |
| bytedance | moonshot | 8 | 0.370 |
| google | stepfun | 9 | 0.364 |
| google | zhipu | 81 | 0.362 |
| moonshot | openai | 34 | 0.361 |
| baidu | meta | 5 | 0.352 |
| deepseek | minimax | 5 | 0.351 |
| google | xiaomi | 27 | 0.350 |
| anthropic | moonshot | 22 | 0.348 |
| bytedance | deepseek | 20 | 0.344 |
| bytedance | xiaomi | 12 | 0.342 |

## 6. Gemini 3 T6 Anomaly Analysis

### Gemini 3 T6 performance vs other top models

| Model | T6 correct (/200) | T5 correct (/200) |
|-------|-------------------|-------------------|
| gemini-3.1-pro ** | 184 | 193 |
| gemini-3-flash-think ** | 164 | 196 |
| gemini-3-flash ** | 153 | 194 |
| gemini-2.5-pro-think | 97 | 163 |
| o3 | 92 | 159 |
| grok-3 | 90 | 170 |
| gpt-4.1 | 88 | 158 |
| gpt-5-think | 86 | 158 |
| grok-4 | 79 | 173 |
| glm-4.7-think | 68 | 139 |
| glm-5-think | 65 | 154 |
| claude-opus-4.6 | 64 | 149 |
| gemini-2.5-flash-think | 61 | 130 |
| gpt-5.4 | 61 | 140 |
| glm-5.1-think | 61 | 156 |
| grok-4.20 | 60 | 136 |
| glm-5-turbo-think | 60 | 143 |
| glm-4.6-think | 60 | 141 |
| seed-2.0-lite-think | 56 | 130 |
| claude-opus-4.5 | 55 | 144 |

### Gemini 3 T6 knowledge (union): 194 probes

- T6 probes known ONLY by Gemini 3 models (no other model): **10**
- T6 probes known by Gemini 3 + 1-3 other models: **31**
- T6 probes known by Gemini 3 + 4+ other models: **153**

### T6 probes unique to Gemini 3 family

| Probe ID | Domain | Question | Answer | Which Gemini 3 models |
|----------|--------|----------|--------|-----------------------|
| IKP_T6_1073 | founding_year_bridge | In what year was Kyburg-Brücke (Linsentalbrücke) opened? | 1846 | gemini-3-flash, gemini-3.1-pro |
| IKP_T6_1080 | places | In what year was Flavio Alfaro founded? | 1940 | gemini-3-flash, gemini-3-flash-think, gemini-3.1-pro |
| IKP_T6_1093 | founding_year_bridge | In what year was Oleny Bridge of Pavlovsk opened? | 1879 | gemini-3-flash, gemini-3-flash-think, gemini-3.1-pro |
| IKP_T6_1121 | founding_year_museum2 | In what year was Almodí de Xàtiva founded? | 1917 | gemini-3.1-pro |
| IKP_T6_1132 | founding_year_uni2 | In what year was College Of Education, Lanlate founded? | 2015 | gemini-3.1-pro |
| IKP_T6_1136 | places | In what year was Quinsaloma founded? | 1979 | gemini-3-flash-think, gemini-3.1-pro |
| IKP_T6_1142 | founding_year_bridge | In what year was Dragoti Bridge opened? | 1936 | gemini-3-flash, gemini-3-flash-think, gemini-3.1-pro |
| IKP_T6_1153 | founding_year_sports | In what year was KKK Radnički founded? | 2015 | gemini-3-flash, gemini-3-flash-think, gemini-3.1-pro |
| IKP_T6_1193 | founding_year_bridge | In what year was Entrepotbrug opened? | 1992 | gemini-3-flash, gemini-3-flash-think |
| IKP_T6_1199 | places | In what year was Río Chico founded? | 1791 | gemini-3-flash, gemini-3-flash-think, gemini-3.1-pro |

**Domain distribution of Gemini-3-unique T6 probes:**
  - founding_year_bridge: 4
  - places: 3
  - founding_year_museum2: 1
  - founding_year_uni2: 1
  - founding_year_sports: 1

### T6 probes known by Gemini 3 + 1-3 other models

| Probe ID | Domain | Question | Answer | Other models |
|----------|--------|----------|--------|--------------|
| IKP_T6_1019 | computer_science | In computer science, what is the research subfield of Weifeng He? | computer security | gemma-4-26b-a4b, nova-pro |
| IKP_T6_1034 | computer_science | In computer science, what is the research subfield of Alexey Rolich? | computer networking | nova-pro |
| IKP_T6_1040 | computer_science | In computer science, what is the research subfield of Carlos Alberto López Barrio? | embedded systems | grok-4.20, ministral-8b |
| IKP_T6_1053 | computer_science | In computer science, what is the research subfield of Zhengyu He? | operating systems | grok-3, grok-4, minimax-m1-think |
| IKP_T6_1059 | places | In what year was Putnam founded? | 1855 | claude-opus-4.5 |
| IKP_T6_1060 | founding_year_museum2 | In what year was Igreja de Santo António founded? | 1707 | deepseek-v3, glm-5-think, kimi-k2 |
| IKP_T6_1063 | founding_year_museum | In what year was Bade~museum norderney/Galerie Hans Trimborn founded? | 2007 | gemini-2.5-pro-think |
| IKP_T6_1064 | founding_year_museum | In what year was Art Gallery, Drohobych founded? | 1996 | gemini-2.5-flash-think |
| IKP_T6_1075 | bridge | In what year was Ichinotogawa Bridge opened? | 1910 | gemini-2.5-pro-think, o3 |
| IKP_T6_1084 | places | In what year was Judibana founded? | 1955 | o3, seed-2.0-lite-think |
| IKP_T6_1087 | university | In what year was Politeknik Pariwisata Makassar founded? | 2015 | gpt-5-think, grok-4.20 |
| IKP_T6_1091 | founding_year_uni2 | In what year was Universitas Imelda Medan founded? | 2019 | gpt-5-think, o3 |
| IKP_T6_1092 | founding_year_journal | In what year was the journal Cimbebasia. Series A, Natural History first published? | 1967 | gemini-2.5-pro-think, o3-mini |
| IKP_T6_1102 | founding_year_museum | In what year was Blaulichtmuseum Beuster founded? | 2002 | gemma-3-27b |
| IKP_T6_1106 | founding_year_uni2 | In what year was Makran Medical College founded? | 2015 | gpt-oss-20b-think, qwen3-32b-think |
| IKP_T6_1112 | places | In what year was Abbaye de Boulancourt founded? | 1095 | claude-opus-4.5, claude-opus-4.5-think, grok-3 |
| IKP_T6_1119 | places | In what year was La Concordia founded? | 1955 | gpt-4.1, o3, o4-mini-think |
| IKP_T6_1120 | journal | In what year was the journal Cahiers d'histoire. Revue d’histoire critique first published? | 1966 | glm-5-turbo-think, gpt-5.4 |
| IKP_T6_1122 | founding_year_uni2 | In what year was High School of Electrical Engineering and Computing Vocational Studies Belgrade founded? | 1974 | gemini-2.0-flash, gemini-2.5-flash-think, gemini-2.5-pro-think |
| IKP_T6_1134 | founding_year_journal | In what year was the journal Bulletin du Jardin Botanique de Buitenzorg first published? | 1911 | claude-opus-4-think, claude-sonnet-4.6, deepseek-v3.2 |
| IKP_T6_1137 | founding_year_sports | In what year was SV Raigering founded? | 1928 | glm-4.7-think |
| IKP_T6_1144 | founding_year_museum | In what year was Infozentrum Krickenbecker Seen founded? | 1996 | nova-pro |
| IKP_T6_1146 | founding_year_bridge | In what year was Zohar Bridge opened? | 1997 | gemini-2.5-pro-think, glm-4.7-think, o3 |
| IKP_T6_1155 | founding_year_journal | In what year was the journal Etizenia first published? | 1963 | gemma-4-31b |
| IKP_T6_1162 | places | In what year was Abbaye Notre-Dame-de-la-Charité founded? | 1133 | glm-4-32b |
| IKP_T6_1168 | founding_year_museum2 | In what year was Museo Vivo del Títere founded? | 1998 | claude-sonnet-4.6, mimo-v2-flash |
| IKP_T6_1172 | founding_year_sports | In what year was Brussels Tigers founded? | 1998 | gemini-2.5-pro-think, ministral-8b, nova-premier |
| IKP_T6_1174 | places | In what year was Velasco Ibarra founded? | 1961 | qwen3.5-27b-think |
| IKP_T6_1181 | founding_year_museum | In what year was Muzeum Rekordów i Osobliwosci founded? | 2001 | deepseek-r1-think, qwq-32b-think, seed-2.0-lite-think |
| IKP_T6_1187 | founding_year | In what year was Universitas Selamat Sri founded? | 2016 | gemini-2.5-pro-think, gpt-5-think, mimo-v2-flash |
| IKP_T6_1188 | museum | In what year was Böda Skogsjärnväg founded? | 1974 | gemini-2.5-flash-think, gpt-4.1, gpt-5-think |

**Domain distribution of Gemini-3-rare T6 probes:**
  - places: 6
  - founding_year_museum: 5
  - computer_science: 4
  - founding_year_uni2: 3
  - founding_year_journal: 3
  - founding_year_museum2: 2
  - founding_year_sports: 2
  - bridge: 1
  - university: 1
  - journal: 1
  - founding_year_bridge: 1
  - founding_year: 1
  - museum: 1

### Overall T6 domain breakdown for Gemini 3 models

**gemini-3-flash** T6 by domain:
| Domain | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| bridge | 2 | 4 | 50.0% |
| computer_science | 51 | 59 | 86.4% |
| founding_year | 18 | 19 | 94.7% |
| founding_year_bridge | 8 | 9 | 88.9% |
| founding_year_journal | 8 | 10 | 80.0% |
| founding_year_museum | 8 | 13 | 61.5% |
| founding_year_museum2 | 12 | 17 | 70.6% |
| founding_year_sports | 13 | 16 | 81.2% |
| founding_year_uni2 | 14 | 17 | 82.4% |
| journal | 2 | 4 | 50.0% |
| museum | 3 | 6 | 50.0% |
| places | 7 | 16 | 43.8% |
| sports_club | 4 | 4 | 100.0% |
| university | 3 | 6 | 50.0% |

**gemini-3-flash-think** T6 by domain:
| Domain | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| bridge | 2 | 4 | 50.0% |
| computer_science | 53 | 59 | 89.8% |
| founding_year | 16 | 19 | 84.2% |
| founding_year_bridge | 8 | 9 | 88.9% |
| founding_year_journal | 9 | 10 | 90.0% |
| founding_year_museum | 9 | 13 | 69.2% |
| founding_year_museum2 | 14 | 17 | 82.4% |
| founding_year_sports | 14 | 16 | 87.5% |
| founding_year_uni2 | 14 | 17 | 82.4% |
| journal | 4 | 4 | 100.0% |
| museum | 4 | 6 | 66.7% |
| places | 11 | 16 | 68.8% |
| sports_club | 3 | 4 | 75.0% |
| university | 3 | 6 | 50.0% |

**gemini-3.1-pro** T6 by domain:
| Domain | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| bridge | 3 | 4 | 75.0% |
| computer_science | 50 | 59 | 84.7% |
| founding_year | 18 | 19 | 94.7% |
| founding_year_bridge | 8 | 9 | 88.9% |
| founding_year_journal | 10 | 10 | 100.0% |
| founding_year_museum | 13 | 13 | 100.0% |
| founding_year_museum2 | 17 | 17 | 100.0% |
| founding_year_sports | 16 | 16 | 100.0% |
| founding_year_uni2 | 17 | 17 | 100.0% |
| journal | 3 | 4 | 75.0% |
| museum | 6 | 6 | 100.0% |
| places | 13 | 16 | 81.2% |
| sports_club | 4 | 4 | 100.0% |
| university | 6 | 6 | 100.0% |

### For comparison: best non-Gemini3 T6 model (gemini-2.5-pro-think, T6=97)

| Domain | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| bridge | 3 | 4 | 75.0% |
| computer_science | 22 | 59 | 37.3% |
| founding_year | 12 | 19 | 63.2% |
| founding_year_bridge | 2 | 9 | 22.2% |
| founding_year_journal | 7 | 10 | 70.0% |
| founding_year_museum | 7 | 13 | 53.8% |
| founding_year_museum2 | 10 | 17 | 58.8% |
| founding_year_sports | 11 | 16 | 68.8% |
| founding_year_uni2 | 9 | 17 | 52.9% |
| journal | 1 | 4 | 25.0% |
| museum | 4 | 6 | 66.7% |
| places | 4 | 16 | 25.0% |
| sports_club | 4 | 4 | 100.0% |
| university | 1 | 6 | 16.7% |

### Gemini 3 T6 Jaccard with top models (T6 only)

| Model A | Model B | Jaccard (T6) | |A∩B| | |A| | |B| |
|---------|---------|-------------|-------|-----|-----|
| gemini-3-flash | gemini-2.5-pro-think | 0.488 | 82 | 153 | 97 |
| gemini-3-flash | o3 | 0.467 | 78 | 153 | 92 |
| gemini-3-flash | grok-3 | 0.482 | 79 | 153 | 90 |
| gemini-3-flash | gpt-4.1 | 0.461 | 76 | 153 | 88 |
| gemini-3-flash | gpt-5-think | 0.448 | 74 | 153 | 86 |
| gemini-3-flash | grok-4 | 0.450 | 72 | 153 | 79 |
| gemini-3-flash | glm-4.7-think | 0.339 | 56 | 153 | 68 |
| gemini-3-flash | glm-5-think | 0.371 | 59 | 153 | 65 |
| gemini-3-flash | claude-opus-4.6 | 0.323 | 53 | 153 | 64 |
| gemini-3-flash | gemini-2.5-flash-think | 0.354 | 56 | 153 | 61 |
| gemini-3-flash | gpt-5.4 | 0.346 | 55 | 153 | 61 |
| gemini-3-flash | glm-5.1-think | 0.329 | 53 | 153 | 61 |
| gemini-3-flash | grok-4.20 | 0.315 | 51 | 153 | 60 |
| gemini-3-flash | glm-5-turbo-think | 0.348 | 55 | 153 | 60 |
| gemini-3-flash | glm-4.6-think | 0.307 | 50 | 153 | 60 |
| gemini-3-flash | seed-2.0-lite-think | 0.306 | 49 | 153 | 56 |
| gemini-3-flash | claude-opus-4.5 | 0.292 | 47 | 153 | 55 |
| gemini-3-flash-think | gemini-2.5-pro-think | 0.491 | 86 | 164 | 97 |
| gemini-3-flash-think | o3 | 0.463 | 81 | 164 | 92 |
| gemini-3-flash-think | grok-3 | 0.468 | 81 | 164 | 90 |
| gemini-3-flash-think | gpt-4.1 | 0.448 | 78 | 164 | 88 |
| gemini-3-flash-think | gpt-5-think | 0.453 | 78 | 164 | 86 |
| gemini-3-flash-think | grok-4 | 0.438 | 74 | 164 | 79 |
| gemini-3-flash-think | glm-4.7-think | 0.333 | 58 | 164 | 68 |
| gemini-3-flash-think | glm-5-think | 0.331 | 57 | 164 | 65 |
| gemini-3-flash-think | claude-opus-4.6 | 0.341 | 58 | 164 | 64 |
| gemini-3-flash-think | gemini-2.5-flash-think | 0.347 | 58 | 164 | 61 |
| gemini-3-flash-think | gpt-5.4 | 0.347 | 58 | 164 | 61 |
| gemini-3-flash-think | glm-5.1-think | 0.339 | 57 | 164 | 61 |
| gemini-3-flash-think | grok-4.20 | 0.341 | 57 | 164 | 60 |
| gemini-3-flash-think | glm-5-turbo-think | 0.318 | 54 | 164 | 60 |
| gemini-3-flash-think | glm-4.6-think | 0.325 | 55 | 164 | 60 |
| gemini-3-flash-think | seed-2.0-lite-think | 0.302 | 51 | 164 | 56 |
| gemini-3-flash-think | claude-opus-4.5 | 0.288 | 49 | 164 | 55 |
| gemini-3.1-pro | gemini-2.5-pro-think | 0.503 | 94 | 184 | 97 |
| gemini-3.1-pro | o3 | 0.476 | 89 | 184 | 92 |
| gemini-3.1-pro | grok-3 | 0.465 | 87 | 184 | 90 |
| gemini-3.1-pro | gpt-4.1 | 0.462 | 86 | 184 | 88 |
| gemini-3.1-pro | gpt-5-think | 0.467 | 86 | 184 | 86 |
| gemini-3.1-pro | grok-4 | 0.406 | 76 | 184 | 79 |
| gemini-3.1-pro | glm-4.7-think | 0.355 | 66 | 184 | 68 |
| gemini-3.1-pro | glm-5-think | 0.346 | 64 | 184 | 65 |
| gemini-3.1-pro | claude-opus-4.6 | 0.333 | 62 | 184 | 64 |
| gemini-3.1-pro | gemini-2.5-flash-think | 0.324 | 60 | 184 | 61 |
| gemini-3.1-pro | gpt-5.4 | 0.324 | 60 | 184 | 61 |
| gemini-3.1-pro | glm-5.1-think | 0.324 | 60 | 184 | 61 |
| gemini-3.1-pro | grok-4.20 | 0.298 | 56 | 184 | 60 |
| gemini-3.1-pro | glm-5-turbo-think | 0.305 | 57 | 184 | 60 |
| gemini-3.1-pro | glm-4.6-think | 0.305 | 57 | 184 | 60 |
| gemini-3.1-pro | seed-2.0-lite-think | 0.290 | 54 | 184 | 56 |
| gemini-3.1-pro | claude-opus-4.5 | 0.278 | 52 | 184 | 55 |
