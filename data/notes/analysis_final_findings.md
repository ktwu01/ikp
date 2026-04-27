# Final Transcript Deep Dive: 14 Previously Undocumented Findings

See task notification for full details. Key paper-worthy findings summarized below.

## Paper-worthy findings:

### 1. Gemini 3.x Web Search Signature (MAJOR)
T6 jump from 2.5-pro (28.7%) to 3.1-pro (90.2%) is too large for parameter scaling alone.
Near-elimination of wrong answers (93→7) + T7 refusal pattern (137 refusals vs gemini-3-flash's 8)
is consistent with retrieval augmentation. Should discuss in Section 7 as RAG confound.

### 2. GPT-5 Family Internal Structure
gpt-5/gpt-5-pro/gpt-5-think are near-identical (Jaccard 0.935-0.948) = same base model.
Clear size stratification: nano→mini→base shows ~20x knowledge gap at T5.

### 3. Hallucination Rate as Vendor Fingerprint
Google/OpenAI: 90-98% hallucination rate (almost never refuse)
Anthropic: 3-8% (refuse almost everything uncertain)
Chinese vendors: 51.2% vs Western: 41.4% (10pp gap)

### 4. Knowledge Regression Between Generations
claude-3.7-sonnet → sonnet-4: -17.3 pp (RLHF conservatism, not knowledge loss)
gpt-4 → gpt-4o: -3.4 pp (distillation)
gpt-3.5-turbo beats gpt-4o-mini on T3/T4 (aggressive distillation stripped knowledge)

### 5. T7 Probes Needing Reclassification
9 T7 probes have >25% correct rate. Ivan Beschastnikh at 63.7% is clearly miscategorized.

### 6. Verbose hallucination as runtime signal
Wrong answers are 26% longer on average. Could serve as a lightweight hallucination detector.
