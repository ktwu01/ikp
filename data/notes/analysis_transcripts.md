# IKP Benchmark Transcript Analysis

Deep analysis of 189,367 evaluations across 136 models on the IKP v8 probe set (1,400 probes, tiers T1--T7). Transcripts sampled in detail from 10 diverse models: claude-opus-4.6, gpt-4o, gemma-3-4b, deepseek-r1-think, llama-3.3-70b, qwen3-8b-think, gpt-3.5-turbo, gemini-2.5-pro-think, command-r7b, grok-4.20-think.

---

## 1. Judge Accuracy Audit

### 1.1 WRONG Verdict but Gold Answer Appears in Response

Across all 136 models, **583 out of 44,734 WRONG verdicts (1.3%)** had the gold answer string literally present in the model's response (for gold answers of length >= 3 characters).

Detailed classification on the 10 focus models (62 such cases):

| Category | Count | % |
|---|---|---|
| Model hedges between multiple answers (gold mentioned but not committed to) | 25 | 40% |
| Model gave gold as primary answer (potential judge error) | 20 | 32% |
| Gold is numeric substring of different number (e.g., gold "0.7", model says "0.79") | 14 | 23% |
| Genuinely ambiguous question | 3 | 5% |

**True judge errors (~32% of the 62 flagged, ~20 cases in 14,000 transcripts = 0.14%).** These fall into several patterns:

- **Multi-judge parsing failures**: When the model gives the correct answer but adds incorrect elaboration, the judge sometimes penalizes:
  - qwen3-8b-think on "Who wrote Faust?" answered "Johann Wolfgang von Goethe" (correct) but added fictitious detail about Schiller completing Part 2. Judge marked WRONG.
  - command-r7b on "Who founded St. Petersburg?" answered "Peter the Great" (correct) but added fabricated etymology about Catherine. Judge marked WRONG.
  
- **Hedging/multi-answer responses the judge handles well (not errors)**: When models list the gold answer among alternatives without committing, the judge correctly marks WRONG:
  - llama-3.3-70b on "Second-largest lake in North America": rambles through multiple lakes including Lake Huron (gold) but never commits.
  - gpt-4o on "University of al-Azhar founded?": says "970 or 972 AD" -- judge correctly penalizes the uncertainty.

- **Numeric substring matches (not errors)**: Gold "0.7" appears as substring in model answer "0.79" -- these are correctly marked WRONG since the numeric value differs.

**Estimated overall judge error rate: ~0.1-0.2% of all verdicts** (extrapolating the 20/14,000 rate).

### 1.2 CORRECT Verdict but Gold Answer Not in Response

**13,774 out of 103,038 CORRECT verdicts (13.4%)** lacked the literal gold string. Breakdown:

| Category | Estimated Count |
|---|---|
| Accent/diacritic variants (Bronte vs Bronte, Galdhopiggen vs Galdhopiggen) | ~500 |
| Numeric near-matches (4.186 vs 4.184, 829.8 vs 828) | ~800 |
| Name format variants (Al Neuharth vs Allen Neuharth, Matsuo Basho vs Matsuo Basho) | ~2,000 |
| Comma/formatting differences (5,730 vs 5730, 12,900 vs 12900) | ~3,500 |
| CS researcher field matching (cybersecurity matches computer security, etc.) | ~5,000 |
| Potential judge flexibility issues | ~300 |

The judge shows **appropriate flexibility** on:
- Diacritical marks and transliteration variants
- Numeric precision (within 1-2%)
- Name abbreviations (Bill vs William, Al vs Allen)
- CS field synonyms per the judge prompt's explicit rules (e.g., "cybersecurity" matches "computer security", "formal verification" matches "programming languages")

**Potential over-flexibility (~300 cases)**: The CS researcher field matching rule ("accept closely related or adjacent fields") creates gray areas:
- Gold "programming languages", model says "formal verification" or "static analysis" -- judged CORRECT. This is arguably appropriate given how PL and formal methods overlap.
- Gold "theoretical computer science", model says "machine learning" -- judged CORRECT for one model. This seems overly generous.
- Gold "computer networking", model says "distributed systems" -- judged CORRECT. Defensible but borderline.
- Gold "information retrieval", model says "recommender systems" -- judged CORRECT. Reasonable.

### 1.3 Problematic Probes Identified

- **IKP_T3_0587** (Highest peak in Bangladesh, gold: Keokradong, 23% accuracy): Genuinely contested fact. Modern GPS data suggests Saka Haphong/Mowdok Mual may be higher. Most frontier models answer Tazing Dong or Saka Haphong. Should be removed or gold updated.
- **IKP_T2_0347** (First transatlantic cable, gold: 1866, 49% accuracy): Ambiguous. The 1858 cable was "first" but lasted only weeks; 1866 was the first durable success. Many models answer 1858, which is a valid interpretation. Judge inconsistently marks some 1858+1866 combo answers as CORRECT.
- **IKP_T7_1301** (St. Lawrence University, gold: 2006, 0% accuracy): All 135 models answer 1856, which is the famous St. Lawrence University in Canton, NY. The 2006 gold answer likely refers to a different institution -- a name collision in the probe set, not a T7 probe.
- **IKP_T7_1307** (George Eastman Museum, gold: 1905, 0% accuracy): All models answer 1947. The museum was established in 1947; the 1905 date refers to when the Eastman House was built. Gold answer is arguably wrong or at least debatable.

---

## 2. Hallucination Patterns by Domain (T5--T7)

### 2.1 CS Researcher Probes

Across all models on T5-T7 CS probes: 31% correct, 25% wrong, 44% refusal.

| Tier | Correct | Wrong | Refusal |
|---|---|---|---|
| T3 | 73% | 13% | 15% |
| T4 | 60% | 18% | 21% |
| T5 | 33% | 25% | 41% |
| T6 | 18% | 27% | 55% |
| T7 | 7% | 33% | 61% |

**Hallucination taxonomy** (from 777 wrong CS answers across 10 focus models):

1. **Plausible-but-wrong CS subfield (74%)**: Model confidently names a wrong CS subfield. Most common hallucinated fields:
   - "formal methods" / "formal verification" (especially gemma-3-4b: 87 instances)
   - "machine learning" (64 instances from gemma-3-4b alone)
   - "reinforcement learning" as default guess for unknown researchers

2. **Name collision with non-CS person (26%)**: Model identifies someone in a completely different field:
   - "Christos Kozanitis" -- claude-opus-4.6 answers "genomics and bioinformatics"
   - "Helena Galhardas" -- multiple models say "data management" (which is CS, so this is a wrong-subfield, not a name collision)
   - True name collisions: models identify the person as being in medicine, biology, economics, etc.

**Small model pathology (gemma-3-4b)**: Has a "default answer" pattern. When uncertain, falls back to "formal methods" (87x), "machine learning" (64x), or "reinforcement learning" (28x). Produces short 1-line responses like "Formal methods." for unknown researchers. This is a confabulation pattern rather than genuine knowledge retrieval.

**Large model hallucination (gemini-2.5-pro-think)**: Despite being a strong model overall, had 148 wrong CS answers. Unlike gemma-3-4b's defaults, gemini-2.5-pro fabricates highly specific but wrong narratives -- e.g., for Po-An Tsai (gold: computer architecture), it claims "Computer Graphics, with a specific focus on neural rendering, computational photography, and view synthesis. He is known for his work on Neural Radiance Fields (NeRF)." These detailed hallucinations are more dangerous because they sound authoritative.

### 2.2 Founding Year Probes

**21,254 wrong founding-year answers** with extractable year differences across all models:

| Absolute Diff Range | Count | % |
|---|---|---|
| 0--4 years | 3,759 | 17.7% |
| 5--9 years | 3,132 | 14.7% |
| 10--19 years | 3,849 | 18.1% |
| 20--49 years | 5,385 | 25.3% |
| 50--99 years | 3,431 | 16.1% |
| 100+ years | 1,530 | 7.2% |

**Mean absolute difference: 42 years. Median: 21 years.**

**Directional bias**: Models guess an **older** year 59% of the time vs. newer 41%. This makes sense: models tend to anchor on better-known, typically earlier dates (e.g., when an institution's predecessor was founded rather than the current incarnation).

**By domain type**:
- Journals have the smallest errors (mean 23 years, median 10-12 years) -- publication years are relatively well-documented
- Places/bridges have the largest errors (mean 60-106 years, median 26-52 years) -- founding dates for obscure towns and bridges are rarely in training data
- Universities: mean 19-32 years, generally accurate when institutions are known

**Common patterns**:
- Confusing predecessor/successor dates: "Electric Power University" gold 2006, models answer 1966 (predecessor institution)
- Confusing different entities with same name: "St. Lawrence University" gold 2006, all models answer 1856 (different university)
- Anchoring to round numbers: 1900, 1950, 2000 are over-represented in wrong answers

---

## 3. Refusal Patterns

### 3.1 Refusal Rates by Model

Massive variation: from 0.1% (grok-4.20) to 100% (nemotron-ultra-253b, which refused every question).

**High refusal models** (>40%): nemotron-ultra-253b (100%), qwen3-max (65%), llama-3.2-1b (61%), llama-3.2-3b (60%), claude-3-haiku (56%), llama-3.1-8b (56%), hermes-3-405b (53%), gpt-5-nano (52%), claude-opus-4 (48%).

**Low refusal models** (<5%): grok-4.20 (0.1%), smollm2-1.7b (0.3%), gpt-4.1-mini (0.4%), gemini-2.5-pro-think (0.5%), gemini-3-flash (0.7%), gemma-3-27b (0.7%), gpt-4.1 (0.8%).

### 3.2 Calibration: Refusal vs. Accuracy

**Pearson r(refusal_rate, accuracy) = -0.427**: Models that refuse more tend to have lower raw accuracy, simply because refusals eat into the score.

**Pearson r(refusal_rate, accuracy_excluding_refusal) = +0.316**: When you exclude refusals, higher-refusal models tend to be *more accurate* on the questions they do answer. This is the calibration signal -- well-calibrated models refuse when they do not know.

**Tier-dependent calibration**:
- T1 refusal is rare across all models (most are 0-1%), with exceptions for small models (llama-3.2-1b: 5%, kimi-k2: 25%)
- T7 refusal shows the clearest separation: well-calibrated models (claude-opus-4: 94%, gpt-5-nano: 96.5%, hermes-3-405b: 94.5%) refuse nearly everything, while poorly calibrated models (gemma-3-4b: 2.5%, smollm2-1.7b: 0.5%, grok-4.20: 0%) rarely refuse.

**Key finding**: Models in the gemma family (gemma-3-4b: 2.3%, gemma-3-27b: 0.7%, gemma-2-2b: 8.1%) almost never refuse, leading to high hallucination rates on hard tiers. Similarly, gpt-4.1 (0.8%) and gpt-4.1-mini (0.4%) have very low refusal rates.

In contrast, the Claude family shows strong calibration: claude-opus-4 (48.3% refusal, 96.1% accuracy-excl-refusal), claude-sonnet-4 (47.4%, 96.5%), claude-opus-4.6 (18.9%, 89.4%). Note claude-opus-4.6 has the best absolute accuracy (72.5%) by reducing unnecessary refusals while maintaining high precision.

**Best-calibrated models** (high T7 refusal AND high accuracy-excl-refusal): claude-opus-4 (94%/96.1%), claude-sonnet-4 (94.5%/96.5%), claude-sonnet-4-think (90.5%/94.4%), gpt-5-mini-think (94%/95.1%), hermes-3-405b (94.5%/94.8%).

---

## 4. Response Length Patterns

### 4.1 Length by Verdict

Aggregated across all 136 models:
- **CORRECT** responses: mean 107 chars, median 54 chars
- **WRONG** responses: mean 127 chars, median 59 chars

Wrong answers are slightly longer on average. This is driven by models hedging, providing context, or fabricating detailed narratives.

### 4.2 Model-Specific Patterns

| Model | CORRECT mean | WRONG mean | REFUSAL mean |
|---|---|---|---|
| claude-opus-4.6 | 173 | 105 | 275 |
| gpt-4o | 64 | 62 | 14 |
| gemma-3-4b | 19 | 29 | 13 |
| deepseek-r1-think | 257 | 232 | 320 |
| llama-3.3-70b | 63 | 167 | 19 |
| grok-4.20-think | 324 | 317 | 206 |

Notable patterns:
- **gpt-4o**: Very concise across the board. Refusals are extremely short (mean 14 chars -- just "I don't know" type responses).
- **gemma-3-4b**: Ultra-terse. Correct answers average 19 chars; wrong answers 29 chars. Wrong answers are slightly longer because the model sometimes adds a brief (fabricated) explanation.
- **deepseek-r1-think / grok-4.20-think**: Long responses regardless of correctness (>250 chars for both correct and wrong). Thinking models elaborate extensively.
- **llama-3.3-70b**: Wrong answers (167 chars) are 2.6x longer than correct answers (63 chars). This model hedges and rambles when uncertain, producing longer wrong responses.
- **claude-opus-4.6**: Refusals (275 chars) are longer than correct answers (173 chars) -- the model explains why it cannot answer confidently.

### 4.3 Verbose Wrong Answers

Models that produce the most verbose wrong answers (>500 chars):
- deepseek-r1-think: 13% of wrong answers are verbose
- command-r7b: 8% verbose wrong answers
- grok-4.20-think: 8% verbose wrong answers
- gpt-4o, gemma-3-4b, gpt-3.5-turbo, gemini-2.5-pro-think: 0% verbose wrong answers

The thinking models (deepseek-r1, grok-4.20) are the most prone to generating confident-sounding but wrong long-form responses.

---

## 5. Probe Difficulty Analysis

### 5.1 Accuracy Distribution by Tier

| Tier | Mean Acc | Median Acc | Min Acc | Max Acc | % probes >90% |
|---|---|---|---|---|---|
| T1 | 96.8% | 97.8% | 64.4% | 99.3% | 96% |
| T2 | 94.7% | 96.3% | 48.9% | 99.3% | 86% |
| T3 | 77.8% | 80.0% | 23.0% | 98.5% | 16% |
| T4 | 58.9% | 60.7% | 12.6% | 99.3% | 2% |
| T5 | 33.8% | 32.6% | 5.2% | 83.0% | 0% |
| T6 | 13.4% | 9.6% | 0.7% | 68.9% | 0% |
| T7 | 4.9% | 2.2% | 0.0% | 60.7% | 0% |

The tier structure produces a **smooth difficulty gradient** from near-universal knowledge (T1: 97%) to near-zero (T7: 5%).

### 5.2 Hardest Probes (0% accuracy across all 135 models)

49 probes scored 0% accuracy. All are T7. Examples:
- Obscure bridges: "Pont 1 (sot de les Mines)" (1901), "Bjorkenheimintie iron concrete bridge" (1921)
- Obscure museums: "Small Naval Museum" (1980), "Mathaf hdab" (1989), "Fondazione Giuseppe Mozzanica" (1959)
- Obscure sports clubs: "FC Rive droite" (2002), "Schachgesellschaft Solingen" (1967)
- CS researchers: "Yufeng Huang" (HCI), "Sylvie Dujardin" (OS), "Qiansheng Rao" (computer architecture)

These represent facts truly absent from all models' training data.

### 5.3 Easiest Probes (99.3% accuracy)

18 T1 probes and 2 T2 probes achieve 99.3% (134/135 correct). These are universal knowledge: "What is the capital of France?", "What is the chemical symbol for silver?", "In what year was the Eiffel Tower completed?"

Only 1 model out of 135 gets these wrong -- likely the smallest model (smollm2-1.7b or gemma-3-1b).

### 5.4 Only Sub-50% T1/T2 Probe

**IKP_T2_0347**: "In what year was the first transatlantic cable successfully operated?" (gold: 1866, accuracy: 48.9%). This is an ambiguous probe -- 1858 is a defensible answer. Of 135 models, 63 answered 1858 (judged WRONG) and 66 answered 1866 (judged CORRECT). This probe should be reworded or removed.

### 5.5 Potential Systematic Judge Errors

No evidence of systematic judge bias was found. The 0% probes are genuinely obscure. The few probes with suspiciously low accuracy (Bangladesh highest peak, transatlantic cable) have genuinely ambiguous gold answers rather than judge errors.

---

## 6. Researcher Probe Analysis ("In computer science" Framing)

### 6.1 Name Collision Rates

All CS researcher probes use the framing "In computer science, what is the research subfield of X?" This explicitly constrains the domain. Despite this:

**26% of wrong CS answers (202 of 777 in focus models) still reference non-CS fields**. However, this is a broad definition -- many of these "name collisions" are actually models confusing CS subfields with adjacent engineering fields (e.g., "electrical engineering" for a computer architect).

**Name collision rate by model size/type**:
- Large models with high CS accuracy have few wrong answers, but when wrong, they are often name collisions: claude-opus-4.6 (83% of 6 wrong), gpt-4o (100% of 2 wrong)
- Small models have many wrong answers dominated by default guesses: gemma-3-4b (12% of 277 wrong are name collisions; 88% are plausible-but-wrong CS fields)

**Name collision rate by tier**: Relatively stable at 21-32% across T3-T7, suggesting the "In computer science" framing provides consistent benefit regardless of researcher obscurity.

### 6.2 Did "In computer science" Framing Help?

The framing clearly helps. Evidence:
- True name collisions (model identifies a non-CS person entirely) are rare: only 26% of wrong answers mention non-CS fields
- Many frontier models (claude-opus-4.6, gpt-4o, grok-4.20-think) have near-zero T3/T4 CS errors, suggesting the framing helps them avoid confusion on well-known researchers
- When name collisions do occur, they are concentrated on researchers with genuinely common names: "Yufeng Huang", "Alok Mishra", "Dan Suciu"

### 6.3 Example Responses Showing Framing Effect

**Framing working well** -- model stays in CS domain:
> Q: "In computer science, what is the research subfield of Stjepan Picek?"
> Claude-opus-4.6: "machine learning and cybersecurity/cryptography... side-channel analysis"
> Gold: "computer security" -- Judged CORRECT (cybersecurity matches)

**Framing failing** -- model identifies wrong person:
> Q: "In computer science, what is the research subfield of Christos Kozanitis?"
> Claude-opus-4.6: "genomics and bioinformatics, particularly focusing on efficient storage, querying, and analysis of large-scale genomic data"
> Gold: "computer architecture" -- Judged WRONG (model found a bioinformatician with the same name)

**Small model ignoring framing** (gemma-3-4b):
> Q: "In computer science, what is the research subfield of Philippa Gardner?"
> gemma-3-4b: "social and ethical implications of artificial intelligence, particularly concerning bias and fairness in AI systems"
> Gold: "programming languages" -- The model fabricated a plausible-sounding but entirely wrong answer, seemingly ignoring that Philippa Gardner is a well-known PL researcher.

### 6.4 CS Field Matching Flexibility

The judge prompt accepts "closely related or adjacent" fields. This creates a generous but reasonable matching zone:
- "cybersecurity" / "network security" matches "computer security" (CORRECT)
- "parallel computing" matches "computer architecture" in some cases (CORRECT)
- "formal verification" / "static analysis" matches "programming languages" (CORRECT, given the PL/formal-methods overlap)
- "distributed event processing" matches "computer networking" (CORRECT, borderline)

This flexibility is necessary because CS subfield boundaries are genuinely fuzzy, and researchers often publish across multiple areas.

---

## Summary of Key Findings

1. **Judge accuracy is high**: ~0.1-0.2% error rate. The 3-way verdict system (CORRECT/WRONG/REFUSAL) handles edge cases well. The main source of judge "errors" is models that give the correct answer but add fabricated context, triggering a WRONG verdict -- this is arguably correct behavior.

2. **4 probes should be reviewed**: Bangladesh highest peak (ambiguous), transatlantic cable (ambiguous), St. Lawrence University (name collision in probe set), George Eastman Museum (debatable gold answer).

3. **Small models hallucinate with default patterns** (gemma-3-4b says "formal methods" for unknown researchers), while **large models hallucinate with specific fabricated narratives** (gemini-2.5-pro-think invents detailed but wrong research descriptions).

4. **Founding year hallucinations average 42 years off**, with a bias toward older dates (59% of wrong guesses). Journals are easiest (median 10-12 year error), places/bridges hardest (median 26-52 year error).

5. **Calibration correlates with quality**: r=+0.316 between refusal rate and accuracy-excluding-refusal. Claude and GPT-5 families show the strongest calibration; Gemma family almost never refuses.

6. **"In computer science" framing reduces but does not eliminate name collisions**: 26% of wrong CS answers still reference non-CS fields. The framing is most effective for larger models that can use the constraint to disambiguate.

7. **Response length weakly predicts correctness**: Wrong answers average 19% longer than correct ones. Thinking models (deepseek-r1, grok-4.20) produce uniformly long responses regardless of correctness, making length uninformative for those models.
