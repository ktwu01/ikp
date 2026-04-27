# Estimating Parameter Counts of Black-Box LLMs via Incompressible Knowledge Probing

## Part I: Research Project Plan

### Phase 0: Infrastructure Setup

#### API Access via OpenRouter (or equivalent)

All models — both calibration (known size) and target (unknown size) — are
accessed through a unified API gateway. This ensures identical prompting
conditions and eliminates local hardware variation.

Calibration anchor models (known parameter counts):

| Size class   | Models (examples)                          | Access     |
|--------------|--------------------------------------------|------------|
| 0.5B-3B      | Qwen-2.5-0.5B/1.5B/3B, Phi-3-mini         | OpenRouter |
| 7B-14B       | Llama-3-8B, Qwen-2.5-7B/14B, Mistral-7B   | OpenRouter |
| 22B-72B      | Mistral-Small-22B, Qwen-2.5-32B/72B       | OpenRouter |
| 70B-123B     | Llama-3-70B, Mistral-Large-123B            | OpenRouter |
| 405B         | Llama-3-405B                               | OpenRouter |
| 671B (MoE)   | DeepSeek-V3, DeepSeek-R1                   | OpenRouter |
| ~1T (MoE)    | Kimi K2                                    | OpenRouter |

Target models (unknown parameter counts, to be estimated):

| Vendor    | Models                                          |
|-----------|-------------------------------------------------|
| OpenAI    | GPT-4o, GPT-4o-mini, GPT-4.1, o1/o3/o4-mini   |
| Anthropic | Claude Opus 4, Claude Sonnet 4, Claude Haiku 3.5|
| Google    | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 3.1 Pro|
| Others    | Grok-3, Cohere Command R+                       |

Reference estimates for sanity-checking (from leaks / Epoch AI):
- GPT-4: ~1.8T (8x220B MoE, leaked via SemiAnalysis)
- GPT-4o: ~200B (Epoch AI inference economics estimate)
- Claude Opus: ~2T (speculative)
- Gemini 3.1 Pro: possibly among the largest, >2T (speculative)


### Phase 1: Probe Generation and Iterative Calibration

The probe set is NOT constructed upfront. It is generated iteratively,
using models themselves and human expertise, then calibrated against
known-size models to verify discriminative power.

Properties of a valid probe:
- Private (never published in full, to prevent contamination)
- Incompressible (arbitrary factual associations, not derivable by reasoning)
- Tiered (spanning the full range of model capacities)
- Balanced (across geographies, domains, and entity types)

#### Tier Design

Each tier targets a different parameter-size range. The key insight is that
knowledge follows a power-law frequency distribution in web corpora, and
model capacity determines how far down the long tail a model can memorize.

**Tier 1 — Universal Knowledge (discriminates 0.1B-1B)**
- Facts any literate person knows; extremely high frequency in web corpora
- Web frequency: >100K documents in Common Crawl
- Examples: capitals of major countries, basic scientific facts (water boils
  at 100C), names of current world leaders, colors of common objects
- Expected: models >1B score ~100%; models <0.5B start failing
- ~200 probes

**Tier 2 — Common Reference Knowledge (discriminates 1B-7B)**
- Facts a well-read person knows; high frequency but more specific
- Web frequency: 10K-100K documents
- Examples: heights of famous landmarks (Eiffel Tower = 330m), populations of
  mid-sized countries, well-known research papers (Attention Is All You Need,
  2017), Nobel Prize winners from the past decade, secondary cities of major
  countries
- Expected: 7B models score well; 1B models start failing
- ~200 probes

**Tier 3 — Domain-Specific Knowledge (discriminates 7B-70B)**
- Facts a domain specialist would know; moderate frequency in corpora
- Web frequency: 1K-10K documents
- Examples: specific building heights in non-capital cities, authors of
  moderately cited papers (~500 citations), population of specific
  districts/neighborhoods, names of department heads at major universities,
  specific historical dates (not "when did WWII end" but "when was the
  Treaty of Shimonoseki signed")
- Expected: 70B models score well; 7B models struggle
- ~200 probes

**Tier 4 — Obscure Knowledge (discriminates 70B-300B)**
- Facts that exist on the web but are rarely mentioned; models in this
  range may have partial or uncertain knowledge of these facts
- Web frequency: 100-1K documents
- Examples: lesser-known research papers (<50 citations), specific scholars
  at smaller institutions, exact populations of small towns, heights of
  non-famous buildings, minor historical figures, niche technical specs
- Behavioral signature: a ~300B model may know a lesser-known scholar's
  paper titles but be unsure about details (co-authors, year, findings)
- Expected: 300B models score moderately; 70B models mostly fail
- ~200 probes

**Tier 5 — Deep Knowledge (discriminates 300B-1T)**
- Facts that very large or frontier-scale models can retrieve with
  confidence, including detailed knowledge about non-famous entities
- Web frequency: 10-100 documents
- Examples: specific details about lesser-known scholars (full name,
  affiliation, research area, specific publication details), biographical
  details of minor public figures, precise statistics for small
  administrative regions, detailed specs of niche products or systems
- Behavioral signature: a frontier model (Claude Opus, Kimi K2 at ~1T)
  not only knows a lesser-known scholar's papers but also their name,
  personal background, and research trajectory
- Expected: only the largest frontier models and open-source models
  (Kimi K2 at ~1T, DeepSeek V3 at 671B MoE) score well; 70B models
  fail almost entirely
- ~200 probes

**Tier 6 — Long-Tail Knowledge (discriminates 1T-5T)**
- Facts at the far end of what the largest models on Earth can retrieve;
  these exist on the web but are mentioned very rarely
- Web frequency: 2-10 documents
- Examples: a very obscure researcher's specific conference paper from a
  minor workshop, a local politician's committee membership in a small
  country, population of a specific village, a niche product's exact
  revision history, a minor historical event's precise date that appears
  in only a few web sources
- Behavioral signature: models in the 1T-2T range get some of these
  right; the very largest models (reported 4-5T scale) get noticeably
  more. This tier provides discrimination at the frontier of current
  model scale.
- Expected: only the very largest closed-source models (e.g., Gemini 3.1
  Pro, rumored to be among the largest deployed models) show meaningful
  accuracy; models under 1T score near zero
- ~200 probes

**Tier 7 — Extreme Long-Tail (ceiling probes)**
- Facts that exist somewhere on the web but appear on essentially a single
  page or document — below the memorization threshold of any current model
- Web frequency: 1 document (or a single web page)
- Examples: an individual known only from their personal homepage, a data
  point buried in a single municipal PDF, a detail from one obscure blog
  post, a specific line item in a single government report
- Purpose: (1) ceiling detection — no current model should score high,
  validating that probes are genuinely beyond current capacity;
  (2) future-proofing — as models grow beyond 5T, T7 probes become the
  new discrimination tier; (3) RAG detection — if a model scores well on
  T7, it almost certainly uses retrieval augmentation rather than
  parametric memory, since these facts appear too rarely for any training
  corpus to include them reliably
- Expected: all current models score near zero
- ~200 probes

**Total: ~1400 probes across 7 tiers, ~200 per tier**

Design principle: equal probe counts across tiers ensures uniform
statistical power. Each tier provides roughly the same estimation
precision for its target parameter range. With 200 probes per tier,
a 95% confidence interval on per-tier accuracy is approximately +/-7%
(assuming binomial distribution at 50% accuracy), which is sufficient
to distinguish between adjacent size classes.

#### Landmark-Based Tier Assignment (Empirical Calibration)

Tiers are NOT assigned by proxy metrics (web frequency, citation count,
sitelink count). Instead, each probe is empirically tested against 6
**landmark models** spanning from tiny to frontier. The smallest
landmark that answers correctly determines the tier.

**Landmark models (ordered smallest to largest):**

| Landmark | Model            | Params   | Tier boundary |
|----------|------------------|----------|---------------|
| L1       | Qwen 2.5 0.5B   | 0.5B     | T1 / T2       |
| L2       | Qwen 2.5 7B     | 7.6B     | T2 / T3       |
| L3       | Qwen 3 32B      | 32B      | T3 / T4       |
| L4       | Qwen 3 235B MoE | 235B     | T4 / T5       |
| L5       | Kimi K2.5        | ~1T      | T5 / T6       |
| L6       | Gemini 3.1 Pro   | Frontier | T6 / T7       |

**Tier assignment rule:**
- T1: L1 (0.5B) answers correctly
- T2: L2 (7B) answers correctly, L1 fails
- T3: L3 (32B) answers correctly, L2 fails
- T4: L4 (235B) answers correctly, L3 fails
- T5: L5 (Kimi) answers correctly, L4 fails
- T6: L6 (Gemini Pro) answers correctly, L5 fails
- T7: All 6 landmarks fail

**Monotonicity filter:** For each probe, the correctness vector across
L1→L6 must be monotonic — once a landmark answers correctly, all larger
landmarks must also answer correctly. If violated, the probe is dropped.

**Range filter by source type:**
- LLM-generated probes: only T1-T4 accepted (drop if T5+)
- Researcher probes: only T3-T7 accepted (drop if T1-T2)
- Wikidata probes: only T3-T7 accepted (drop if T1-T2)

**Target probe counts:**

| Tier | LLM | Researcher | Wikidata | Total |
|------|-----|-----------|----------|-------|
| T1   | 200 | —         | —        | 200   |
| T2   | 200 | —         | —        | 200   |
| T3   | 50  | 50        | 100      | 200   |
| T4   | 50  | 50        | 100      | 200   |
| T5   | —   | 100       | 100      | 200   |
| T6   | —   | 100       | 100      | 200   |
| T7   | —   | 100       | 100      | 200   |

*Rationale for T3-T4 split:* LLM generation is inefficient for T3+
(empirically, 82% of LLM-generated probes land in T1-T2). Reducing LLM
targets to 50 and increasing Wikidata to 100 for T3-T4 reflects the
actual yield of each source and avoids repeated oversampling.

**Iterative generation process:**
1. Generate a batch of probes (for one source type)
2. Run each probe through all 6 landmarks simultaneously
3. Apply monotonicity filter and range filter
4. Count survivors per tier — identify gaps
5. Generate more probes to fill gaps
6. Repeat until all targets met

#### Why Monotonicity Filtering is Essential

Empirical observation: approximately 10-15% of candidate probes exhibit
non-monotonic correctness patterns across the landmark ladder. Analysis
of the dropped probes reveals that non-monotonicity is NOT caused by
model bugs or evaluation errors — it is a reliable signal that the
probe's answer is ambiguous or debatable.

**Larger models know more nuance, which can cause "correct" disagreements:**

Examples from our calibration (L3 = Qwen3-32B, L6 = Gemini 3.1 Pro):

L3 (Qwen3-32B, 32B) non-monotonic examples — knows enough to second-guess:

| Probe | Gold answer | 7B answer | 32B answer | Issue |
|-------|-----------|-----------|------------|-------|
| Longest river in the world? | Nile | Nile ✓ | Amazon ✗ | Recent measurements support Amazon; debatable |
| Ottoman Empire dissolved? | 1922 | 1922 ✓ | 1923 ✗ | 1922 = sultanate abolished; 1923 = republic proclaimed |
| First circumnavigation? | Magellan | Magellan ✓ | Elcano ✗ | Magellan died en route; Elcano completed the voyage |
| Largest volcano by volume? | Mauna Loa | Mauna Loa ✓ | Tamu Massif ✗ | Newer research identified Tamu Massif as larger |
| Northernmost capital? | Reykjavik | Reykjavik ✓ | Nuuk ✗ | Nuuk is northernmost if Greenland counts as sovereign |
| First transatlantic cable? | 1858 | 1858 ✓ | 1866 ✗ | 1858 = first attempt (failed); 1866 = first permanent |
| First blood transfusion? | 1818 | 1818 ✓ | 1667 ✗ | 1667 = first animal-to-human; 1818 = first human-to-human |

L6 (Gemini 3.1 Pro, frontier) non-monotonic examples — knows even more nuance:

| Probe | Gold answer | Kimi K2.5 answer | Gemini answer | Issue |
|-------|-----------|-----------------|--------------|-------|
| Oldest university? | Bologna | Bologna ✓ | al-Qarawiyyin ✗ | Founded 859, older than Bologna (1088) by different criteria |
| Longest tunnel? | Gotthard Base | Gotthard Base ✓ | Delaware Aqueduct ✗ | Water tunnel is longer; gold implicitly means rail/road |
| Driest desert? | Atacama | Atacama ✓ | McMurdo Dry Valleys ✗ | Antarctic valleys are drier by some measurements |
| Largest volcano by volume? | Mauna Loa | Mauna Loa ✓ | Pūhāhonu ✗ | 2020 study found Pūhāhonu is larger than Mauna Loa |
| Who first observed Brownian motion? | Robert Brown | Robert Brown ✓ | Jan Ingenhousz ✗ | Ingenhousz observed it in 1785, 42 years before Brown |
| Boiling point of gold? | 2856°C | 2856°C ✓ | 2,970°C ✗ | Different sources cite different values |
| Researcher field: Otmar Hilliges | computer graphics | computer graphics ✓ | CV + HCI ✗ | More precise characterization doesn't match gold label |

In each case, the larger model gives a **defensible alternative answer**
based on more detailed knowledge (recent research, definitional nuances,
historical precision). A smaller model gives the "textbook" answer that
matches the gold standard, while a larger model knows enough to
second-guess it.

This is precisely why monotonicity filtering is critical: **a valid
incompressible knowledge probe must have an answer that is unambiguous
across all knowledge levels.** If knowing more leads to a different
answer, the probe is testing interpretation, not memorization — and it
violates the incompressibility assumption.

For researcher field probes, a similar pattern emerges: a researcher
classified as "computer graphics" may actually work at the intersection
of computer vision and HCI. A small model may guess the gold field
correctly by pattern-matching, while a large model gives a more accurate
but non-matching characterization. The monotonicity filter correctly
removes these ambiguous cases.

The ~15% drop rate is a feature, not a bug — it guarantees that every
surviving probe has a single, unambiguous answer that all model sizes
agree on when they know it.

#### Observation: LLMs Systematically Underestimate Small Model Capabilities

When generating "hard" factual probes, large LLMs exhibit a systematic
bias: they anchor on facts that are well-known within their own training
distribution, which tend to also be well-known to much smaller models.

Empirical evidence from our probe generation:

| Generation prompt | → T1 | → T2 | → T3 | → T4 | T3+T4 yield |
|-------------------|------|------|------|------|-------------|
| Unspecified       | 212  | 325  |  48  |  16  | 11%         |
| "Medium difficulty" | 28 |  59  |   8  |   1  | 9%          |
| "Hard — 32B+ only" |  5  |  62  |  12  |   6  | 21%         |
| "Very hard — 200B+ only" | 12 | 77 | 13 | 7  | 18%         |

Even when explicitly prompted to generate questions "only a 200B+
parameter model could answer," **82% of generated probes are answerable
by a 7B model or smaller.** The LLM cannot reliably predict the knowledge
boundary of models much smaller than itself.

This has methodological implications:
1. LLM-generated probes are efficient for T1-T2 but wasteful for T3+
2. Targeting harder tiers requires generating large oversamples (~5-10x)
   and filtering empirically via the landmark calibration pipeline
3. For T5-T7, corpus-grounded sources (Wikidata, researcher databases)
   are far more efficient than LLM generation, because entity obscurity
   can be controlled by external metrics (citation count, sitelink count)

This observation supports the paper's multi-source probe design: LLM
generation for easy tiers, corpus-grounded generation for hard tiers.

#### Independent Validation: Correlating Tiers with External Frequency Metrics

The landmark-based tier assignments are empirically grounded but
model-defined. To validate that tiers reflect real-world information
frequency (not just model-specific training artifacts), we correlate
tier assignments with independent, model-external frequency metrics:

**Researcher probes: Citation count (OpenAlex)**

Each researcher probe has an associated citation count from OpenAlex.
We expect a strong negative correlation: T3 researchers (known by 32B+
models) should have significantly more citations than T7 researchers
(unknown to all landmarks). This would confirm that model knowledge
capacity tracks real-world academic prominence.

Expected pattern:
- T3 researchers: ~5,000-100,000+ citations (prominent, widely known)
- T4 researchers: ~1,000-5,000 citations (moderately known)
- T5 researchers: ~200-1,000 citations (niche)
- T6 researchers: ~50-200 citations (obscure)
- T7 researchers: <50 citations (very obscure)

Analysis: plot log(citation count) vs tier, fit regression, report R².
A high R² validates that the landmark-based tiers genuinely measure
information frequency, not arbitrary model behavior.

**Wikidata probes: Google Search result count**

For each Wikidata probe, query Google Search with the full question
and record the number of returned results. This serves as a proxy for
the probe's web frequency — how often this fact appears on the internet.

Expected pattern:
- T3 probes: millions of Google results (well-documented facts)
- T4 probes: hundreds of thousands of results
- T5 probes: tens of thousands of results
- T6 probes: thousands of results
- T7 probes: hundreds or fewer results

Analysis: plot log(Google results) vs tier, fit regression, report R².
This provides a completely model-independent validation that tiers
track real-world information frequency.

Together, these two analyses strengthen the paper's core claim: model
factual capacity follows a log-linear relationship with parameters,
mediated by the power-law frequency distribution of facts on the web.

#### Probe Format

Each probe is:
- A factual question with a single, verifiable, unambiguous answer
- Expressed in 3 phrasings (to separate storage from retrieval):
  - Direct question: "What is the height of Taipei 101?"
  - Fill-in-the-blank: "Taipei 101 stands at ___ meters tall."
  - Contextual: "Among the tallest buildings in Asia, Taipei 101 reaches
    a height of ___."
- Scored as correct/incorrect (not partial credit) based on the best
  response across phrasings (measuring storage, not retrieval fluency)

#### Balance Requirements

To avoid geographic/cultural bias, each tier should have roughly equal
representation across:
- Regions: North America, Europe, East Asia, South Asia, Middle East,
  Africa, Latin America, Oceania
- Domains: people, places, publications, measurements, events, organizations
- No single category should exceed 25% of probes within a tier


#### Step 1: LLM-Generated Probes for Tiers 1-4 (within generator capacity)

A capable LLM generates candidate probes for tiers WITHIN its own
knowledge capacity (T1-T4). This is valid because:
- T1-T4 facts are within the generator model's knowledge
- The generator can produce accurate answers for these tiers
- Calibration filtering (Step 3) empirically validates each probe

**Critical methodological constraint**: LLM-generated probes are ONLY
valid for tiers where the generator model has reliable knowledge. For
T5-T7 (probes beyond the generator's capacity), using LLM recall
creates a circularity problem — see Step 2.

A capable LLM (e.g., Claude Opus, GPT-4) generates candidate probes.
The LLM is prompted with the tier definitions (parameter range, web
frequency band, example types) and asked to produce factual questions
with verified answers.

Generation prompt structure (per tier):
```
Generate 400 factual questions for Tier [N] of an Incompressible
Knowledge Probe set. Each question must:
- Have a single, objectively verifiable answer
- Be an incompressible fact (not derivable by reasoning)
- Fall within web frequency band [X-Y documents]
- Cover the following regions equally: [list]
- Cover the following domains equally: [list]

For each question, provide:
1. The question
2. The answer
3. A source URL or citation where the answer can be verified
4. An estimate of web frequency (how many pages mention this fact)
5. Two alternative phrasings of the same question
```

Why over-generate (400 candidates for 200 final probes):
- Many candidates will fail calibration (answered by models outside the
  target range, or missed by models inside it)
- Some will turn out to be derivable by reasoning (not truly incompressible)
- Filtering for balance across regions/domains will eliminate some

#### Step 2: Corpus-Grounded Probes for Tiers 5-7 (beyond generator capacity)

**Critical constraint**: Probes for T5-T7 CANNOT be generated from LLM
recall. An LLM asked to produce "very obscure facts" will either:
  (a) produce facts it already knows → too easy, not truly T5-T7
  (b) hallucinate facts that don't exist → invalid probes
This is a circularity problem: the generator can only generate facts
within its own knowledge, making it impossible to probe BEYOND its
knowledge using its own output.

**Solution: Corpus-driven discovery with frequency verification**

The pipeline for T5-T7 probe generation is:

```
Step A: SAMPLE entities from an external corpus
        (NOT from LLM recall — from structured data sources)
                    ↓
Step B: For each entity, MEASURE document frequency
        via Google result count or equivalent metric
                    ↓
Step C: ASSIGN to tier based on verified frequency
        (thresholds calibrated empirically — see below)
                    ↓
Step D: FORMULATE probe question and VERIFY answer
        against the source document (not LLM recall)
```

**Category-specific sampling sources and frequency metrics:**

| Category      | Sampling Source                    | Frequency Metric           |
|---------------|------------------------------------|-----------------------------|
| Researchers   | DBLP conference proceedings        | Citation count (Scholar/S2)  |
| Cities/places | Census databases, Wikipedia lists  | Population + Google results  |
| Buildings     | Architectural databases            | Google result count          |
| Events/dates  | Historical databases, news archives| Google result count          |
| Organizations | Government registries              | Google result count          |
| Publications  | DBLP, PubMed, arXiv               | Citation count               |

**Example: Researcher probe generation (validated methodology)**

1. Fetch ALL authors from SIGCOMM/OSDI/SOSP/NSDI 2020-2025 via DBLP API
   → This gives hundreds of real authors at ALL prominence levels,
     not just the ones an LLM would recall
2. Look up each author's citation count via Semantic Scholar API
3. Assign to tiers based on empirically calibrated thresholds:

   | Tier | Citation Range (GS) | Citation Range (S2) | Example                |
   |------|---------------------|---------------------|------------------------|
   | T3   | 2,000-10,000        | 1,000-5,000         | Justine Sherry (CMU)   |
   | T4   | 500-2,000           | 200-1,000           | Bojie Li (Pine AI)     |
   | T5   | 100-500             | 50-200              | Zekun He (Tencent)     |
   | T6   | 20-100              | 10-50               | Lakshay Rastogi        |
   | T7   | <20                 | <10                 | Very recent PhDs       |

4. For each sampled researcher, formulate probes:
   - "What university is [Name] affiliated with?"  (verify via DBLP/website)
   - "What is [Name]'s research area?"             (verify via homepage)
   - "Name a paper authored by [Name]"             (verify via DBLP)

5. Answer verification uses external sources, NEVER LLM self-knowledge

**Example: City population probe generation**

1. Fetch city lists from national census databases or Wikipedia
   category pages (e.g., "List of cities in [Country] by population")
2. Measure Google result count for "[city name] population"
3. Assign tiers: T5 = 100-1000 results, T6 = 10-100, T7 = <10
4. Formulate probe: "What is the population of [city]?"
5. Verify answer from census data source

**Tier boundary calibration**: The thresholds above are INITIAL estimates.
After collecting probe candidates, inspect a sample from each tier to
verify that the tier assignment feels correct (i.e., the fact is indeed
at the expected difficulty level for the target model size range).
Adjust thresholds as needed based on this empirical calibration.

**Chinese-language probes**: Same methodology, different corpus sources:
- Researchers: Use Chinese conference proceedings (e.g., CCF-A venues)
  and CNKI/Wanfang citation databases
- Cities: Use National Bureau of Statistics census data (国家统计局)
- Historical events: Use Chinese history databases and archives
- The same frequency verification via Baidu search result counts

This ensures Chinese probes are grounded in real Chinese-language web
data, not in an English-language LLM's (potentially biased) recall of
Chinese facts.

#### Step 3: Calibration Filtering

This is the critical step that transforms raw candidates into a
calibrated probe set. Run all candidates through representative
anchor models at each size class:

| Anchor size class | Representative model(s)         |
|-------------------|---------------------------------|
| ~1B               | Qwen-2.5-1.5B                  |
| ~7B               | Llama-3-8B, Qwen-2.5-7B        |
| ~70B              | Llama-3-70B, Qwen-2.5-72B      |
| ~200B             | Mistral-Large-123B (closest)    |
| ~400B             | Llama-3-405B                    |
| ~670B             | DeepSeek-V3                     |
| ~1T               | Kimi K2                         |

For each candidate probe, record which anchor models answer correctly.
Then filter:

**Tier N probe is valid if:**
- Models BELOW Tier N's range mostly fail (<20% accuracy)
- Models WITHIN Tier N's range show intermediate accuracy (20-80%)
- Models ABOVE Tier N's range mostly succeed (>80% accuracy)

Probes that don't show this sigmoid pattern are discarded:
- If all models get it right → too easy, discard
- If all models get it wrong → too hard (or bad question), discard
- If small models get it right but large ones don't → not measuring
  capacity, likely a training data artifact, discard

Select the best ~200 probes per tier from the filtered candidates,
prioritizing geographic and domain balance.

#### Step 4: Frequency Verification

For each surviving probe, independently estimate web frequency:
- Use Common Crawl index counts or Google search result estimates
- Verify that tier assignments align with actual frequency:

| Tier | Expected web frequency    |
|------|---------------------------|
| T1   | >100K documents           |
| T2   | 10K-100K documents        |
| T3   | 1K-10K documents          |
| T4   | 100-1K documents          |
| T5   | 10-100 documents          |
| T6   | 2-10 documents            |
| T7   | ~1 document               |

Probes whose actual frequency doesn't match their tier assignment
are reassigned to the correct tier or discarded.

**Total: ~1400 validated probes across 7 tiers, ~200 per tier**


### Phase 2: Full Calibration and Curve Fitting

#### Scaling Model: Per-Tier Sigmoid, Aggregate Log-Linear

The theoretical expectation, grounded in Kandpal's R^2=0.98 finding
and the compression theory framework:

**Per-tier accuracy follows a logistic (sigmoid) function of log(N):**

```
T_i(N) = L_i / (1 + exp(-k_i * (log(N) - m_i)))
```

Where for tier i:
- N = parameter count
- L_i = saturation accuracy (ceiling, ideally ~1.0 for well-designed probes)
- k_i = steepness (how sharply the transition occurs)
- m_i = midpoint (the log(N) at which accuracy = L_i/2)

The midpoint m_i should correspond to the tier's target parameter range:
- m_1 ≈ log(0.5B), m_2 ≈ log(4B), m_3 ≈ log(20B), m_4 ≈ log(150B),
  m_5 ≈ log(500B), m_6 ≈ log(2T), m_7 ≈ log(>10T)

**Aggregate accuracy is the mean of per-tier accuracies:**

```
A(N) = (1/7) * Σ T_i(N)
```

Since the sum of shifted sigmoids approximates a straight line over a
wide range, aggregate accuracy should be approximately log-linear in N,
consistent with Kandpal's observation:

```
A(N) ≈ a * log(N) + b       (R^2 expected ≈ 0.95+)
```

This is the key relationship: **the x-axis is log(parameters), the
y-axis is accuracy, and the relationship is approximately linear.**
This matches the scaling trends observed in recent papers on fact
learning (Kandpal et al. ICML 2023).

#### Why This Distribution?

The log-linear aggregate relationship arises because:

1. Web knowledge follows a **power-law frequency distribution** (Zipf's law)
2. Model capacity determines a **frequency cutoff**: facts above the cutoff
   are memorized, facts below it are not
3. As parameters grow, the cutoff moves down the power-law tail
4. The number of facts above a power-law cutoff grows logarithmically
   with the cutoff position
5. Therefore: accuracy (fraction of probes above cutoff) scales as
   log(capacity) ≈ log(parameters)

This is NOT an exponential distribution. It is a **log-linear** relationship:
accuracy is linear in log(params). Equivalently, each 10x increase in
parameters yields roughly the same absolute increase in accuracy.

#### Running the Calibration

For each of the ~30 calibration models (known sizes 0.5B to ~1T):
1. Run all ~1400 probes in all 3 phrasings (~4200 API calls)
2. Score per-probe: correct if best-of-3 phrasings is correct
3. Record 7-dimensional accuracy vector: [T1%, T2%, ..., T7%]
4. Record aggregate accuracy

Fit two calibration functions:

**(a) Simple log-linear (aggregate):**
```
log(N) = α * A(N) + β
```
where A(N) is aggregate accuracy. Report R^2.

**(b) Per-tier logistic regression (7-dimensional):**
Fit logistic curves T_i(N) for each tier. Then invert:
```
log(N̂) = weighted_median({ (logit(T_i) + b_i) / a_i })
```
where the weight for each tier reflects how much information it
provides about the model's size (tiers where the model scores
near 0% or 100% contribute little; tiers near 50% contribute most).

The per-tier estimator should be more precise than aggregate because
it uses the full shape of the accuracy profile, not just the mean.

#### Diagnostic Checks

- Plot all 30 calibration points on the log(N) vs accuracy curve
- Inspect residuals: are they random, or do specific model families
  systematically deviate?
- Check within-family consistency: do Qwen models at 0.5B, 7B, 72B
  fall on the same curve as Llama at 8B, 70B, 405B?
- MoE analysis: DeepSeek V3 (671B total, ~37B active) and Mixtral
  8x7B (47B total, ~13B active). Does probe accuracy correlate with
  total params or active params? Expected: total params.


### Phase 3: Validation

#### Leave-One-Out Cross-Validation

For each of the ~30 calibration models:
1. Remove it from the calibration data
2. Fit the curve on remaining models
3. Predict the held-out model's parameter count from its accuracy
4. Compare predicted vs actual parameter count
5. Report prediction error as a multiplicative factor (e.g., 1.5x means
   the estimate is off by 50%)

Target: median prediction error < 2x (better than Epoch AI's
acknowledged 2x uncertainty for inference economics).

#### Cross-Family Transfer

1. Fit calibration on Qwen family only (0.5B-72B)
2. Predict Llama family sizes → compare to actual
3. Repeat for all family pairs
4. Report transfer error matrix
5. Key question: does the curve generalize?

#### Temporal Stability Test (Critical Experiment)

Test the incompressibility claim directly:
- Llama-2-7B vs Llama-3-8B: same approximate size, different generation
- The incompressibility argument predicts similar IKP scores (both have
  ~7-8B params for factual storage), even though Llama-3 scores much
  higher on reasoning benchmarks (procedural improvement)
- If confirmed: this is strong evidence that IKPs measure capacity,
  not capability, validating the core theoretical claim
- If refuted: the procedural/factual partition is less clean than
  theorized, and this must be discussed honestly

#### Comparison with Inference Economics

- Collect Epoch AI's parameter estimates for all target models
- For models where both IKP and Epoch AI estimates exist, compare:
  - Do they agree? (mutual validation)
  - Where they disagree, which has supporting evidence?
  - Report uncertainty bands for both methods


### Phase 4: Frontier Model Estimation

#### Running the Probes

Run the full 1400-probe set against all target models via API.
Record per-tier and aggregate accuracy for each.

#### Estimation

Apply the calibration curve to estimate parameter counts:
- Report point estimates from both aggregate and per-tier estimators
- Report 95% confidence intervals (from bootstrap on calibration fit)
- Compare to known leaks (GPT-4 ~1.8T MoE) and Epoch AI estimates

#### Key Analyses

**Frontier differentiation (the main result):**
- T4 probes (70B-300B): Do GPT-4o-mini and Claude Haiku separate here?
- T5 probes (300B-1T): Do Claude Opus and GPT-4o separate here?
- T6 probes (1T-5T): Does Gemini 3.1 Pro separate from the pack?
  If it scores meaningfully on T6 while others don't, this suggests
  it is significantly larger.

**Reasoning model analysis:**
- o1/o3 vs GPT-4o: same base weights, different reasoning
- IKP scores should be similar (reasoning doesn't add factual storage)
- If confirmed: IKPs are robust to reasoning fine-tuning

**Distillation detection:**
- If a model reasons like a 400B model but scores on IKPs like a 30B
  model → it's likely distilled
- Practical application for API auditing and procurement

#### T7 Sanity Check

All models should score near zero on T7. If any model scores
significantly above zero:
- Suspect retrieval augmentation (RAG) rather than parametric storage
- Check response latency (RAG typically adds latency)
- Flag and report separately


### Phase 5: Robustness and Ablation Studies

1. **Tier ablation**: Estimate using only T3-T6 probes (the core
   discriminative tiers). Does estimation degrade gracefully when
   floor (T1-T2) and ceiling (T7) tiers are removed?

2. **Probe count ablation**: Subsample probes within each tier.
   What's the minimum count needed for <2x estimation error?
   (Important for cost: 1400 probes * 3 phrasings * API cost adds up)

3. **Contamination simulation**: Publish a subset of T3-T5 probes,
   wait for model updates, re-test. Does accuracy inflate?
   (Validates the need for private probes.)

4. **Geographic ablation**: Restrict probes to one region. How much
   does bias affect estimates? (Quantifies the balance requirement.)

5. **Frequency correlation**: For each model, plot per-probe accuracy
   as a function of that probe's web frequency. Verify that larger
   models push the accuracy frontier further down the frequency axis.

6. **Chinese-language probes**: Generate ~90 probes in Chinese (easy/
   medium/hard) about Chinese geography, history, academia, and culture.
   Test whether Chinese-origin models (Qwen, DeepSeek) outperform
   Western models on Chinese-language probes vs English-language probes.
   Key questions:
   - Does the "home advantage" hypothesis hold when probes are in
     the native language?
   - How much does probe language (English vs Chinese) affect parameter
     estimates? Does Chinese language reduce or amplify geographic bias?
   - Can we construct language-balanced probes for fairer estimation
     across model families?
   Early finding from English probes: Chinese models do NOT strongly
   outperform on East Asian content when probed in English (+3.4pp),
   but the effect may be larger in Chinese.

7. **Researcher knowledge probes (motivating case study)**: This is
   the observation that inspired the entire IKP paper. Bojie Li noticed
   that frontier models have memorized individual researchers "inside
   the weights" — models know names, affiliations, research areas, and
   specific publications of established CS researchers, but not
   less prominent ones. The depth of this knowledge correlates with
   model capacity:
     - GPT-5.4 knows the fewest details about researchers
     - Claude Sonnet knows more
     - Claude Opus knows even more
     - Gemini 3.1 Pro is the most knowledgeable model
   This spectrum of knowledge depth gave the intuition that factual
   probes could estimate parameter counts. The experiment:
   - Generate probes about ~50 real CS systems/networking researchers
     at 3 prominence levels (well-known, established, emerging)
   - Test across all model sizes to show that researcher knowledge
     tracks with parameter count
   - Show concrete examples: "Gemini 3.1 Pro knows [emerging researcher]'s
     specific paper titles, while GPT-4o only knows their affiliation"
   - This serves as both a validation AND an intuitive motivating
     example for the Introduction (Section 1.1)

### Phase 6: Knowledge Fingerprinting and Distillation Detection

The IKP framework measures total parametric storage capacity, but the
**specific set of rare facts** a model knows constitutes a unique
fingerprint of its training process. Two independently trained models
will share common knowledge (T1-T4) but diverge on which particular
rare facts (T5-T7) they memorize. Distillation transfers not just
general capability but this specific knowledge fingerprint, creating a
detectable provenance signal.

#### 6.1 Theoretical Basis: Knowledge Fingerprints

Consider a probe set of 600 rare-knowledge probes (T5-T7, 200 each).
Each model produces a binary knowledge vector K ∈ {0,1}^600 indicating
which probes it answers correctly. This vector is the model's
**knowledge fingerprint**.

Key properties:
- **Uniqueness**: For rare facts, the specific subset a model memorizes
  depends on training data mix, ordering, curriculum, and random seed.
  Two independently trained models of the same size should have
  different fingerprints (different specific T5-T7 facts known).
- **Heritability**: Distillation trains the student on the teacher's
  outputs. The student inherits the teacher's specific knowledge
  (and specific errors), not random knowledge at the same difficulty
  level. Therefore: K_student ⊂ K_teacher (approximately).
- **Asymmetry**: Distillation can only transfer knowledge the teacher
  has. If model B knows rare facts that model A does not, B was not
  purely distilled from A. This rules out distillation definitively.

#### 6.2 Statistical Framework: Overlap Analysis

**Null hypothesis (independent training):**
Two models trained independently on similar web corpora. For T6 probes:

```
Model A: knows |K_A| = 60 out of 200 probes (30%)
Model B: knows |K_B| = 10 out of 200 probes (5%)
Expected overlap under H0: |K_A ∩ K_B| = 60 × 10 / 200 = 3 probes
Distribution: hypergeometric(N=200, K=60, n=10)
```

**Distillation hypothesis (B derived from A):**
B's knowledge is a subset of A's knowledge:

```
Expected overlap: |K_A ∩ K_B| ≈ |K_B| = 10 probes
```

**Test statistic**: observed overlap vs. expected under H0.
For the example above: observing 10 overlap vs. 3 expected gives
p < 0.001 (hypergeometric test). Even small knowledge sets produce
high statistical power because rare facts have low baseline overlap.

**For T7 probes (most diagnostic)**:

```
Model A: knows 10 out of 200 (5%)
Model B: knows 2 out of 200 (1%)
Expected overlap under H0: 10 × 2 / 200 = 0.1 probes
P(both of B's correct are in A's 10): C(10,2)/C(200,2) ≈ 0.2%
```

A single T7 overlap is suspicious; two is near-conclusive.

#### 6.3 Knowledge Similarity Score

Define a composite fingerprint similarity metric between models A and B:

```
FPS(A, B) = Σ_t  w_t × [ |K_A^t ∩ K_B^t| / E_H0[|K_A^t ∩ K_B^t|] ]
```

Where:
- t ∈ {T5, T6, T7} (only rare-knowledge tiers contribute signal)
- w_t = tier weight (higher for rarer tiers; suggested: w_T5=1, w_T6=2, w_T7=4)
- Numerator = observed overlap on tier t
- Denominator = expected overlap under independence

Interpretation:
- FPS ≈ 1.0: independent training (overlap matches random expectation)
- FPS >> 1.0: excess overlap → evidence of derivation
- Threshold for flagging: FPS > 3.0 (3× more overlap than expected)

#### 6.4 Shared Hallucination Analysis

Beyond correct-answer overlap, **shared wrong answers** are an even
stronger provenance signal. When both models answer a probe incorrectly,
do they produce the same specific wrong answer?

Example: "What year did [obscure researcher] publish [specific paper]?"
- Ground truth: 2021
- Model A answers: 2019
- Model B answers: 2019
- Probability of the same specific wrong year under independence: ~1/10
  (assuming ~10 plausible years)
- If this pattern repeats across many probes: near-certain derivation

**Hallucination similarity score**:

```
HSS(A, B) = |{p : A(p) = B(p) ≠ gold(p)}| / E[shared_wrong_under_H0]
```

Where the denominator accounts for the entropy of the wrong-answer
distribution. For numeric answers (years, populations), the space of
plausible wrong answers is large, making shared errors highly diagnostic.
For categorical answers (names), the space is smaller but still
informative.

**Combined provenance score**:

```
Provenance(A, B) = α × FPS(A, B) + β × HSS(A, B)
```

With α, β calibrated on known distillation pairs (e.g., DeepSeek-R1
distilled variants, Llama fine-tuned variants like Hermes-3).

#### 6.5 Experimental Design

**Known distillation pairs (positive controls):**
These are model pairs where derivation is publicly known, used to
validate that the fingerprint method detects known relationships:

| Student | Teacher/Base | Relationship |
|---------|-------------|--------------|
| DeepSeek-R1 | DeepSeek-V3 | Reasoning fine-tune of same base |
| DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1 + Qwen-2.5-Math-1.5B | R1 knowledge distilled into Qwen base |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1 + Qwen-2.5-7B | R1 knowledge distilled into Qwen base |
| DeepSeek-R1-Distill-Qwen-14B | DeepSeek-R1 + Qwen-2.5-14B | R1 knowledge distilled into Qwen base |
| DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1 + Qwen-2.5-32B | R1 knowledge distilled into Qwen base |
| DeepSeek-R1-Distill-Llama-8B | DeepSeek-R1 + Llama-3.1-8B | R1 knowledge distilled into Llama base |
| DeepSeek-R1-Distill-Llama-70B | DeepSeek-R1 + Llama-3.3-70B | R1 knowledge distilled into Llama base |
| DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528 + Qwen3-8B | Newer R1 generation distilled into Qwen3 |
| Hermes-3-405B | Llama-3.1-405B | SFT fine-tune (same weights) |
| Hermes-4-405B | Llama-3.1-405B | SFT fine-tune with ~60B tokens |
| QwQ-32B | Qwen-2.5-32B | Reasoning fine-tune of same base |
| Nemotron-70B | Llama-3.1-70B | NVIDIA RLHF fine-tune |
| Nemotron-Super-49B | Llama-3.3-70B | NAS pruning + distillation (70B→49B) |
| Llama-3.3-70B | Llama-3.1-70B | Same family, different generation |
| Gemma-3-27B | Gemma-2-27B | Same family, different generation |

The DeepSeek-R1 distilled variants are the most valuable positive
controls because each has TWO known parents: the R1 teacher (knowledge
source) and the Qwen/Llama base (architecture). The fingerprint analysis
can test whether the distilled model's knowledge profile looks more
like its teacher (R1) or its base model (Qwen/Llama). If the R1
distilled variants show higher FPS with R1 than with their base models,
this confirms that knowledge distillation transfers the teacher's
knowledge fingerprint.

The Nemotron-Super-49B is especially interesting: it was pruned from
70B to 49B parameters via NAS, then distilled. Does pruning preserve
the knowledge fingerprint, or does the 30% parameter reduction change
which facts the model retains?

**Known independent pairs (negative controls):**
Models from different families/vendors trained independently:

| Model A | Model B | Expected |
|---------|---------|----------|
| Llama-3.1-70B | Qwen-2.5-72B | Independent (different vendors) |
| DeepSeek-V3 | Mistral-Large | Independent |
| Gemma-2-27B | Phi-4 | Independent |
| R1-Distill-Llama-8B | Qwen-2.5-7B | R1 distill vs independent Qwen (cross-check) |
| R1-Distill-Qwen-32B | Llama-3.1-70B | R1 distill vs independent Llama (cross-check) |

**Target analysis (unknown provenance):**
Apply fingerprint matching to all target models against all calibration
models. Flag any target model whose fingerprint similarity exceeds the
threshold. Key questions:
- Do any closed-source models show fingerprint overlap with specific
  open-source models? (Would suggest training on open-source outputs)
- Do reasoning models (o3, o4-mini) share fingerprints with their
  suspected base models (GPT-4o)?
- Do models within the same vendor share fingerprints across
  generations (GPT-4o vs GPT-4.1)?

#### 6.6 Extended Phrasing Probes for Fingerprint Resolution

The standard IKP pipeline uses 3 phrasings per probe with best-of-3
scoring. For fingerprint analysis, we need additional data:

**Per-phrasing correctness vectors**: Instead of collapsing to best-of-3,
retain the 3-dimensional binary vector [direct, fill_blank, contextual]
per probe. Two models derived from the same base may show correlated
per-phrasing patterns (both fail on fill-in-the-blank but succeed on
direct question for the same probe).

**Extended phrasings for a fingerprint subset**: For a subset of ~100
probes from T5-T7, generate 7 additional phrasings (10 total per probe)
to increase fingerprint resolution. More phrasings per probe means more
bits in the fingerprint vector, increasing the power to distinguish
independent from derived models.

Phrasing variants for extended set:
1. Direct question (existing)
2. Fill-in-the-blank (existing)
3. Contextual (existing)
4. Reversed question (ask about the entity given the attribute value)
5. Multiple-choice (correct answer + 3 distractors)
6. True/false statement
7. Negated question ("Is it true that X is NOT Y?")
8. Embedded in a longer context
9. Different formality register (casual vs formal)
10. Paraphrased with synonyms

#### 6.7 Datasets Required

**Fingerprint probe set**: ~100 probes from T5-T7 with 10 phrasings each
(1000 API calls per model). These are a curated subset of the main
probe set, selected for maximum fingerprint resolution:
- Probes where calibration models show high inter-model variance
  (some know it, others don't) — these are the most discriminating
- Probes with high-entropy wrong-answer distributions (numeric
  answers, specific dates/names) — these support hallucination analysis
- Balanced across T5, T6, T7 to capture fingerprints at different
  depth levels

**Wrong-answer capture**: The current pipeline records model responses
as text. For fingerprint analysis, we need to retain the actual wrong
answers (not just correct/incorrect), as shared hallucinations are a
key signal. This requires a minor pipeline modification: store the
raw response text even for incorrect answers (already done in
_responses.json files).

**Pairwise comparison matrix**: For N models, compute N×(N-1)/2
fingerprint similarity scores. With ~50 models (32 calibration + 17
target), this is ~1225 pairs — computationally trivial once the probe
results are collected.

#### 6.8 Limitations and Confounds

- **Shared training data**: Models trained on the same web corpus
  (e.g., Common Crawl) will share some rare knowledge even without
  distillation. Control for this by comparing observed overlap to
  within-family baselines (e.g., Llama-3-8B vs Llama-3-70B share
  training data but are not distilled from each other).
- **Fine-tuning can add/remove knowledge**: A distilled model that
  undergoes substantial additional fine-tuning may gain new knowledge
  or lose inherited knowledge, weakening the fingerprint signal. The
  positive controls (Hermes-3, QwQ) help calibrate how much fine-tuning
  degrades the signal.
- **Contamination**: If probe answers appear in fine-tuning data, a
  model may appear to share knowledge with another model when it
  actually learned the facts independently. The private nature of
  probes mitigates this.
- **Small knowledge sets**: If both models know very few T5-T7 facts
  (e.g., both know 2 out of 200), statistical power is low. The test
  is most powerful when at least one model has substantial rare
  knowledge.


### Additions to Paper Outline

#### 1.0 Motivating Example (new, before 1.1)

Open the paper with the researcher knowledge observation:
  "When we ask frontier LLMs about individual computer science
  researchers, a striking pattern emerges. The largest models can
  name a researcher's specific papers, co-authors, and research
  trajectory — while smaller models only know their affiliation or
  nothing at all. This spectrum of knowledge depth, we argue, is not
  a coincidence: it directly reflects the information-theoretic
  capacity of each model's parameters."

This provides an immediate, visceral hook before the formal problem
statement. The reader can try it themselves with their own name.

#### 4.8 Linguistic Bias Analysis (new subsection in Methodology)

Describe the Chinese-language probe experiment:
  - Motivation: probes in English may favor Western training data
  - Experimental design: same difficulty levels, Chinese language,
    Chinese-specific content
  - Comparison: Chinese-origin vs Western models on Chinese probes
  - Finding: whether linguistic bias is larger than geographic bias
    observed in English probes

#### 3.7 Knowledge Fingerprints and Distillation Detection (new subsection)

Extend the theoretical framework to model provenance:
- The specific set of rare facts (T5-T7) a model knows constitutes a
  **knowledge fingerprint** — a binary vector K ∈ {0,1}^n over rare probes
- Independently trained models have low fingerprint overlap (governed
  by hypergeometric distribution under the null)
- Distillation transfers the teacher's specific knowledge fingerprint
  to the student: K_student ⊂ K_teacher (approximately)
- This creates a detectable provenance signal, formalized as a
  fingerprint similarity score FPS(A,B) that compares observed overlap
  to expected overlap under independence
- Shared hallucinations (same wrong answers) provide an additional
  signal with even higher specificity: the probability of two
  independent models producing the same specific wrong answer is
  ~1/|answer_space|, which is very small for numeric or named-entity
  answers

#### 4.9 Distillation Detection Methodology (new subsection)

Describe the fingerprint-based distillation detection pipeline:
- **Knowledge overlap test**: For each model pair, compute per-tier
  overlap on T5-T7 probes; test against hypergeometric null using
  Fisher's exact test or equivalent
- **Hallucination similarity**: Compare actual wrong answers on shared
  failures; score shared-error rate against chance expectation
- **Combined provenance score**: Weighted combination of overlap and
  hallucination signals across tiers
- **Validation on known pairs**: Calibrate thresholds using known
  fine-tuning and distillation relationships (DeepSeek-R1/V3,
  Hermes-3/Llama-3.1, QwQ/Qwen-2.5)
- **Extended phrasings**: For a fingerprint subset of ~100 T5-T7
  probes, generate 10 phrasings instead of 3 for higher-resolution
  fingerprint vectors

#### 6.8 Case Study: Researcher Knowledge Spectrum (new subsection in Results)

Present the researcher probe results:
  - Show that researcher recognition tracks with model size
  - Show the knowledge depth spectrum (affiliation → research area →
    specific papers → co-authors → personal details)
  - This is the concrete instantiation of the IKP principle that
    first motivated the study

#### 6.9 Knowledge Fingerprinting and Distillation Detection (new subsection)

Present the fingerprint analysis results:
  - **Validation on known pairs**: Show that known fine-tuned/distilled
    pairs (DeepSeek-R1/V3, Hermes-3/Llama-405B, QwQ/Qwen-2.5-32B)
    produce high FPS scores, while known independent pairs (cross-family)
    produce FPS ≈ 1.0. Report sensitivity and specificity.
  - **Pairwise similarity matrix**: Heatmap of FPS(A,B) across all
    model pairs. Cluster structure should reveal model families.
    Unexpected high-similarity pairs are the key finding.
  - **Hallucination analysis**: Show examples of shared specific wrong
    answers between related models; quantify HSS for known and unknown
    pairs.
  - **Provenance inference for target models**: For each closed-source
    target model, report which (if any) open-source model's fingerprint
    it most closely matches. Flag models with suspiciously high overlap
    with specific open-source families.
  - **Cross-generation analysis**: Do GPT-4o and GPT-4.1 share a
    fingerprint? Do Claude Opus 4 and Claude Sonnet 4? This reveals
    whether different model tiers from the same vendor share weights
    or training data.
  - **Reasoning model analysis**: Do o3/o4-mini share GPT-4o's
    fingerprint? If yes, confirms same base model with reasoning
    fine-tuning. If no, suggests a different base.

#### 7.3 Distillation Detection: Implications (new subsection in Discussion)

  - Practical value: unauthorized distillation detection, API model
    substitution, open-source model provenance verification
  - Comparison to existing methods: LLMmap identifies known models
    by fingerprinting (~42 models); knowledge fingerprinting detects
    unknown derivation relationships, a harder problem
  - Limitations: shared training data creates baseline overlap;
    extensive fine-tuning can degrade the signal; contamination
    creates false positives
  - Privacy implications: knowledge fingerprints are a form of model
    identification — discuss dual-use concerns


---


## Part II: Paper Outline (Motivation and Design)

### Title

"Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts
via Information-Theoretic Factual Capacity"

### Abstract (sketch)

- Problem: Closed-source frontier LLMs do not disclose parameter counts.
  Current estimation methods rely on inference economics (token speed, API
  pricing), which are noisy (2x+ uncertainty) and depend on external factors
  (hardware, batching, quantization) rather than intrinsic model properties.
- Insight: Factual knowledge (specific names, measurements, paper titles) is
  incompressible — it has an information-theoretic storage lower bound that
  architectural improvements cannot circumvent. Unlike reasoning or linguistic
  capability, factual capacity is tightly coupled to parameter count.
- Method: We construct Incompressible Knowledge Probes (IKPs) — a tiered,
  geographically balanced, private set of factual questions at varying obscurity
  levels. We calibrate a log-linear mapping from IKP accuracy to parameter count
  using N open-source models across K families, then apply it to estimate sizes
  of closed-source frontier models.
- Results: Calibration on 62 open models (1B-1040B) yields R^2 = 0.893.
  Each 10x increase in parameters adds ~15.2 pp of IKP accuracy. For MoE
  models, total parameters predict knowledge (R^2=0.84) while active
  parameters do not (R^2=0.43). Validated against GPT-4's leaked 1.8T
  architecture (predicted 1.2T, within expected error from training data
  quality differences). Frontier estimates: Gemini 3.1 Pro ~22.6T
  equivalent, Claude Opus 4.6 ~2.5T, GPT-5 ~3.2T. Secondary finding:
  LLM recognition of researchers correlates moderately with citations
  (rho=0.575) but name uniqueness and subfield web visibility are
  equally important predictors.


### 1. Introduction

#### 1.1 The Problem: Unknown Parameters of Frontier Models

- Frontier labs (OpenAI, Anthropic, Google) do not disclose parameter counts
- Parameter count matters for: AI governance/regulation (EU AI Act references
  compute thresholds), procurement decisions, scientific understanding of
  scaling, detecting model substitution in APIs
- Current estimation relies on inference economics (Epoch AI), which is
  indirect and uncertain
- Cite: Epoch AI parameter count methodology, Cai et al. 2025 (model
  substitution auditing), EU AI Act compute thresholds

#### 1.2 The Opportunity: Factual Knowledge as a Capacity Probe

- Language models store factual knowledge in their parameters (cite Allen-Zhu
  & Li ICLR 2025: 2 bits/parameter)
- Factual accuracy on rare knowledge scales log-linearly with model size
  (cite Kandpal et al. ICML 2023: R^2 = 0.98)
- These two results together imply: if you can measure how much a model knows,
  you can estimate how many parameters it has
- The inverse problem has never been attempted

#### 1.3 The Key Insight: Incompressibility

- The "Densing Law" (Nature MI 2025) shows capability per parameter doubles
  every ~3.5 months — meaning benchmark scores are an unreliable proxy for
  size because procedural capability can be compressed into fewer parameters
- BUT factual knowledge is incompressible: "the population of Liechtenstein
  is ~39,000" cannot be derived from any other fact. It must be stored.
  Shannon entropy gives a lower bound on the bits required.
- Therefore, factual probes measure a resource (parametric storage) that is
  robust to architectural improvements, unlike reasoning or linguistic
  benchmarks
- This is the core theoretical contribution of this paper

#### 1.4 Contributions

1. We introduce Incompressible Knowledge Probes (IKPs), a tiered probe
   methodology designed to estimate LLM parameter counts from black-box access
2. We provide a theoretical argument, grounded in information theory, for why
   factual capacity is a more robust size estimator than benchmark performance
3. We calibrate and validate the approach on 62 open-source models across 15+
   families (R^2 = 0.893), and show that total parameters (not active) predict
   MoE knowledge capacity
4. We provide parameter count estimates for 50+ closed-source frontier models,
   validated against GPT-4's leaked 1.8T architecture
5. We demonstrate knowledge fingerprinting for distillation detection, showing
   that distilled models inherit measurable knowledge patterns from both teacher
   and base models
6. We reveal that LLM recognition of researchers is a joint function of citation
   impact, name uniqueness, and subfield web visibility — challenging the
   assumption that bibliometric impact alone determines what LLMs "know"


### 2. Background and Related Work

#### 2.1 Knowledge Capacity of Language Models

- Allen-Zhu & Li (ICLR 2025): 2 bits of knowledge per parameter, derived
  from synthetic factual tuples. Establishes the theoretical link between
  parameters and factual storage.
- "Understanding LLM Behaviors via Compression" (2025): Kolmogorov Structure
  Function applied to LLM scaling. Models compress compressible patterns first,
  then progressively encode rarer incompressible knowledge as capacity permits.
- Chang et al. (NeurIPS 2024): "How Do LLMs Acquire Factual Knowledge During
  Pretraining?" — 7B models have significantly greater factual knowledge
  acquisition effectivity than 1B models. Model size qualitatively changes
  knowledge acquisition.

#### 2.2 Long-Tail Knowledge and Scaling

- Kandpal et al. (ICML 2023): accuracy on rare facts scales log-linearly with
  model size (R^2 = 0.98 in BLOOM family). Larger models ARE better at
  long-tail knowledge. The "orders of magnitude" extrapolation concerns
  reaching human-level performance, not the existence of a discriminative
  signal.
- Tirumala et al. (2022): larger models mitigate loss of long-tail knowledge
  rather than compressing it into noisy representations.
- "Long-Tail Knowledge in LLMs: Taxonomy" (2026 survey): comprehensive
  treatment confirming long-tail failures persist but scale helps.

#### 2.3 Knowledge Overshadowing and Probe Design

- "The Law of Knowledge Overshadowing" (ACL 2025): popular knowledge
  suppresses less popular knowledge. Hallucination rate increases log-linearly
  with knowledge popularity. Implication: obscure probes are more
  discriminating because they are less susceptible to "lucky guessing" from
  popular-knowledge interference.

#### 2.4 Parameter Specialization

- Hong et al. (NeurIPS 2025): "The Rise of Parameter Specialization for
  Knowledge Storage." Stronger models develop more specialized parameter
  vectors for factual knowledge (measured by PSS). The ratio of parameters
  dedicated to factual knowledge trends predictably with model quality,
  partially addressing the P_factual/P_total ratio concern.

#### 2.5 Existing Model Size Estimation

- Epoch AI inference economics: token speed + API pricing + hardware modeling.
  Acknowledged 2x uncertainty, depends on external factors.
- LLMmap (USENIX Security 2025): fingerprinting identifies WHICH model (42
  versions, >95% accuracy, 8 queries), but does not estimate size of unknown
  models.
- Cai et al. (2025): model substitution auditing showed text-output
  statistical tests fail at ~50% for detecting quantization. BUT this is a
  different task (detecting subtle numerical changes, not 10x size differences).
- TRUCE (2024): private benchmarking via confidential computing. Addresses
  contamination but not parameter estimation.
- PEEK (2025): proxy embeddings estimate what knowledge an LLM has, but do
  not connect this to parameter count estimation.

#### 2.6 The Densing Law and Why Benchmarks Fail

- Densing Law (Nature MI 2025): capability density doubles every ~3.5 months.
- Implication: a 7B model from 2026 may match a 70B model from 2023 on
  reasoning benchmarks, making benchmark scores useless as size estimators.
- Our argument: this applies to compressible capabilities only. Incompressible
  factual storage cannot "dense" — it obeys a hard information-theoretic lower
  bound. This is what makes IKPs robust where benchmarks are not.


### 3. Theoretical Framework

#### 3.1 Factual Knowledge as Incompressible Information

- Define "incompressible factual knowledge": a factual association (entity,
  attribute, value) where the value cannot be derived, computed, or inferred
  from other known facts or from structural regularities in language
- Formalize using Kolmogorov complexity: K(fact) >= H(value|entity, attribute)
  where H is Shannon entropy of the value given the entity-attribute pair
- Examples: the population of a city has high conditional entropy (could be
  any number); the capital of a country has moderate entropy (constrained to
  city names but still arbitrary)

#### 3.2 The Capacity Bound

- From Allen-Zhu: a model with N parameters stores at most ~2N bits of
  factual knowledge under ideal training
- From Kandpal: observed accuracy scales as acc ~ a * log(N) + b
- Combining: for a probe set with known information content, observed
  accuracy constrains N from below
- Formalize: given a probe set P with total information content I(P) bits
  and observed accuracy acc(P), estimate N >= f(acc, I(P))

#### 3.3 Why Procedural Improvements Don't Help

- Decompose model parameters: N = N_fact + N_proc + N_ling
- Densing Law improves the efficiency of N_proc and N_ling (same procedural
  capability in fewer parameters), freeing capacity for N_fact OR allowing
  smaller total N
- But N_fact is bounded below by the information content of stored facts
- IKPs exclusively measure N_fact, which provides a lower bound on N
- The ratio N_fact/N may vary, but Phase 1 calibration empirically captures
  this ratio across model families

#### 3.4 Tiered Probes and the Frequency-Capacity Relationship

- Web knowledge follows a **power-law frequency distribution** (Zipf's law):
  a few facts appear in millions of documents, most facts appear in very few
- Models with more capacity store facts further down the long tail
  (compression theory: compress high-frequency first, then progressively
  encode lower-frequency knowledge as capacity permits)
- This creates a natural "frequency cutoff" per model: facts above the
  cutoff are memorized, facts below it are not. Larger models have a
  lower cutoff (they reach deeper into the tail).

#### 3.5 The Scaling Distribution: Why Log-Linear

The expected relationship between accuracy and parameters:

**Per-tier**: Each tier samples a narrow frequency band. As model size
increases, accuracy on that tier follows a **logistic (sigmoid) curve**:
near-zero for models well below the tier's range, transitioning sharply,
then saturating for models well above. Formally:
  T_i(N) = 1 / (1 + exp(-k_i * (log(N) - m_i)))

**Aggregate**: The sum of 7 sigmoids with staggered midpoints
approximates a **straight line** over a wide range of log(N). Therefore,
aggregate accuracy is approximately **log-linear** in parameters:
  A(N) ≈ a * log(N) + b

This is NOT exponential and NOT power-law in parameters directly. It is
**linear in log-parameters** — meaning each 10x increase in parameters
yields roughly the same absolute increase in accuracy. This matches
Kandpal's empirical R^2=0.98 on BLOOM models.

**Why log-linear arises from power-law frequency**:
1. Web facts follow a power-law frequency distribution: P(freq > f) ~ f^(-α)
2. A model with capacity C memorizes facts with frequency above some
   threshold f_c, where f_c decreases as C increases
3. The number of facts above threshold f_c is proportional to f_c^(-α)
4. For a fixed probe set, accuracy = fraction of probes with freq > f_c
5. Since C ~ N (Allen-Zhu: 2 bits/parameter), and the power-law relationship
   means f_c ~ N^(-1/α), accuracy grows as log(N) after appropriate
   transformations

**Inverting the curve**: Since A(N) ≈ a * log(N) + b, we can estimate:
  log(N̂) = (A - b) / a
This is the core calibration function. The per-tier logistic inversion
provides a more precise estimate using the full accuracy profile.

#### 3.6 Tier Structure

- 7 tiers sample the power-law frequency distribution at logarithmically
  spaced frequency bands, from >100K documents (T1) to ~1 document (T7)
- T1-T3 discriminate small-to-medium models (0.1B-70B)
- T4-T5 discriminate large and frontier models (70B-1T)
- T6 discriminates the very largest deployed models (1T-5T)
- T7 serves as a ceiling/control that no current model should pass
- Equal probe counts (~200 per tier) ensure uniform statistical power


### 4. Methodology

#### 4.1 Probe Generation: Two-Phase Pipeline

**Phase A: LLM-generated probes (T1-T2 primary, T3-T4 supplementary)**
- A capable LLM generates 400 candidate probes per tier
- Valid because T1-T4 facts are within the generator's reliable knowledge
- Structured prompts specify frequency band, geographic/domain balance
- Each candidate's answer verified for correctness
- Empirical finding: ~82% of generated probes land in T1-T2; LLM targets
  reduced to 50 for T3-T4 to reflect actual yield

**Phase B: Corpus-grounded probes (T3-T7)**
- Probes CANNOT be generated from LLM recall (circularity problem for T5+)
- Instead: sample entities from external corpora (DBLP, census databases,
  Wikipedia category pages, open-source training data snippets)
- Measure each entity's document frequency via Google/Scholar result counts
- Assign to tier based on empirically calibrated frequency thresholds
- Verify answer against the source document, not LLM recall
- Wikidata serves as primary source for T3-T4 (100 per tier) alongside
  researcher probes (50 per tier)
- See Phase 1, Step 2 for the detailed corpus-grounded methodology

**Probe quality filters (applied to all sources):**
- Computable probes excluded: probes whose answers can be derived by rule
  (e.g., IUPAC systematic element naming → atomic numbers) are removed
  because they test reasoning capacity, not memorized knowledge
- Researcher name disambiguation: two-character Chinese names (single-char
  given name + surname, e.g., "Yi Gao"), names with only initials
  (e.g., "P.L. Wang"), and garbled entries are excluded to avoid
  LLM hallucination from name collisions

This two-phase design is a key methodological contribution: it ensures
probes are valid across the full parameter range while avoiding the
circularity of using an LLM to probe beyond its own capacity.

#### 4.2 Calibration Filtering

- Run all candidates through representative anchor models at each
  size class (0.5B to ~1T via open-source models on OpenRouter)
- Valid Tier N probe: <20% accuracy below range, 20-80% within range,
  >80% above range (sigmoid discrimination pattern)
- Discard probes that don't discriminate: too easy (all pass), too
  hard (all fail), or anti-correlated with size (training data artifact)
- Select ~200 best probes per tier, prioritizing balance

#### 4.3 Frequency Verification

- Independent web frequency estimation (Common Crawl, Google) for
  each probe
- Cross-check against tier assignment; reassign or discard mismatches
- This step validates that the tier structure corresponds to a real
  web-frequency ladder, not just model behavior

#### 4.4 Scoring and Probe Format

- 3 phrasings per probe (direct, fill-in-blank, contextual)
- Binary scoring: correct if best-of-3 phrasings is correct
- All queries at temperature=0 for determinism
- Hallucination-penalized scoring to discourage confident wrong answers:
  - Correct answer: +1
  - Refusal ("I don't know"): 0
  - Confident wrong answer (hallucination): -0.5
  - Rationale: Models that ignore the "say I don't know" instruction and
    confidently guess on every probe can accumulate false positives by
    chance. The penalty ensures that random guessing has negative expected
    value, so only genuine knowledge improves a model's score. Without the
    penalty, a model guessing randomly on T7 (e.g., 3% hit rate) scores
    +0.03 per probe; with the penalty it scores 0.03 - 0.97*0.5 = -0.455.
- Per-tier accuracy: penalized score summed per tier, divided by tier size
- Aggregate accuracy: total penalized score divided by total probes
- Both penalized and raw (unpenalized) accuracy are reported

#### 4.5 Calibration Curve Fitting

Two estimators:

**(a) Aggregate log-linear:** log(N) = α * A + β, where A is aggregate
accuracy. Motivated by Kandpal's R^2=0.98 finding and the theoretical
argument that accuracy grows linearly in log(params) because knowledge
frequency follows a power law.

**(b) Per-tier logistic inversion:** Fit logistic sigmoid per tier,
invert to estimate log(N) from each tier's accuracy, combine via
weighted median (weight = information content, highest near 50%
accuracy). This is more precise because it uses the full 7-dimensional
accuracy profile.

Report fit quality: R^2, RMSE in log-param space, residual analysis.

#### 4.6 Estimation Procedure for Black-Box Models

- Run all 1400 probes against target model via API
- Compute 7-dimensional accuracy vector + aggregate
- Apply both estimators; report point estimates and 95% CI
- Anomaly detection: flag models with non-monotonic tier profiles
  (e.g., high T2 but low T3 → training data gap, not capacity limit)
- T7 sanity check: models scoring well on T7 likely use RAG

#### 4.7 Addressing Known Confounds

- **Training data coverage**: balanced probes approximate a "typical"
  web corpus; anomaly detection flags unusual training distributions
- **Instruction tuning and RLHF**: mitigated by multi-phrasing and
  temperature=0; hallucination penalty ensures models that refuse
  appropriately are not disadvantaged vs. models that guess wildly
- **MoE architectures**: report both total-param and active-param
  interpretations; expect IKPs to correlate with total params (all
  expert weights store facts)
- **Retrieval augmentation**: T7 serves as RAG detector; response
  latency analysis as secondary signal
- **Contamination risk**: probe set kept private; only construction
  methodology and aggregate statistics published


### 5. Experimental Setup

[Standard: describe models, hardware, API details, evaluation protocol]
[Details from Phase 1 and Phase 2 of research plan]


### 6. Results

Subsections:

- 6.1 Calibration Curve Quality
  - R^2 = 0.893 across 62 open models (1B to 1040B), log-linear fit
  - Slope = 0.152 per decade: each 10x params adds ~15.2 pp accuracy
  - Penalized accuracy fits better than raw (0.893 vs 0.864)
  - Dense non-thinking subset: R^2 = 0.820
  - MoE: total params R^2 = 0.842 vs active params R^2 = 0.430

- 6.2 Per-Tier Discrimination Analysis
  - T3 is the most informative tier (Spearman rho = 0.978 with overall)
  - T4 has highest variance (0.112) — best population discriminator
  - T7 is effectively broken: measures hallucination, not knowledge
  - Per-tier scaling slopes: T3 = 0.425/decade (steepest), T1 = 0.053

- 6.3 Cross-Family Transfer
  - Within-size spread at 32B: 7.6 pp (training data explains most)
  - Within-size spread at 70B: 5.8 pp
  - Llama 70B family consistently overperforms (+7-8 pp residual)
  - Gemma models consistently slightly underperform

- 6.4 Cross-Generation Improvement
  - Claude Opus 4→4.5: +17.9 pp (largest single-gen improvement)
  - Qwen 2.5→3→3.5 at ~8B: +3.6 pp per generation
  - GLM 4→5.1: steady +4-5 pp per generation
  - GPT-4→4o: -3.4 pp regression (distillation knowledge loss)
  - Claude 3.5 Haiku→4.5 Haiku: -7.1 pp regression

- 6.5 Frontier Model Estimates (open-model calibration only)
  - Gemini 3.1 Pro: EXCLUDED (landmark model — T6 score inflated by
    construction, since T6 probes are defined as facts this model knows)
  - Grok-4: ~3,400B
  - GPT-5-think: ~3,200B
  - Claude Opus 4.6: ~2,500B
  - GPT-4.1: ~2,000B
  - GPT-4 validation: predicted 1,188B vs known ~1,800B (34% error)

- 6.6 Thinking Mode Analysis
  - +3.0 pp mean boost across 12 base/think pairs (100% positive)
  - Peak at T3-T4 (medium-hard): +5.0-6.7 pp
  - Near-zero at T7: thinking helps retrieval, not storage
  - Largest beneficiary: Claude Opus 4 (+9.8 pp)

- 6.7 Ablation Studies
  - Hallucination penalty improves size correlation: r=0.723→0.792
  - T3-alone estimation achieves R^2 = 0.801 (viable single-tier)

- 6.8 What Determines Whether an LLM Knows Something?
  
  We cross-reference IKP recognition rates with external metrics for
  both researcher probes (345 probes, OpenAlex citations) and Wikidata
  probes (557 probes, sitelinks, Wikipedia pageviews).

  **6.8.1 Researcher probes: citations are necessary but not sufficient.**
  Core correlation: Spearman rho = 0.575 (log-citations vs recognition).
  The unexplained variance is driven by three verified factors:
    (1) Named, widely-adopted tools — researchers whose work is embedded
        in practitioner workflows (FlashAttention, SVMlight, IPFS) get
        name-dropped in thousands of derivative documents. Tri Dao
        achieves 100% recognition at 3K citations; typical systems
        researchers at 3-5K citations achieve 30-60%.
    (2) Name uniqueness — common names create retrieval interference.
        "Eric Mitchell" (DPO creator, 15K citations) → 57%. "Aditi
        Raghunathan" (unique name, 5K citations) → 100%. Controlling
        for citations, common East Asian surnames → 22.6% vs 44.6%.
    (3) Practitioner ecosystem amplification — ML work generates massive
        secondary content (blog posts, tutorials, model cards) that
        mentions researchers by name. This creates an "ML recognition
        floor": even junior PhD students with <300 citations achieve
        43-57% (verified as genuine knowledge, not hallucination).

  **6.8.2 Wikidata probes: pageviews beat sitelinks; fact specificity
  matters more than entity prominence.**
  Wikipedia pageviews (r=0.77) are a far stronger predictor than
  sitelinks (r=0.50). In multiple regression, sitelinks add nothing
  beyond pageviews. The central finding: the gap between entity-name
  knowledge and specific-fact knowledge (e.g. founding year) widens
  with entity prominence. For entities with 16+ sitelinks, models know
  the name 73% of the time but the founding year only 47% — a 26-point
  gap. For obscure entities, the gap is only 4 points. This demonstrates
  that LLMs learn facts proportionally to each fact's specific mention
  frequency, not the entity's overall importance.

  Domain effects are large: journal founding years are easy (+0.215
  residual vs sitelinks) because years appear in every bibliography
  and citation record. Place founding years are hard (-0.310) because
  they are buried trivia rarely stated on web pages about the place.
  Founding years in general invite hallucination (37-47% wrong rate)
  because the year format makes confident guessing easy.

  **6.8.3 Unifying principle: effective mention frequency.**
  Both researcher and Wikidata analyses converge on the same principle:
  LLM recognition of any fact is determined by its effective mention
  frequency — the number of training-corpus documents stating that
  specific fact in an unambiguous, retrievable form. For researchers,
  this is mediated by name uniqueness and tool adoption. For entities,
  this is mediated by the ratio of fact-specific mentions to general
  mentions (a famous bridge is mentioned often, but its opening year
  rarely). Citation count and sitelink count are proxies for effective
  mention frequency, but with substantial domain-dependent noise.

- 6.9 Knowledge Fingerprinting and Distillation Detection
  - Gemini 3 cluster: Jaccard 0.85-0.90 (tightest in dataset)
  - Distillation detectable: teacher vs base knowledge ratio measurable
  - deepseek-r1-distill-llama-70b: 1:1 teacher/base inheritance
  - deepseek-r1-distill-qwen-32b: 1:2.5 teacher/base (base dominates)
  - Within-vendor mean Jaccard 0.328 vs between-vendor 0.290 (1.13x)


### 7. Discussion

Planned topics:
- 7.1 What IKPs measure: factual storage capacity, not total parameters.
  The relationship between the two.
- 7.2 Practical applications: governance, procurement, model auditing,
  distillation detection, provenance verification
- 7.3 Distillation detection: implications (see Additions above)
- 7.4 Limitations: training data coverage assumption, RAG confound,
  P_factual/P_total ratio uncertainty, shared training data confound
  for fingerprinting, RLHF conservatism confound (below)
  **RLHF conservatism confound.** Safety-tuned models may refuse to
  answer probes they actually know, producing systematic underestimates.
  The clearest example: Claude Opus 4.1 scores 53.1% (estimated ~147B)
  while Claude 3.7 Sonnet scores 67.3% (~2T) — yet Opus 4.1 is almost
  certainly the larger model. On T5 probes, Opus 4.1 refuses 157/200
  times with verbose "I don't know" responses, while 3.7 Sonnet refuses
  only 35/200 and correctly answers 151. The same pattern appears in
  Opus 4 (49%) vs Opus 4.5 (67%): the 4→4.5 jump reflects reduced
  conservatism, not more parameters. The hallucination penalty mitigates
  this (refusal scores 0 rather than -0.5), but cannot recover the +1
  points lost to unnecessary refusals. IKP estimates for heavily
  safety-tuned models should therefore be interpreted as lower bounds.
  When base/think pairs exist, the thinking variant's score is typically
  closer to the true capacity (thinking reduces over-refusal by allowing
  the model to reason through uncertainty before committing).

- 7.5 The privacy requirement: why the probe set must remain private,
  and implications for reproducibility (discuss releasing construction
  methodology + a subset for verification)
- 7.6 IKP as a Knowledge Presence Metric
- 7.7 Practitioner Amplification and the Divergence of Academic vs LLM Impact

#### 7.6 IKP as a Proxy for Knowledge Presence

Beyond parameter estimation, the IKP framework yields a novel metric for
measuring how thoroughly an entity (researcher, institution, historical
event) has permeated LLM training data. This section discusses findings
from cross-referencing IKP visibility with traditional bibliometric and
web-based metrics across 140 models and 345 CS researchers with OpenAlex
citation data.

**Core correlation:** Spearman rho = 0.575 between log(citations) and
recognition rate across 345 CS researchers. The relationship is
log-linear, not linear (Pearson r improves from 0.38 to 0.59 with
log-transform). The tier system is validated: median citations drop
monotonically from T3=6,859 to T7=325, and median h-index from T3=38
to T7=7.

**Verified findings (from 8-cell web search analysis of 20 researchers
across ML/Systems × High/Low citations × High/Low recognition):**

1. **LLM recognition is driven by "effective mention frequency" — the
   number of training-corpus documents with unambiguous name-field
   co-occurrence.** Citation count is one contributor, but three other
   factors are equally or more important:

   (a) **Named, widely-adopted tools create outsized recognition.**
       Tri Dao (FlashAttention, 23K GitHub stars) achieves 100%
       recognition at only 3K citations. Thorsten Joachims (SVMlight)
       achieves 97% at 52K citations. Both have their names embedded
       in thousands of READMEs, tutorials, model cards, and blog
       posts. The key is not the paper but the tool's adoption —
       every `import flash_attn` or `svmlight` reference in training
       data reinforces the name-field association.

       This also works outside ML: Yiannis Psaras (IPFS/Protocol
       Labs, 318 citations) achieves 69% because the IPFS/web3
       ecosystem generates massive practitioner content mentioning
       him by name. Susan Lysecky (zyBooks textbook author, 382
       citations) achieves 54% because her name appears in thousands
       of university course syllabi.

   (b) **Name uniqueness acts as a multiplicative factor.** Common
       names create retrieval interference in the model's associative
       memory. Controlling for citations (1,000-10,000 range),
       researchers with common East Asian surnames are recognized
       22.6% vs 44.6% for distinctive names — a 2x gap at the same
       citation level.

       Verified example: Eric Mitchell created DPO (one of the most
       impactful ML papers of 2023, ~15K citations) but achieves
       only 57% recognition because "Eric Mitchell" collides with an
       actor, filmmaker, and thousands of others. Aditi Raghunathan
       (unique name, 5K citations) achieves 100%.

   (c) **Practitioner ecosystem amplification.** ML researchers at
       3-5K citations achieve 86-100% recognition while systems
       researchers at the same level achieve 30-60%. Investigation
       reveals this is NOT because papers are behind paywalls (both
       ML and systems conference papers are in training data), but
       because ML work generates massive secondary content: blog
       posts explaining papers, YouTube tutorials, HuggingFace model
       cards, GitHub READMEs, Twitter threads. Each of these mentions
       the researcher by name. Systems research generates far less
       of this secondary amplification.

       ML labs also create a "recognition floor": even junior Stanford
       PhD students with <300 citations achieve 43-57% recognition
       (verified as genuine knowledge, not hallucination) because lab
       blog posts, joint papers with famous advisors, and project
       pages cross-reference all members.

2. **Citation count validates tier assignments but imperfectly.**
   Median citations drop monotonically (T3=6,859 → T7=325).
   Spearman rho between tier and log-citations is -0.509. The
   imperfection is explained by factors (a)-(c) above: researchers
   with unique names + tool adoption can be well-known despite modest
   citations, while researchers with common names can be unknown
   despite high citations.

3. **Data quality warning for automated bibliometric lookups.**
   OpenAlex name disambiguation is unreliable for common names.
   Three "high-citation unrecognized" researchers in our T6-T7
   probes turned out to be name collisions:
   - "Yan Jiao" (46K citations): actually a chemistry professor
   - "Xinming Wang" (44K citations): actually an atmospheric chemist
   - "Sylvie Dujardin" (149 citations): actually a sleep researcher
   Any study correlating LLM knowledge with bibliometrics must
   manually verify matches for common names.

**The recognition hierarchy (verified):**

| Factor                              | Impact       | Evidence                    |
|-------------------------------------|--------------|-----------------------------|
| Wikipedia page                      | Very high    | Gerla 89%, Mamba helps Gu 86% |
| Named tool with >10K GitHub stars   | Very high    | FlashAttention → Dao 100%   |
| Name uniqueness                     | Multiplicative | Raghunathan 100% vs Mitchell 57% at higher citations |
| Practitioner content amplification  | High         | ML 3-5K cit: 86-100% vs Systems: 30-60% |
| Educational content (YouTube, textbooks) | Medium-high | Micinski 59%, Lysecky 54% |
| Elite lab ecosystem membership      | Medium       | Hazy Research → Chen 57%, Eyuboglu 57% |
| Non-academic ecosystem (web3, startups) | Medium    | IPFS → Psaras 69%          |
| Citation count (log-scale)          | Moderate     | rho = 0.575                 |

**What separates 57% from 100% recognition:** The jump requires at
least one "anchor" — a Wikipedia page, a named tool with >10K GitHub
stars, or a perfectly unique name combined with multiple prestigious
awards. Researchers at 57% have solid academic presence but lack a
single viral anchor.

**Implications:**
- LLM knowledge of named entities is a function of effective mention
  frequency, not academic impact. Tool builders, educators, and
  researchers in practitioner-heavy fields accumulate more effective
  mentions per citation than researchers in traditional academic fields.
- Name uniqueness is a previously unrecognized confound in LLM
  evaluations involving named entities. LLMs systematically
  underrepresent researchers with common names, regardless of impact.
- Any benchmark using researcher names must account for this bias
  or risk measuring name uniqueness rather than model knowledge.

#### 7.7 Effective Mention Frequency and Practical Implications

Our analysis of 345 researchers and 557 Wikidata entities reveals a
unifying principle: LLM knowledge is determined by **effective mention
frequency** — the number of training documents stating a specific fact
in retrievable form. This is distinct from traditional prominence
metrics (citations, sitelinks) and has a consistent structure across
both people and entities.

For researchers, effective mention frequency is amplified by tool
adoption (FlashAttention's 23K GitHub stars generate thousands of
name-bearing documents), educational content (YouTube lectures,
textbooks), and practitioner ecosystems (lab blogs, project pages).
It is attenuated by name ambiguity (common names split mentions
across multiple people). For entities, it is amplified by domains
where the specific fact is routinely stated (journal founding years
appear in every bibliography) and attenuated by domains where the
fact is buried trivia (a famous bridge's opening year is rarely
mentioned despite the bridge itself being well-known).

**Practical implications.** As LLMs increasingly mediate information
retrieval, an entity's LLM recognition becomes a form of
discoverability. Our findings suggest what drives this:
- For researchers: the single most effective strategy is building a
  widely-adopted open-source tool with clear name attribution. Each
  README, import statement, and tutorial that references the tool
  also references the author, creating compounding mention frequency.
  Educational content (lectures, blog posts) indexed by search engines
  is the second most effective channel. Using a distinctive professional
  name avoids retrieval interference from namesakes.
- For entities more broadly: our Wikidata analysis shows that LLMs
  learn specific facts in proportion to how often each fact is
  explicitly stated in web text, not in proportion to the entity's
  general prominence. Journals are well-recognized because their
  founding years appear in every bibliography and citation record —
  the fact is repeated as a side effect of normal academic practice.
  Places and bridges are poorly recognized despite high prominence
  because their founding dates are buried trivia, rarely stated in
  the documents that mention them. This suggests that entities in
  domains with structured, repeatedly-stated metadata (journals,
  universities with founding years in every course catalog) will be
  better represented in LLM knowledge than entities of equal
  prominence in domains where key facts are stated only on a
  dedicated Wikipedia page.



### 8. Conclusion

[TBD — summarize contributions and results]


---


## Appendix: Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Calibration R^2 is low (<0.8) | High | Use per-tier accuracies (7-dim vector) as features instead of aggregate; try nonparametric fits |
| Cross-family transfer fails | High | Report family-specific calibrations; still useful if you know the model family |
| MoE models break the curve | Medium | Analyze MoE separately; report total vs active param estimates |
| Probes leak and get contaminated | Medium | Keep probes private; design replacement protocol for periodic refresh |
| Frontier models use RAG | Medium | Include ceiling probes (Tier 7) as RAG detectors; report latency analysis |
| Models refuse to answer probes | Low | Multi-phrasing; exclude refusals from scoring; use base models when available |
| Geographic bias skews results | Medium | Strict balance requirements; per-region accuracy analysis in ablations |
| Allen-Zhu 2-bit bound doesn't hold for real models | Medium | Calibration is empirical, not dependent on exact bound; bound provides motivation only |
| Fingerprint overlap due to shared training data | Medium | Use within-family baselines (same data, different sizes) as controls; report effect sizes relative to these baselines |
| Fingerprint degraded by post-distillation fine-tuning | Low | Known fine-tuned pairs (Hermes-3, QwQ) calibrate how much fine-tuning weakens signal; report sensitivity vs fine-tuning degree |
