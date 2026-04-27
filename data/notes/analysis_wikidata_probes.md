# Wikidata Probe Analysis: What Determines Whether LLMs Know a Fact?

## 1. Dataset Overview

- **557 Wikidata probes** across 7 tiers (T1-T7), 16 domains
- **391 probes** with sitelink_count (number of Wikipedia language editions linking to the entity)
- **145 models** evaluated; recognition rate = fraction of models answering correctly
- Dominant fact type: founding/opening years (532 probes, 95.5%); remainder are identity facts (capitals, currencies, names)

### Recognition Rate by Tier

| Tier | n   | Mean Recognition |
|------|-----|-----------------|
| T1   | 6   | 0.991           |
| T2   | 5   | 0.989           |
| T3   | 94  | 0.750           |
| T4   | 111 | 0.575           |
| T5   | 100 | 0.356           |
| T6   | 141 | 0.121           |
| T7   | 100 | 0.032           |

## 2. Sitelink Count vs Recognition Rate

### Overall Correlation (n=391)

| Metric                        | Pearson r | Spearman rho |
|-------------------------------|-----------|-------------|
| sitelink_count vs recognition | 0.384     | 0.459       |
| log(sitelinks) vs recognition | **0.502** | 0.459       |

The log-transformed sitelink count is a better predictor (Pearson r=0.502 vs 0.384), indicating a diminishing-returns relationship: going from 1 to 10 sitelinks matters more than going from 100 to 200.

### By Sitelink Bin

| Sitelink Range | n   | Mean Recognition | Stdev |
|----------------|-----|-----------------|-------|
| 1-2            | 126 | 0.226           | 0.249 |
| 3-5            | 98  | 0.301           | 0.264 |
| 6-10           | 58  | 0.452           | 0.278 |
| 11-20          | 69  | 0.591           | 0.272 |
| 21-50          | 28  | 0.440           | 0.287 |
| 101-300        | 11  | 0.990           | 0.006 |

Note the dip at 21-50 sitelinks -- these are mostly T5-T6 "places" probes asking about founding years, which are hard regardless of entity prominence.

### Correlation Within Tiers

| Tier | n   | Pearson r (log sitelinks) | p-value   |
|------|-----|---------------------------|-----------|
| T3   | 72  | 0.472                     | 2.8e-05   |
| T4   | 77  | 0.155                     | 0.18      |
| T5   | 67  | 0.110                     | 0.38      |
| T6   | 102 | 0.144                     | 0.15      |
| T7   | 62  | -0.057                    | 0.66      |

**Key finding**: Sitelinks strongly predict recognition within T3 (r=0.47), but become uninformative within T4-T7. Within a given difficulty tier, sitelinks alone cannot distinguish which facts models know.

## 3. Wikipedia Pageviews vs Recognition Rate

For 78 probes sampled across tiers, we fetched 2025 annual English Wikipedia pageviews.

| Metric                        | Pearson r | Spearman rho |
|-------------------------------|-----------|-------------|
| log(pageviews) vs recognition | **0.774** | **0.761**   |
| log(sitelinks) vs recognition (same subset, n=56) | 0.704 | 0.654 |

### Multiple Regression (n=56)

| Model                        | R-squared |
|------------------------------|-----------|
| log(sitelinks) only         | 0.496     |
| log(pageviews) only         | 0.587     |
| log(sitelinks) + log(pageviews) | 0.587 |

**Pageviews subsume sitelinks entirely.** In the joint model, the sitelink coefficient drops to -0.003 (essentially zero). Sitelinks are merely a noisy proxy for what actually matters: how frequently the entity's Wikipedia page is accessed, which correlates with how often the fact appears in training corpora.

### Sitelinks vs Pageviews Correlation

log(sitelinks) and log(pageviews) are highly correlated (r=0.922), but pageviews capture the residual variance that sitelinks miss.

## 4. Domain-Specific Analysis

### Domain Correlation with Sitelinks

| Domain                  | n   | Mean Rec | Mean Sitelinks | log(sl) r |
|-------------------------|-----|----------|----------------|-----------|
| university              | 36  | 0.533    | 9.4            | **+0.823** |
| founding_year_sports    | 37  | 0.217    | 7.8            | +0.666    |
| sports_club             | 24  | 0.516    | 13.9           | +0.619    |
| founding_year_uni2      | 60  | 0.445    | 4.7            | +0.577    |
| journal                 | 39  | 0.614    | 7.8            | +0.524    |
| bridge                  | 17  | 0.422    | 8.9            | +0.497    |
| founding_year_museum2   | 37  | 0.221    | 4.4            | +0.482    |
| museum                  | 26  | 0.281    | 7.4            | +0.447    |
| founding_year_journal   | 40  | 0.449    | 2.2            | +0.376    |
| founding_year_bridge    | 20  | 0.051    | 2.8            | +0.319    |
| places                  | 44  | 0.142    | 12.2           | +0.245    |

University probes show the strongest sitelink-recognition correlation (r=0.823); places are the weakest (r=0.245).

### Domain Residual Analysis (Controlling for Sitelinks)

Positive residual = domain is *easier* than sitelinks predict; negative = *harder*.

| Domain                  | n   | Mean Residual |
|-------------------------|-----|---------------|
| journal                 | 39  | **+0.215**    |
| founding_year_journal   | 40  | **+0.201**    |
| founding_year_uni2      | 60  | +0.117        |
| university              | 36  | +0.120        |
| capital                 | 8   | +0.092        |
| bridge                  | 17  | +0.033        |
| sports_club             | 24  | +0.032        |
| founding_year_museum2   | 37  | -0.102        |
| museum                  | 26  | -0.118        |
| founding_year_sports    | 37  | -0.144        |
| founding_year_bridge    | 20  | **-0.219**    |
| places                  | 44  | **-0.310**    |

**Journals are the easiest domain** given their sitelink count (residual +0.215). Journal names and founding years appear in bibliographies, citation records, and indexing databases -- providing high "effective mention frequency" relative to their Wikipedia presence. **Places are the hardest** (residual -0.310): even entities with 12+ sitelinks have low recognition because founding year is rarely mentioned on web pages about a place.

## 5. Paired Comparison: Name Facts vs Founding Year Facts

For entities of the same type, asking "what is the name of X" is easier than "when was X founded":

| Entity Type    | Name Rec | Year Rec | Delta   |
|----------------|----------|----------|---------|
| Bridge         | 0.422    | 0.033    | **+0.389** |
| Sports club    | 0.516    | 0.201    | +0.315  |
| Journal        | 0.614    | 0.408    | +0.206  |
| University     | 0.527    | 0.438    | +0.089  |
| Museum (avg)   | 0.278    | 0.203    | +0.076  |

The name-vs-year gap **widens with entity prominence** (controlling for sitelinks):

| Sitelink Range | Name Rec | Year Rec | Delta   |
|----------------|----------|----------|---------|
| 1-5 (obscure)  | 0.299    | 0.259    | +0.040  |
| 6-15 (moderate) | 0.614   | 0.533    | +0.081  |
| 16+ (prominent) | 0.728   | 0.473    | **+0.255** |

This is a critical finding: **more prominent entities have a *larger* gap between name knowledge and temporal knowledge.** A famous bridge is mentioned thousands of times (reinforcing its name), but its opening year is stated in only a tiny fraction of those mentions.

## 6. Hallucination Analysis

| Domain                  | Wrong Rate | Refusal Rate | Recognition |
|-------------------------|------------|-------------|-------------|
| founding_year_sports    | **46.7%**  | 31.6%       | 0.217       |
| museum                  | 46.4%     | 25.5%       | 0.281       |
| founding_year_museum2   | 44.0%     | 33.9%       | 0.221       |
| founding_year_bridge    | 40.9%     | 54.0%       | 0.051       |
| places                  | 38.6%     | 47.2%       | 0.142       |
| founding_year_uni2      | 37.1%     | 18.4%       | 0.445       |
| founding_year_journal   | 36.6%     | 18.5%       | 0.449       |
| bridge                  | 34.9%     | 23.0%       | 0.422       |
| university              | 31.9%     | 14.8%       | 0.533       |
| journal                 | 29.6%     | 9.0%        | 0.614       |
| sports_club             | 34.2%     | 14.3%       | 0.516       |
| capital                 | **0.1%**  | 0.8%        | 0.991       |

**Founding years invite hallucination.** Models confidently guess plausible-sounding years, producing wrong rates of 37-47%. Capitals have near-zero hallucination -- models either know the answer or refuse. The "year" answer format is particularly hallucination-prone because any 4-digit number is a plausible guess.

## 7. Temporal Era Effect

| Era       | n   | Mean Recognition | Mean Sitelinks |
|-----------|-----|-----------------|----------------|
| pre-1800  | 31  | 0.247           | 14.3           |
| 1800-1900 | 63  | 0.388           | 9.1            |
| 1900-1950 | 118 | **0.404**       | 8.1            |
| 1950-2000 | 208 | 0.338           | 5.5            |
| 2000+     | 112 | 0.256           | 5.2            |

Entities founded in 1900-1950 are best-known despite lower sitelink counts than pre-1800 entities. This likely reflects the "documentation sweet spot": old enough to be historically significant and well-documented, but recent enough to have comprehensive web coverage.

## 8. Variance Decomposition

| Model                                  | R-squared |
|----------------------------------------|-----------|
| Tier alone                             | **0.812** |
| Domain alone                           | 0.391     |
| log(sitelinks) alone                   | 0.252     |
| Tier + log(sitelinks)                  | 0.830     |
| log(sitelinks) + domain                | 0.566     |
| Tier + domain                          | 0.845     |
| Tier + log(sitelinks) + domain         | **0.858** |

**Tier is by far the strongest predictor** (R^2=0.812), which is expected since tiers were designed to capture difficulty. Within-tier, domain adds more predictive power than sitelinks (tier+domain R^2=0.845 vs tier+sitelinks R^2=0.830). The full model explains 85.8% of variance.

## 9. Comparison with Researcher Probes

| Probe Type   | Predictor            | Pearson r | Spearman rho | n   |
|-------------|----------------------|-----------|-------------|-----|
| Researcher  | log(citations)       | 0.599     | 0.586       | 345 |
| Wikidata    | log(sitelinks)       | 0.502     | 0.459       | 391 |
| Wikidata    | log(pageviews)       | **0.774** | **0.761**   | 78  |

**Wikidata findings are more predictable than researcher findings** when using pageviews (rho=0.761 vs 0.586), but less predictable when using sitelinks (rho=0.459 vs 0.586). This makes sense: citation count directly measures how often a researcher is discussed in academic text (their training-relevant "mention frequency"), while sitelinks measure only the breadth of multilingual coverage -- a weaker proxy for English-language training frequency.

## 10. Key Conclusions: The "Effective Mention Frequency" for Wikidata Facts

For researcher probes, the paper found that what predicts LLM knowledge is not raw prominence but "effective mention frequency" -- how often the specific fact appears in training data. The Wikidata analysis reveals the exact same principle operating through a different mechanism:

### Sitelinks measure entity prominence, not fact salience

Sitelinks count how many Wikipedia languages have an article about an entity. This correlates with recognition (r=0.50) but misses a crucial distinction: knowing *about* an entity (its name, that it exists) is different from knowing a *specific fact* about it (when it was founded).

### Pageviews are the better proxy for training frequency

English Wikipedia pageviews (r=0.77) completely subsume sitelinks in a joint model. Pageviews better approximate how often an entity appears in English-language web text, which is the actual training signal.

### The fact-type gap is the Wikidata "effective mention frequency" effect

The central finding: **at the same entity prominence level, identity facts (names, capitals) are known far better than temporal facts (founding years).** The gap is +0.04 for obscure entities but +0.26 for prominent ones. This is the Wikidata analog of the researcher finding: a famous bridge's name appears in thousands of documents, but its opening year appears in perhaps dozens. Entity prominence amplifies name knowledge but not temporal knowledge, because the fraction of mentions that include the specific fact *decreases* with prominence.

### Domain-specific "mention multipliers"

Some fact types have structural advantages that inflate their effective mention frequency:
- **Journals** (+0.215 residual): Publication years appear in every citation and bibliography entry
- **Universities** (+0.120 residual): Founding years appear on institutional pages, CV entries, rankings
- **Places** (-0.310 residual): Founding years are buried in history sections, rarely in general mentions
- **Bridges** (-0.219 residual): Opening dates are trivia, not frequently cited

### Temporal facts are uniquely hallucination-prone

Founding years have wrong-answer rates of 37-47%, far above identity facts (<1% for capitals). The year format makes every 4-digit number a plausible guess, and models rarely refuse. This is a qualitatively different failure mode from researcher probes.

### The "documentation sweet spot" for temporal facts

Entities from 1900-1950 are best-known (0.404 recognition) despite lower sitelink counts than older entities. This suggests a training data coverage curve: old enough to be historically notable, recent enough for comprehensive web documentation.

---

**Bottom line**: For Wikidata probes, the best predictor of whether an LLM knows a fact is not the entity's prominence (sitelinks) but the frequency with which that *specific fact* appears in English-language text (approximated by pageviews, r=0.77). The gap between entity-level prominence and fact-level knowledge is the Wikidata equivalent of the "effective mention frequency" finding for researchers -- and it manifests most dramatically in the growing gulf between name knowledge and temporal knowledge as entities become more famous.
