# Wikidata Long-Tail Data Quality Findings

Audit notes from the diverse-fact-type sourcing rounds. Drawn from manual web verification of ~120+ T6/T7 candidates across rounds 1-6 plus the original probe-set audit.

## Headline numbers (for main paper body)

- **~12% of T5-T7 founding-year wikidata probes carry an ambiguity defect** (bare common-name subjects: "Putnam", "Wells", "Norwich" — Wikidata has the right entity, but the question as phrased cannot be unambiguously interpreted by a language model).
- **~33-46% of diverse-fact-type wikidata candidates fail web cross-check** at the obscure tier — far higher than the ~5% rate for `P571 inception` (founding year), which is the most-curated date field.
- **Per-fact-type Wikidata reliability at the long tail (verified empirically across rounds 4-6):**
  | Fact type | Wikidata pass rate | Failure mode |
  |---|---|---|
  | `P571` inception (founding year) | ~95% | data extraction bugs (e.g. building year vs museum founding year — pipeline issue, not Wikidata) |
  | `P403` river mouth | 100% | reliable |
  | `P17` country (cape/island/lake) | 75-100% | reliable |
  | `P57` director (film) | 67-100% | reliable |
  | `P112` founder | 75% | sometimes returns the founding *organization* instead of person |
  | `P170` creator (sculpture) | ~70% | **frequently returns the bronze foundry** (Susse Frères, Eugène Rudier) instead of the sculptor (Giacometti, Bourdelle) |
  | `P58` screenwriter | 50% | conflates story credit with screenplay credit |
  | `P159` headquarters | 25% | **stale** — often reflects HQ from 5+ years ago (Roku still listed in Los Gatos despite 2019 move to San Jose; Vista Outdoor listed in Clearfield despite move to Anoka) |
  | `P138` named after | mixed | self-referential errors ("Na Klang named after Na Klang"); descriptive Thai placename incorrectly marked as eponymous |
  | `P170` creator (painting) | ~70%, but title-collision rate near 100% at obscure tier | titles like "Madonna and Child", "Holy Trinity", "Saint Apollonia" each refer to dozens of works by different painters; Wikidata has the right entity, but the question is unanswerable without disambiguation |

## Failure-mode taxonomy (proposed for appendix)

### A. Wikidata-itself-wrong
Outright incorrect facts in Wikidata that can be falsified via Wikipedia or primary source.
- **A1. Stale data**: `P159` headquarters often reflects state from years ago (Roku, Vista Outdoor)
- **A2. Wrong field semantics**: `P170` creator returns foundry, not sculptor (Giacometti's *L'Homme qui marche I* attributed to Susse Frères; Bourdelle's *Monument to Alvear* attributed to Eugène Rudier)
- **A3. Self-referential / circular**: "Na Klang named after Na Klang" (descriptive Thai name marked as eponymous)
- **A4. Plain wrong year/value**: Makran Medical College listed as 2015 when Wikipedia infobox says 2017; University of Ruse listed as 1954 when Wikipedia documents 1945 (precursor) and 1995 (current entity); Lady Snowblood screenplay attributed to manga illustrator instead of Norio Osada

### B. Pipeline extraction bugs (on our side)
- **B1. Wrong date field**: George Eastman Museum stored as 1905 (the year the house was built) instead of `P571=1947` (museum chartering)

### C. Probe-construction ambiguity (Wikidata correct, question unanswerable)
- **C1. Generic title — religious art**: "Madonna and Child", "Holy Trinity", "Saint Apollonia" — dozens of paintings under each title across centuries
- **C2. Generic title — geographic feature**: "Stone Bridge", "Footbridge", "South Island" (could be New Zealand, Kenya/Lake Turkana, ...), "Hog Island" (one in Falklands, several in US)
- **C3. Bare common-name place**: "Putnam", "Norwich", "Wells" — Wikidata has Putnam, Connecticut (1855); model defaults to more famous Putnam County or Putnam (England)
- **C4. Generic institution name**: "Maritime Museum" (one in Jakarta vs San Diego vs London Greenwich), "St. Lawrence University" (one in Kampala, Uganda vs famous one in NY)
- **C5. Compound/sponsor naming**: "Merkur Arena" — football stadium with insurance sponsor "Merkur"; Wikidata may attribute "named after" to the sponsor in awkward ways

### D. Politically/legally contested entities
- **D1. Disputed-sovereignty answers**: Loaita Cay listed under Vietnam (one of four claimants — Philippines, China, Taiwan, Vietnam — physically held by Philippines). Any single-country answer to "in what country is X located" is contested for entities in the Spratlys, Crimea, Western Sahara, Northern Cyprus, etc. Wikidata picks one; reasonable models will pick a different one.
- **D2. Historical-vs-current statehood**: Mount Ashley (South Georgia) listed under "United Kingdom" (technically South Georgia is a separate British Overseas Territory, not part of the UK proper).

### E. Multi-actor attribution
- **E1. Co-founder selection**: Wikidata picks one founder (Cinémathèque de Tanger has 3 founders — Lahlou, Auriol, Barrada — judge must accept any of them); Braille Without Borders has Tenberken AND Kronenberg
- **E2. Co-author / story-vs-screenplay**: Twenty-Four Eyes screenplay by Kinoshita, but Wikidata attributed Sakae Tsuboi (the *novelist*); Amazing Panda Adventure screenplay by Rothberg/Elehwany but Wikidata listed Wilcox (story credit)

## Methodological implications

1. **Calibration depends on answer correctness.** When a probe's `answer` field is corrected (e.g., Twenty-Four Eyes from Sakae Tsuboi → Keisuke Kinoshita), the probe must be **re-calibrated** through the landmark pipeline, because models that previously answered "Kinoshita" were judged wrong vs the old gold and right vs the new gold. In our audit, all 5 corrected probes shifted from T7 to T5 after re-calibration — they were never genuinely T7 probes.

2. **Sourcing diverse fact types yields fewer T7 candidates per query than founding-year sourcing.** Diverse-fact-type SPARQL queries (P57/P170/P138/etc.) yielded ~25% T6/T7 vs ~38% for founding-year queries. Reason: entities famous enough to have a curated diverse property tend to fall into T5-T6 rather than T7.

3. **Painter probes are essentially unusable for T7 obscurity.** Obscure paintings tend to have generic religious-art titles (Madonna and Child, Saint X, Holy Y), which collide with hundreds of works across centuries. Title-collision rate ≈100% at our obscurity tier.

4. **Geography-with-country is the most reliable diverse fact type at the long tail.** River-mouth (`P403`) and country-of-location (`P17` for capes, lakes, islands) consistently passed verification. These properties are well-curated and the answer (a country) is hard to confuse.

5. **The drop-and-recalibrate iteration is slow** — each web-verification round catches ~30-46% problematic candidates. Replacing 60 T7 probes from diverse-only sources required 6 sourcing rounds, ~1,000 calibration API calls per round, and ~120 manual web-verifications in this audit.

## Audit telemetry (raw counts for appendix table)

| Round | Sourced | Calibrated valid | T6/T7 yield | Web-verified pass | Used as replacement |
|-------|---------|------------------|-------------|-------------------|---------------------|
| 1 (founding-year) | 186 | 155 | 70 (T6=25, T7=45) | partial (founding-year mostly clean) | initial — many later rejected per "no founding-year" mandate |
| 2 (mixed) | 152 | 138 | 39 (T6=20, T7=19) | 18/27 diverse pass (67%) | partial |
| 3 (composer/birthplace failed via SPARQL timeouts) | 0 | — | — | — | — |
| 4 (diverse-only) | 140 | 124 | 24 (T6=10, T7=14) | 15/24 (62%) | yes |
| 5 (12 fact types) | 250 | 228 | 68 (T6=39, T7=29) | 21/29 T7 (72%) | yes |
| 6 (in progress) | 275 | TBD | TBD | TBD | TBD |


## Updated audit telemetry (final, after rounds 5-7)

| Round | Sourced | Calibrated valid | T6/T7 yield | Web-verified pass | Used as replacement |
|-------|---------|------------------|-------------|-------------------|---------------------|
| 1 (founding-year) | 186 | 155 | 70 (T6=25, T7=45) | partial — most rejected per "no founding-year" mandate | **9 T7 retained** |
| 2 (mixed) | 152 | 138 | 39 (T6=20, T7=19) | 18/27 diverse (67%) | partial |
| 3 (composer/birthplace SPARQL timeouts) | 0 | — | — | — | — |
| 4 (diverse-only) | 140 | 124 | 24 (T6=10, T7=14) | 15/24 (62%) | **8 T7 retained** |
| 5 (12 fact types) | 250 | 228 | 68 (T6=39, T7=29) | 21/29 T7 (72%) | **21 T7 retained** |
| 6 (11 fact types, +castle) | 275 | 256 | 64 (T6=36, T7=28) | 18/28 T7 (64%) | **18 T7 retained** |
| 7 (7 reliable types only) | 140 | 130 | 21 (T6=9, T7=12) | 11/12 T7 (**92%**) | **11 T7 retained** |

Total API spend: ~10,000 OpenRouter calls across calibration; ~120 manual web-verification fetches.

## Per-fact-type final pass-rate ranking (across rounds 4-7)

Most-to-least reliable for obscure-tier wikidata sourcing:

1. **`P403` river mouth** — 100% pass rate. Geography. Well-curated.
2. **`P57` director (film)** — 67-100%. Well-curated for non-disputed films.
3. **`P17` country (cape/island/lake/mountain)** — 75-100%. Well-curated for non-disputed territories. **Caveat**: politically contested entities (Crimea, Spratlys, Western Sahara) consistently fail web cross-check because Wikidata picks one claimant.
4. **`P112` founder (organization)** — 75%. Sometimes returns the founding *organization* rather than the founding *person* (P112 has loose typing).
5. **`P170` creator (sculpture)** — ~70%. **Frequently returns the bronze foundry instead of the sculptor.** Examples: Giacometti's *L'Homme qui marche I* attributed to "Susse Frères"; Bourdelle's *Monument to Alvear* attributed to "Eugène Rudier".
6. **`P58` screenwriter** — 50%. Conflates "story by" credit with "screenplay by" credit.
7. **`P159` headquarters** — 25%. **Stale**. Often reflects HQ from years ago (Roku still listed as Los Gatos despite 2019 move to San Jose; Vista Outdoor still listed as Clearfield despite move to Anoka).
8. **`P170` creator (painting)** — passes attribution check but **~100% title-collision rate** at obscurity. Generic religious art names ("Madonna and Child", "Holy Trinity", "Saint Apollonia", "Last Judgment") refer to dozens of works. Even where Wikidata picks a real obscure painter correctly, the question is unanswerable without disambiguation.
9. **`P138` named after** — mixed. Self-referential errors observed ("Na Klang named after Na Klang"); descriptive Thai/Native placenames incorrectly marked as eponymous.

## Final replacement pool composition

After 7 sourcing rounds + calibration + web-verification + selection, the 72 replacement probes (10 T6 + 62 T7) span **16 fact types**:

| Fact type | Count |
|-----------|-------|
| river_mouth | 12 |
| director | 11 |
| cape_country | 7 |
| sculpture_creator | 6 |
| lake_country | 5 |
| founder | 5 |
| painter | 4 |
| mountain_country | 4 |
| screenwriter | 3 |
| founder_person | 3 |
| island_country | 3 |
| hq_city | 3 |
| architect | 2 |
| composer_song | 2 |
| named_after | 1 |
| founding_year_bridge | 1 |

Compare to the *prior* T6/T7 wikidata-source mix: **100% founding_year** across both tiers. The new mix removes the methodological mono-culture and means model accuracy at obscure tiers reflects breadth-of-knowledge across multiple fact-property types, not just one.

## Recommended paper structure

**Main body** (Limitations / Methodology section, ~1 page):
- Headline numbers: ambiguity defect rate, per-fact-type Wikidata reliability table
- Two illustrative cases:
  1. **L'Homme qui marche I** → Wikidata creator field returns the foundry "Susse Frères" instead of Giacometti — a systematic confusion of `P170` semantics
  2. **Loaita Cay** → claimed by 4 countries (Philippines, China, Taiwan, Vietnam); Wikidata picks one; reasonable models pick a different one based on which holds it physically vs which claims it
- Methodological note: corrections require re-calibration (in our audit all 5 corrected probes shifted T7→T5)
- Mono-culture finding: founding-year was 100% of T6/T7 wikidata before this audit

**Appendix** (~3-4 pages):
- Five-bucket failure-mode taxonomy (Wikidata-wrong / Pipeline-bug / Probe-ambiguity / Politically-contested / Multi-actor)
- Full audit telemetry table (per-round counts)
- Per-fact-type reliability ranking with example failures
- Expanded case studies: Putnam-class bare-name ambiguity, Madonna-and-Child title collision, Roku stale HQ, Höllental Railway date-extraction-bug
- Final replacement-pool fact-type composition table

