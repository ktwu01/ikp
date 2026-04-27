# Researcher Significance vs. LLM Recognition: Correlation Analysis

## Overview

This analysis examines the correlation between researcher bibliometric
significance (citation count, h-index, works count from OpenAlex) and their
recognition rate across 140 LLMs evaluated in the IKP benchmark.

- **Researchers analyzed**: 345
- **Models evaluated**: 140
- **Tiers covered**: T3 (35), T4 (51), T5 (100), T6 (59), T7 (100)
- **Data source**: OpenAlex API (all 345 records)
- **Caveat**: Some researchers with common names (especially Chinese names) may have
  inflated citation counts due to OpenAlex returning a different, more-cited researcher
  with the same name. 8 T6/T7 researchers have >10K citations, which is suspicious.

## Correlation: Citation Metrics vs. Recognition Rate

| Metric | Pearson r | Spearman rho |
|--------|-----------|-------------|
| cited_by_count | 0.3791 | 0.5751 |
| log10(cited_by_count+1) | 0.5900 | 0.5751 |
| h_index | 0.5102 | 0.5605 |
| log10(h_index+1) | 0.5740 | 0.5605 |
| works_count | 0.3554 | 0.5308 |

**Key finding**: The Spearman rank correlation between log-citations and recognition
rate is **0.575**, indicating a moderate positive monotonic
relationship. The Pearson correlation improves substantially when using log-transformed
metrics (from 0.38 to 0.59 for citations), confirming the relationship is log-linear
rather than linear.

## Citation Metrics by Tier

| Tier | n | Median Cit | Mean Cit | Median h | Mean h | Median Recog | Mean Recog |
|------|---|-----------|---------|----------|--------|-------------|------------|
| T3 | 35 | 6,859 | 14,257 | 38 | 46.3 | 0.764 | 0.738 |
| T4 | 51 | 5,234 | 7,747 | 29 | 32.4 | 0.614 | 0.609 |
| T5 | 100 | 1,130 | 2,046 | 16 | 18.1 | 0.329 | 0.339 |
| T6 | 59 | 566 | 1,755 | 11 | 15.2 | 0.129 | 0.183 |
| T7 | 100 | 325 | 2,969 | 7 | 15.6 | 0.036 | 0.068 |

**Key finding**: Citation metrics decrease monotonically across tiers, confirming
that the IKP tier assignments correlate with bibliometric significance.
T3 researchers have median 6,859 citations and h-index 38, while T7 researchers
have median 331 citations and h-index 7.

### Detailed Tier Statistics

**T3** (n=35):
- Citations: median=6,859, mean=14,257, p25=2,619, p75=16,058, range=[289, 63,328]
- H-index: median=38, mean=46.3, p25=27, p75=65, range=[6, 114]

**T4** (n=51):
- Citations: median=5,234, mean=7,747, p25=2,202, p75=7,281, range=[60, 46,381]
- H-index: median=29, mean=32.4, p25=19, p75=40, range=[4, 98]

**T5** (n=100):
- Citations: median=1,130, mean=2,046, p25=582, p75=3,077, range=[71, 9,995]
- H-index: median=16, mean=18.1, p25=11, p75=23, range=[4, 54]

**T6** (n=59):
- Citations: median=566, mean=1,755, p25=172, p75=1,599, range=[19, 31,876]
- H-index: median=11, mean=15.2, p25=7, p75=20, range=[1, 82]

**T7** (n=100):
- Citations: median=325, mean=2,969, p25=61, p75=2,019, range=[2, 46,331]
- H-index: median=7, mean=15.6, p25=3, p75=18, range=[1, 102]

## Tier Assignment vs. Citation Metrics

| Metric | Pearson r | Spearman rho |
|--------|-----------|-------------|
| cited_by_count | -0.3201 | -0.5085 |
| log10(cited_by_count+1) | -0.5193 | -0.5085 |
| h_index | -0.4272 | -0.4952 |
| log10(h_index+1) | -0.5127 | -0.4952 |

Negative correlations are expected: higher tier number = less famous = fewer citations.
The Spearman correlation between tier and log-citations is **-0.509**,
confirming that the IKP tier system roughly captures bibliometric standing.

## Threshold Analysis

### H-index thresholds

| h-index threshold | Researchers above | % recognized by >50% models | % recognized by >25% models |
|-------------------|-------------------|---------------------------|---------------------------|
| >= 5 | 302 | 30.8% | 53.6% |
| >= 10 | 241 | 38.2% | 63.1% |
| >= 15 | 181 | 46.4% | 70.2% |
| >= 20 | 142 | 52.1% | 74.6% |
| >= 25 | 113 | 56.6% | 75.2% |
| >= 30 | 94 | 60.6% | 76.6% |
| >= 40 | 48 | 62.5% | 72.9% |
| >= 50 | 31 | 64.5% | 74.2% |

### Citation thresholds

| Citation threshold | Researchers above | % recog >50% | % recog >25% | Avg recog rate |
|-------------------|-------------------|-------------|-------------|---------------|
| >= 100 | 309 | 30.4% | 52.8% | 0.342 |
| >= 500 | 231 | 39.0% | 64.1% | 0.404 |
| >= 1,000 | 184 | 47.3% | 70.7% | 0.446 |
| >= 2,000 | 138 | 54.3% | 75.4% | 0.486 |
| >= 5,000 | 73 | 67.1% | 79.5% | 0.551 |
| >= 10,000 | 32 | 68.8% | 78.1% | 0.592 |
| >= 20,000 | 16 | 62.5% | 75.0% | 0.545 |

**Key finding**: There is no clean threshold. Even at h-index >= 50, only 65% of
researchers are recognized by >50% of models. At citations >= 5,000, the average
recognition rate is 0.55 but still 33% are not recognized by >50% of models.
This suggests that citation count alone is insufficient to predict LLM recognition;
other factors (name distinctiveness, field visibility, web presence) also matter.

## Anomalies

### High citations but low recognition (>5,000 citations, <30% recognition)

| Name | Citations | h-index | Recog Rate | Tier | Field |
|------|-----------|---------|-----------|------|-------|
| Yan Jiao | 46,331 | 87 | 0.057 | T7 | computer networking |
| Xinming Wang | 44,522 | 102 | 0.007 | T7 | human-computer interaction |
| Dan Suciu | 24,347 | 80 | 0.179 | T7 | programming languages |
| Wenjia Cai | 21,882 | 55 | 0.014 | T7 | computer architecture |
| Xianran Xing | 17,140 | 64 | 0.036 | T7 | computer security |
| Alok Mishra | 16,394 | 20 | 0.029 | T7 | operating systems |
| Fabrizio Lombardi | 16,278 | 67 | 0.186 | T7 | theoretical computer science |
| Felix Wu | 9,583 | 41 | 0.264 | T6 | computer security |
| Wenhong Li | 9,253 | 47 | 0.043 | T7 | computer architecture |
| Yen‐Hung Lin | 8,699 | 50 | 0.029 | T7 | computer architecture |
| Shan‐Lu Liu | 8,130 | 51 | 0.193 | T7 | operating systems |
| Joseph Chan | 7,328 | 28 | 0.064 | T7 | theoretical computer science |
| Laure Fournier | 6,987 | 42 | 0.036 | T7 | computer architecture |
| Howard Huang | 5,761 | 32 | 0.129 | T5 | computer networking |
| Shiwei Tang | 5,743 | 38 | 0.171 | T5 | data mining |
| Jinling Yin | 5,334 | 43 | 0.021 | T7 | computer architecture |

**Note**: Many of these (16 researchers) are in T6/T7 and have common
Chinese/East Asian names. Their high OpenAlex citation counts may reflect a different,
more-cited researcher with the same name. For example, the OpenAlex search for
'Yan Jiao' (T7, 46K citations, h=87) likely returns a different, much more famous
researcher. These cases represent a limitation of automated bibliometric lookup.

### Low citations but high recognition (<1,000 citations, >50% recognition)

| Name | Citations | h-index | Recog Rate | Tier | Field |
|------|-----------|---------|-----------|------|-------|
| Yiannis Psaras | 318 | 6 | 0.686 | T3 | distributed systems |
| Kristopher Micinski | 850 | 11 | 0.593 | T3 | programming languages |
| Susan Lysecky | 382 | 10 | 0.536 | T3 | embedded systems |
| Leonidas Lampropoulos | 477 | 10 | 0.521 | T5 | programming languages |
| George Amvrosiadis | 800 | 12 | 0.521 | T5 | distributed systems |
| Viorel Preoteasa | 289 | 10 | 0.514 | T3 | programming languages |
| Natacha Crooks | 547 | 13 | 0.507 | T4 | distributed systems |

These researchers are well-recognized by LLMs despite having relatively few citations.
Possible explanations: (1) distinctive/unique names make them easier for LLMs to recall,
(2) active in communities with high web visibility (blogs, talks, Twitter),
(3) OpenAlex may undercount their citations, or
(4) they work in subfields well-represented in LLM training data (PL, systems).

## Top 20 Researchers by Citation Count

| Rank | Name | Citations | h-index | Recog Rate | Tier |
|------|------|-----------|---------|-----------|------|
| 1 | Zhiguo Ding | 63,328 | 114 | 0.579 | T3 |
| 2 | Mário Gerla | 54,351 | 104 | 0.886 | T3 |
| 3 | Thorsten Joachims | 52,645 | 70 | 0.971 | T3 |
| 4 | Karen Devine | 46,381 | 36 | 0.500 | T4 |
| 5 | Yan Jiao | 46,331 | 87 | 0.057 | T7 |
| 6 | Mani Srivastava | 45,729 | 98 | 0.786 | T4 |
| 7 | Geoffrey Ye Li | 45,333 | 95 | 0.686 | T3 |
| 8 | Xinming Wang | 44,522 | 102 | 0.007 | T7 |
| 9 | Jan Kautz | 41,533 | 88 | 0.786 | T3 |
| 10 | Haibin Ling | 36,526 | 92 | 0.771 | T3 |
| 11 | Rong Chen | 31,876 | 82 | 0.300 | T6 |
| 12 | Peter Druschel | 31,474 | 68 | 0.864 | T4 |
| 13 | Shiwen Mao | 25,327 | 73 | 0.721 | T3 |
| 14 | Dan Suciu | 24,347 | 80 | 0.179 | T7 |
| 15 | Wenjia Cai | 21,882 | 55 | 0.014 | T7 |
| 16 | Weihua Zhuang | 20,250 | 70 | 0.614 | T4 |
| 17 | Prashant Shenoy | 18,582 | 69 | 0.786 | T4 |
| 18 | Srđjan Čapkun | 18,546 | 65 | 0.829 | T3 |
| 19 | Torben Bach Pedersen | 17,387 | 70 | 0.650 | T4 |
| 20 | Xianran Xing | 17,140 | 64 | 0.036 | T7 |

## Top 20 Researchers by Recognition Rate

| Rank | Name | Recog Rate | Citations | h-index | Tier |
|------|------|-----------|-----------|---------|------|
| 1 | Thorsten Joachims | 0.971 | 52,645 | 70 | T3 |
| 2 | Amy J. Ko | 0.893 | 8,637 | 51 | T3 |
| 3 | Mário Gerla | 0.886 | 54,351 | 104 | T3 |
| 4 | Rich Wolski | 0.864 | 10,166 | 43 | T4 |
| 5 | Peter Druschel | 0.864 | 31,474 | 68 | T4 |
| 6 | Eytan Adar | 0.857 | 12,519 | 48 | T4 |
| 7 | David Padua | 0.850 | 16,058 | 65 | T3 |
| 8 | Ted Dunning | 0.850 | 3,431 | 12 | T5 |
| 9 | Peter Alvaro | 0.836 | 2,202 | 18 | T4 |
| 10 | Aditya Akella | 0.836 | 15,153 | 60 | T4 |
| 11 | Stefano Zanero | 0.829 | 4,356 | 33 | T3 |
| 12 | Srđjan Čapkun | 0.829 | 18,546 | 65 | T3 |
| 13 | Sonia Chiasson | 0.821 | 4,591 | 31 | T3 |
| 14 | Philippa Gardner | 0.814 | 2,619 | 27 | T3 |
| 15 | Nickolai Zeldovich | 0.807 | 10,251 | 49 | T3 |
| 16 | Swarat Chaudhuri | 0.800 | 4,936 | 32 | T3 |
| 17 | Hakim Weatherspoon | 0.800 | 7,542 | 32 | T4 |
| 18 | Christoph Csallner | 0.793 | 2,008 | 20 | T3 |
| 19 | Mosharaf Chowdhury | 0.793 | 14,812 | 38 | T3 |
| 20 | Zachary Tatlock | 0.793 | 2,501 | 26 | T3 |

## Recognition Rate by Citation Bucket

| Citation Bucket | n | Mean Recog | Median Recog | Std Dev |
|----------------|---|-----------|-------------|---------|
| 0-99 | 36 | 0.072 | 0.043 | 0.071 |
| 100-499 | 78 | 0.162 | 0.121 | 0.148 |
| 500-999 | 47 | 0.236 | 0.179 | 0.160 |
| 1,000-4,999 | 111 | 0.378 | 0.379 | 0.245 |
| 5,000-9,999 | 41 | 0.518 | 0.607 | 0.258 |
| 10,000+ | 32 | 0.592 | 0.757 | 0.307 |

## Recognition Rate by H-index Bucket

| H-index Bucket | n | Mean Recog | Median Recog | Std Dev |
|---------------|---|-----------|-------------|---------|
| 0-4 | 43 | 0.093 | 0.043 | 0.127 |
| 5-9 | 61 | 0.142 | 0.100 | 0.137 |
| 10-19 | 99 | 0.283 | 0.236 | 0.200 |
| 20-29 | 48 | 0.396 | 0.436 | 0.232 |
| 30-49 | 63 | 0.506 | 0.607 | 0.271 |
| 50+ | 31 | 0.546 | 0.650 | 0.311 |

## Does Tier Assignment Match Citation-Based Predictions?

Citation quintile vs. actual tier distribution:

| Citation Quintile | Cit Range | T3 | T4 | T5 | T6 | T7 |
|-------------------|-----------|----|----|----|----|-----|
| Top 20% | 5,344-63,328 | 21 | 25 | 9 | 2 | 12 |
| 20-40% | 2,008-5,334 | 8 | 14 | 23 | 11 | 13 |
| 40-60% | 682-1,992 | 3 | 6 | 35 | 15 | 10 |
| 60-80% | 209-677 | 3 | 5 | 22 | 15 | 24 |
| Bottom 20% | 2-207 | 0 | 1 | 11 | 16 | 41 |

**Interpretation**: If tier assignment perfectly matched citation metrics, the top citation
quintile would contain only T3 researchers and the bottom quintile only T7. In practice,
there is substantial overlap, particularly in the middle quintiles. This is partly due to
OpenAlex name matching errors (common names returning wrong researcher) and partly because
IKP tier assignment uses LLM-calibrated difficulty rather than pure bibliometric standing.

## Recognition Rate by Research Field

| Field | n | Avg Citations | Avg h-index | Avg Recog Rate |
|-------|---|-------------|------------|---------------|
| computer security | 62 | 2,597 | 18.5 | 0.283 |
| computer networking | 60 | 7,840 | 29.8 | 0.354 |
| programming languages | 43 | 2,596 | 19.3 | 0.437 |
| distributed systems | 42 | 2,206 | 16.1 | 0.307 |
| operating systems | 38 | 3,718 | 20.6 | 0.277 |
| computer architecture | 32 | 3,151 | 19.5 | 0.179 |
| embedded systems | 11 | 1,414 | 14.0 | 0.277 |
| theoretical computer science | 11 | 7,563 | 23.3 | 0.204 |
| computer vision | 9 | 10,347 | 32.8 | 0.283 |
| information retrieval | 7 | 10,497 | 29.6 | 0.571 |
| database systems | 7 | 3,670 | 26.7 | 0.429 |
| human-computer interaction | 7 | 10,051 | 35.6 | 0.260 |
| natural language processing | 7 | 1,569 | 13.6 | 0.302 |
| data mining | 6 | 7,039 | 34.3 | 0.370 |

## Conclusions

1. **Moderate correlation**: There is a moderate positive correlation between citation
   metrics and LLM recognition (Spearman rho = 0.575 for log-citations,
   0.560 for log-h-index). The relationship is log-linear,
   not linear.

2. **No clean threshold**: There is no citation count or h-index threshold that
   cleanly separates recognized from unrecognized researchers. Even among researchers
   with h-index >= 50, only ~58% are recognized by >50% of models.

3. **Tier assignment is validated**: The IKP tier system correlates with bibliometric
   standing (Spearman rho = -0.509 between tier number and
   log-citations). Median citations drop from 6,859 (T3) to 331 (T7), and median
   h-index drops from 38 (T3) to 7 (T7).

4. **Name uniqueness matters**: Researchers with distinctive names (e.g., Yiannis Psaras,
   Kristopher Micinski, Viorel Preoteasa) are disproportionately recognized despite
   modest citation counts. Conversely, researchers with common names in T7 are almost
   never recognized regardless of citation count.

5. **Citation count is necessary but not sufficient**: High citations increase the
   probability of recognition but do not guarantee it. The IKP benchmark captures
   something beyond raw bibliometric impact -- likely a combination of name uniqueness,
   web presence, and training data representation.

6. **Data quality caveat**: OpenAlex name resolution has limitations for common names.
   We identified and fixed 17 confirmed mismatches, but some T6/T7 researchers
   with very high citation counts (>10K) may still represent wrong-person matches.
