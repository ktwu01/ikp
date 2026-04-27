# IKP Data Directory

This directory holds the probe set, per-model raw answers, and derived
summaries used throughout the paper.

## Canonical artifacts (used by the paper)

| File | Purpose |
|---|---|
| `probes/final_probe_set_v8.json` | **The 1,400-probe IKP benchmark.** 200 items × 7 tiers (T1–T7). This is the probe set referenced everywhere in the paper and consumed by `scripts/ikp_estimate.py`. |
| `results/<model>.json` | Per-model evaluation output — one file per scored model (168 files). Contains per-tier accuracy, hallucination rate, and per-probe verdicts. |
| `results/evaluation_summary.json` | Aggregated summary across all models (one record per model). Produced by `scripts/run_evaluation.py` / `scripts/run_all_models.py`; consumed by every figure script. |
| `calibration/calibration_fit.json` | Fitted log-linear calibration (slope, intercept, R², N). Produced by `scripts/03_fit_calibration.py`. |
| `researcher_citations.json` | Citation count + h-index per researcher entity used in T4–T7. |
| `researcher_recognition_rates.json` | Per-researcher recognition rate across all models. |
| `densing_analysis_data.csv` | Table used for the Densing Law falsification figure/tests. Produced by `scripts/15_densing_law_analysis.py`. |

## Per-probe schema (`probes/final_probe_set_v8.json`)

```json
{
  "id": "IKP_T3_0042",
  "question": "…",
  "answer": "Gold answer (string; ';' separates acceptable alternatives)",
  "tier": "T3",
  "source_type": "wikidata | llm | researcher | manual",
  "domain": "geography | scientist | …"
}
```

## Per-model result schema (`results/<model>.json`)

```json
{
  "model_name": "gpt-4.1",
  "model_id": "openai/gpt-4.1",
  "params_B": null,
  "family": "gpt",
  "vendor": "openai",
  "arch": "unknown",
  "accuracy": 0.62,           // penalized (wrong answers cost 0.5)
  "raw_accuracy": 0.68,       // unpenalized
  "hallucination_penalty": -0.5,
  "judge_model": "google/gemini-3-flash-preview",
  "correct": 952, "total": 1400,
  "tier_accuracy": {"T1": 0.99, …, "T7": 0.03},
  "tier_stats":    {"T1": {"correct":…, "total":…, "refusal":…, "wrong":…}, …},
  "probe_results": [
    {"id":"IKP_T1_0000", "tier":"T1",
     "question":"…", "gold_answer":"…", "response":"…", "verdict":"CORRECT"}
    // one entry per probe
  ]
}
```

Verdicts come from a Gemini 3 Flash Preview judge and are one of
`CORRECT`, `WRONG`, `REFUSAL`. Penalized accuracy is
`(correct − 0.5·wrong) / total`.

## Directory layout

```
data/
├── probes/
│   ├── final_probe_set_v8.json   ← THE 1,400 probes (the benchmark)
│   ├── researcher_probes.json    ← current researcher sub-probe source
│   └── archive/                  ← earlier probe iterations (v1..v7, batches, Tk candidates, …)
├── results/                      ← per-model evaluations + evaluation_summary.json
├── calibration/                  ← fitted log-linear calibration + raw calibration responses
├── researcher_citations.json
├── researcher_recognition_rates.json
├── densing_analysis_data.csv     ← built by scripts/15_densing_law_analysis.py
├── notes/                        ← exploratory analysis markdown (analysis_*.md, ANALYSIS_REPORT.md)
└── archive/                      ← superseded data snapshots (results_v7/, …)
```

## Directories you can ignore for reproduction

These are upstream scratch or regenerable caches; none are read by the
paper figures, and most are gitignored:

- `api_cache/` — per-call HTTP cache (regenerable, gitignored)
- `raw_responses/`, `chinese_responses/`, `researcher_responses/` — intermediate per-probe text dumps
- `backups/` — local snapshots of in-progress runs (gitignored)
- `archive/` — superseded run bundles (prior probe set, `results_v7/`, …)
- `probes/archive/` — legacy probe iterations (`final_probe_set_v{1..7}*.json`, `T{1..7}_*.json`, `llm_probes_*`, etc.); **v8 is the one the paper uses**
- `pipeline_store.json`, `*.log` — pipeline state and run logs
- `notes/*.md` — exploratory narrative analysis (not cited by the paper)
