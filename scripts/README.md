# scripts/

## Public entrypoints — use these

| Script | Purpose |
|---|---|
| `ikp_estimate.py` | Score one model against the 1,400-probe set and print an estimated parameter count. See `../TOOLKIT.md`. |
| `run_all_models.py` | Batch-score every model in `configs/all_models.json` (idempotent, resumable). |
| `run_evaluation.py` | Score a single model by name; called from `run_all_models.py` but runnable standalone. |

## Dataset pipeline (numbered, reproducible)

Run in order if you want to regenerate the probe set and re-score every
model from scratch. Most users never need to run these — the data
already ships in `data/`.

| Stage | Script | Output |
|---|---|---|
| Probe gen (T1–T4) | `01_generate_probes.py`, `01b_generate_t6_t7.py` | Tier candidates |
| Calibration eval | `02_run_calibration.py` | Per-calibration-model raw answers |
| Calibration fit | `03_fit_calibration.py` | `data/calibration/calibration_fit.json` |
| Target eval | `04_run_targets.py` | Per-target-model raw answers |
| Chinese probes | `08_chinese_probes.py` | Chinese-language subset |
| Researcher probes | `09_researcher_probes.py` | T4–T7 researcher subset |
| Web-grounded | `10_web_grounded_probes.py` | Web-frequency grounded probes |
| Corpus-grounded | `11_corpus_grounded_t5t7.py` | T5–T7 corpus-grounded probes |
| Fingerprint probes | `12_fingerprint_probes.py` | Extended-phrasing subset for §D |
| Distillation det. | `13_distillation_detection.py` | Early fingerprint analysis |
| Fingerprinting | `14_comprehensive_fingerprinting.py` | Fig 9 + `results/tables/fp_*.tex` |
| Densing analysis | `15_densing_law_analysis.py` | `data/densing_analysis_data.csv` + appendix stats |
| Dataset assembly | `assemble_final_dataset.py` | `data/probes/final_probe_set_v8.json` |
| Metadata | `add_release_dates.py` | Augments `configs/all_models.json` |

## Analysis / post-hoc

| Script | Purpose |
|---|---|
| `analyze_results.py` | Aggregates per-tier accuracy, builds `data/results/evaluation_summary.json`, fits scaling curves. |
| `loo_cv_analysis.py` | Leave-one-out CV + writes Fig 7 (`paper/figures/fig7_loo_validation.pdf`). |
| `show_progress.py` | Quick text progress dump across `data/results/`. |

## legacy/

Older one-off scripts from earlier dataset iterations, batch retries,
and superseded analyses. None are required to reproduce any figure or
table in the paper; they are kept for audit/reference. `PROJECT_ROOT`
in every file under `legacy/` is written as `.parent.parent.parent`
(three `..`), so the scripts still find `data/`, `configs/`, `src/`,
etc. after the move.
