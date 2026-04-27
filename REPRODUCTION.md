# Reproducing every figure and table in the paper

This document maps each labelled figure and table in `paper/main.tex`
and `paper/appendix.tex` to the script that produces it, the data
files it reads, and the output path it writes. If you have
`data/results/` and `data/probes/final_probe_set_v8.json`, the whole
paper rebuilds in a few minutes; raw evaluation (running every probe
against every model) takes hours to days depending on API rate limits.

## 0. One-shot rebuild

```bash
pip install -r requirements.txt

# Re-draw every figure and re-emit every .tex table that the paper uses
python paper/figures/generate_figures.py             # Figs 1–6, 8
python paper/figures/generate_appendix_figures.py    # Figs A1–A4
python scripts/loo_cv_analysis.py                    # Fig 7
python scripts/14_comprehensive_fingerprinting.py    # Fig 9 + fingerprint tables
python scripts/15_densing_law_analysis.py            # Densing CSV + appendix stats

# Rebuild the PDF
cd paper && latexmk -pdf main.tex
```

All figure scripts are idempotent and only read `data/` + `configs/`.

## 1. Inputs

| Artifact | Produced by | Read by |
|---|---|---|
| `data/probes/final_probe_set_v8.json` | `scripts/01_generate_probes.py`, `scripts/01b_generate_t6_t7.py`, `scripts/assemble_final_dataset.py` | every evaluator + `ikp_estimate.py` |
| `data/results/<model>.json` (×188) | `scripts/run_all_models.py` → `scripts/run_evaluation.py` → `src/probe_runner.py` | `evaluation_summary.json` builder + fingerprint scripts |
| `data/results/evaluation_summary.json` | `scripts/run_evaluation.py` (rewritten after each model finishes) | every figure script |
| `configs/all_models.json` | manual / `scripts/add_release_dates.py` | every figure script that needs metadata |
| `data/researcher_citations.json` | `scripts/09_researcher_probes.py` (pulls DBLP + OpenAlex) | Fig 5, Tables 3–4 |
| `data/researcher_recognition_rates.json` | `scripts/09_researcher_probes.py` | Fig 5 |
| `data/densing_analysis_data.csv` | `scripts/15_densing_law_analysis.py` | Fig 8 + Tables A5–A6 |

Earlier probe iterations and superseded runs live under
`data/probes/archive/` and `data/archive/`. One-off dev scripts
(batch-specific retries, earlier calibration utilities) live under
`scripts/legacy/`. Neither is needed to reproduce the paper — they are
kept for audit only. `scripts/README.md` is the full script index.

## 2. Main-text figures

| Label | PDF | Producer | Function | Inputs | What it shows |
|---|---|---|---|---|---|
| `fig:calibration` (Fig 1) | `paper/figures/fig1_calibration.pdf` | `paper/figures/generate_figures.py` | `fig1_calibration()` | `data/results/evaluation_summary.json`, `configs/all_models.json` | Log-linear IKP curve across 89 open-weight models (R² = 0.917), dashed projections for proprietary models |
| `fig:heatmap` (Fig 2) | `paper/figures/fig2_tier_heatmap.pdf` | `paper/figures/generate_figures.py` | `fig2_tier_heatmap()` | `data/results/evaluation_summary.json` | Per-tier accuracy heatmap, top 25 models |
| `fig:thinking` (Fig 3) | `paper/figures/fig3_thinking_effect.pdf` | `paper/figures/generate_figures.py` | `fig3_thinking_effect()` | `data/results/evaluation_summary.json` | Per-pair Δaccuracy for `-think` vs base |
| `fig:moe` (Fig 4) | `paper/figures/fig4_moe_params.pdf` | `paper/figures/generate_figures.py` | `fig4_moe_params()` | `data/results/evaluation_summary.json`, `configs/all_models.json` | MoE total-vs-active parameter fits (R² 0.79 vs 0.51) |
| `fig:researcher` (Fig 5) | `paper/figures/fig5_researcher_citations.pdf` | `paper/figures/generate_figures.py` | `fig5_researcher_scatter()` | `data/researcher_citations.json`, `data/researcher_recognition_rates.json` | Recognition rate vs log citation count (Spearman ρ = 0.575) |
| `fig:fingerprint` (Fig 6) | `paper/figures/fig6_fingerprint_heatmap.pdf` | `paper/figures/generate_figures.py` | `fig6_fingerprint_heatmap()` | `data/results/*.json`, `configs/all_models.json` | T5–T6 hallucination-similarity Jaccard heatmap, 15 frontier models |
| `fig:loo` (Fig 7) | `paper/figures/fig7_loo_validation.pdf` | `scripts/loo_cv_analysis.py` | `__main__` | `data/results/evaluation_summary.json`, `configs/all_models.json` | Predicted vs actual log-parameters under leave-one-out CV |
| `fig:densing` (Fig 8) | `paper/figures/fig8_densing_law.pdf` | `paper/figures/generate_figures.py` | `fig8_densing_law()` | `data/densing_analysis_data.csv` | Residuals of IKP vs time — refutes the Densing prediction of +0.0132/month |
| `fig:lineage` (Fig 9) | `paper/figures/fig9_family_lineage.pdf` | `scripts/14_comprehensive_fingerprinting.py` | `plot_family_lineage()` | `data/results/*.json`, `configs/all_models.json` | Per-family trajectory of HSS vs Jaccard with retrain / lineage / shared-base labels |

**Note:** `paper/figures/generate_figures.py` writes figures straight
into `paper/figures/`; `loo_cv_analysis.py` and
`14_comprehensive_fingerprinting.py` also write there. The
`results/figures/` directory holds earlier-draft copies (e.g.
`fig1_calibration_curve.pdf`) — nothing in the current paper pulls
from it.

## 3. Appendix figures

| Label | PDF | Producer | Function | Inputs |
|---|---|---|---|---|
| `fig:tier-boxplots` (Fig A1) | `paper/figures/fig_a1_tier_boxplots.pdf` | `paper/figures/generate_appendix_figures.py` | `fig_a1_tier_boxplots()` | `data/results/evaluation_summary.json` |
| `fig:vendor-hallucination` (Fig A2) | `paper/figures/fig_a2_vendor_hallucination.pdf` | same | `fig_a2_vendor_hallucination()` | `data/results/evaluation_summary.json` |
| `fig:gen-trajectories` (Fig A3) | `paper/figures/fig_a3_generation_trajectories.pdf` | same | `fig_a3_generation_trajectories()` | `data/results/evaluation_summary.json` |
| `fig:gpt5-family` (Fig A4) | `paper/figures/fig_a4_gpt5_family.pdf` | same | `fig_a4_gpt5_family()` | `data/results/evaluation_summary.json` |

## 4. Tables

### Generated `.tex` files

These are emitted by scripts and pulled in via `\input{...}`:

| Label | File | Producer |
|---|---|---|
| Fingerprint within-family summary (App. D, referenced as `fp_all_families`) | `results/tables/fp_all_families.tex` | `scripts/14_comprehensive_fingerprinting.py` → `within_family_table()` / `format_series_table()` |
| Fingerprint control pairs (App. D, `fp_controls`) | `results/tables/fp_controls.tex` | same |
| Cross-vendor outliers (App. D, `fp_cross_vendor`) | `results/tables/fp_cross_vendor.tex` | `scripts/14_comprehensive_fingerprinting.py` → `cross_family_outliers()` |
| Calibration summary (`table1_calibration`) | `results/tables/table1_calibration.tex` | `scripts/loo_cv_analysis.py` (tex block) |

### Hand-authored tables (numbers drawn from scripts)

These are typeset directly in `main.tex` / `appendix.tex`; the source
numbers come from the listed script, so running it and plugging the
values in is how you refresh them.

| Label | Paper location | Source numbers from |
|---|---|---|
| `tab:scaling` | §"What determines LLM knowledge" | `scripts/loo_cv_analysis.py` (R² printouts) + `scripts/analyze_results.py` |
| `tab:frontier` | §"Frontier estimates" | `scripts/loo_cv_analysis.py` prediction-interval block |
| `tab:citation-hindex-buckets` | §"Researcher recognition" | `scripts/09_researcher_probes.py` + `data/researcher_citations.json` |
| `tab:recog-by-field` | §"Researcher recognition" | same |
| `tab:8cell` | §"Researcher recognition" (qualitative) | curated; cases are in `data/analysis_8cell_websearch.md` |
| `tab:full-accuracy` | Appendix §Full results | `data/results/evaluation_summary.json`; build via `scripts/analyze_results.py` |
| `tab:full-hallucination` | Appendix §Full results | same |
| `tab:densing-full` | Appendix §Densing | `scripts/15_densing_law_analysis.py` (prints a LaTeX-ready block) |
| `tab:densing-tests` | Appendix §Densing | same |

## 5. End-to-end rerun (from scratch)

If you want to rebuild *everything* from raw model queries:

```bash
# 1. Rebuild the probe set (optional; the shipped probe set is already frozen)
#    NOTE: the shipped probe set incorporates a 10-round audit of the T5–T7
#    Wikidata source (see data/notes/wikidata_quality_findings.md and
#    Appendix B of the paper). If you regenerate from scratch, the
#    pipeline/generate_wikidata.py templates do NOT yet embed the
#    grounding-in-question-template pattern that the audit established;
#    apply the audit's grounding step manually or use the shipped probe
#    set as-is.
python scripts/01_generate_probes.py
python scripts/01b_generate_t6_t7.py
python scripts/assemble_final_dataset.py

# 2. Calibrate: filter out probes too easy / too hard for the reference set
python scripts/02_run_calibration.py
python scripts/03_fit_calibration.py      # writes data/calibration/calibration_fit.json

# 3. Score every model in configs/all_models.json (long-running)
export OPENROUTER_API_KEY=sk-or-...
python scripts/run_all_models.py --skip-existing

# 4. Rebuild every figure and table
python paper/figures/generate_figures.py
python paper/figures/generate_appendix_figures.py
python scripts/loo_cv_analysis.py
python scripts/14_comprehensive_fingerprinting.py
python scripts/15_densing_law_analysis.py

# 5. Compile
cd paper && latexmk -pdf main.tex
```

Expected wall-clock, full pipeline from probes to PDF: ~24–72h
depending on the vendor rate limits you can negotiate; costs roughly
$100–$300 at OpenRouter list prices for scoring all 188 models on
1,400 probes each.

## 6. Troubleshooting

- **`evaluation_summary.json` is stale** — run `python
  scripts/run_evaluation.py --rebuild-summary` or simply score one new
  model via `run_all_models.py`; it rewrites the summary at the end of
  every model.
- **Fig 6 / Fig 9 crash with "KeyError"** — a new model entry exists in
  `configs/all_models.json` but has no `data/results/<model>.json`
  yet. Either run that model or remove it from the config.
- **Densing figure looks wrong** — `densing_analysis_data.csv` was not
  regenerated after scoring new models. Rerun
  `scripts/15_densing_law_analysis.py` to refresh the CSV.
- **Judge disagreements** — the judge is `google/gemini-3-flash-preview`
  via OpenRouter. If the model is deprecated by OpenRouter, edit
  `JUDGE_MODEL` in `scripts/ikp_estimate.py` and `src/scorer.py`. Any
  switch to a different judge needs a new calibration fit.
