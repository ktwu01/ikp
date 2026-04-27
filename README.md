# Incompressible Knowledge Probes (IKP)

Evaluation toolkit and reproduction bundle for the paper:

> **Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter
> Counts via Factual Capacity.** Bojie Li, Pine AI.

IKP is a 1,400-question factual benchmark — 200 items × 7 obscurity tiers
(T1: universal knowledge … T7: extreme long-tail). Accuracy on IKP
scales log-linearly with parameter count across 89 open-weight models
from 135M to 1.6T (R² = 0.917), so a single black-box API call budget is
enough to estimate the effective knowledge capacity of any deployed
model — including closed-source frontier models whose sizes are
undisclosed.

- **Paper PDF:** `paper/main.pdf`
- **Companion website (interactive):** https://01.me/research/ikp
- **Source:** https://github.com/19PINE-AI/ikp

## Quickstart — estimate a model

```bash
# 1. Install deps (Python ≥ 3.10)
pip install -r requirements.txt

# 2. Point at any OpenAI-compatible endpoint and run
export OPENROUTER_API_KEY=sk-or-...
python scripts/ikp_estimate.py --model openai/gpt-4.1
```

Output:

```
  ╔══════════════════════════════════════════════════════════╗
  ║  IKP Estimation Results                                 ║
  ║  Model:     openai/gpt-4.1                              ║
  ║  Probes:    1400                                         ║
  ║  Accuracy:  58.2% (penalized)  63.9% (raw)              ║
  ║  Estimated:  400B parameters                             ║
  ╚══════════════════════════════════════════════════════════╝
  T1   99%  …  T7    4%
  Effective tier: T6
  Estimated size: 400B (calibrated on 89 open models, R²=0.917)
```

Faster stratified sample (200 probes, ~1 min):

```bash
python scripts/ikp_estimate.py --model openai/gpt-4.1 --sample 200
```

Non-OpenRouter endpoint (vLLM, OpenAI, Together, local):

```bash
python scripts/ikp_estimate.py \
    --api-base http://localhost:8000/v1 \
    --api-key  <your-key> \
    --model    my-local-model
# Judge always runs on OpenRouter (google/gemini-3-flash-preview);
# OPENROUTER_API_KEY must still be set for the judge.
```

Full CLI reference, including how to plug in a different judge or export
per-probe verdicts: see [`TOOLKIT.md`](TOOLKIT.md).

## Reproducing the paper

Every figure and table in the paper, with the exact script, inputs and
expected outputs, is listed in [`REPRODUCTION.md`](REPRODUCTION.md).

Short path:

```bash
# Fastest: regenerate all figures from already-scored results
python paper/figures/generate_figures.py
python paper/figures/generate_appendix_figures.py

# Rebuild PDF (TeX Live)
cd paper && latexmk -pdf main.tex
```

To score additional models and extend the dataset:

```bash
python scripts/run_all_models.py --skip-existing
python scripts/run_evaluation.py --rebuild-summary  # refreshes evaluation_summary.json
```

## Build the paper / website

The `Makefile` is the single entry point.

```bash
make help              # list every target

# Paper
make figs              # regenerate every figure under paper/figures/
make pdf               # one pdflatex pass (fast, no bibtex)
make full              # full rebuild with bibtex (4 passes)

# Calibration / data refresh after a new model lands in data/results/
make calibration       # rerun loo_cv_analysis.py + analyze_results.py
make website           # rebuild website/public/data/*.json (must precede website-build)
make data              # = calibration + website

# Website
make website-dev       # vite dev server  → http://localhost:5173
make website-build     # static build     → website/dist/
make website-preview   # preview the production build
make website-deploy    # rsync website/dist/ to DEPLOY_HOST:DEPLOY_PATH
                       # override per invocation:
                       #   make website-deploy DEPLOY_HOST=user@host \
                       #                       DEPLOY_PATH=/var/www/research/ikp/

make all               # data → figs → pdf
```

For subpath deploys (e.g. `https://example.com/research/ikp/`), set
`BASE_URL=/research/ikp/ make website-build`. See `website/README.md` for full
website documentation, nginx config, and GitHub Pages instructions.

## Repo layout

```
ikp-paper/
├── README.md               ← this file
├── TOOLKIT.md              ← ikp_estimate.py reference
├── REPRODUCTION.md         ← figure/table ⇄ script map
├── requirements.txt
│
├── paper/                  ← LaTeX sources
│   ├── main.tex  main.pdf  appendix.tex  references.bib
│   ├── research-plan.md    ← original planning document
│   └── figures/            ← PDF/PNG figures + generators (all main & appendix figs)
│       ├── generate_figures.py            (main-text figs 1–6, 8)
│       └── generate_appendix_figures.py   (appendix figs A1–A4)
│
├── configs/
│   ├── experiment.json     ← tier definitions, API settings, seeds
│   ├── models.json         ← calibration-set models (open, known size)
│   └── all_models.json     ← full roster (188 models evaluated)
│
├── data/                   ← see data/README.md for schemas
│   ├── probes/
│   │   ├── final_probe_set_v8.json  ← THE 1,400 probes (the benchmark)
│   │   ├── researcher_probes.json   ← researcher sub-probe source
│   │   └── archive/                 ← earlier probe versions (v1..v7, batches, candidates)
│   ├── results/<model>.json         ← per-model raw evaluations (188 files)
│   ├── results/evaluation_summary.json  ← aggregated, consumed by every figure
│   ├── calibration/calibration_fit.json ← fitted log-linear calibration
│   ├── researcher_citations.json        ← T4–T7 researcher metadata
│   ├── researcher_recognition_rates.json
│   ├── densing_analysis_data.csv        ← Densing-Law table (for Fig 8)
│   ├── notes/                           ← exploratory analysis markdown
│   └── archive/                         ← superseded runs (results_v7, …)
│
├── results/
│   ├── figures/archive/    ← early-draft plots (superseded by paper/figures/)
│   └── tables/             ← .tex tables \input'ed by the paper
│
├── scripts/                ← see scripts/README.md for a full index
│   ├── ikp_estimate.py     ← one-model estimator (public entrypoint)
│   ├── run_all_models.py   ← bulk evaluation across the full roster
│   ├── run_evaluation.py   ← single-model evaluator
│   ├── 01_..15_*.py        ← numbered dataset pipeline
│   ├── analyze_results.py, loo_cv_analysis.py, show_progress.py
│   └── legacy/             ← one-off / superseded dev scripts (kept for audit)
│
├── pipeline/               ← probe generation + calibration library
├── src/                    ← evaluation runtime (api_client, probe_runner, scorer, …)
└── website/                ← React companion site
```

All active scripts resolve paths via `Path(__file__).parent.parent`, so
they expect to live in `scripts/`. Scripts under `scripts/legacy/`
have been patched to three-`..` (`.parent.parent.parent`) and still
work when invoked directly.

## How it works (one paragraph)

Each probe is a short factual question with a gold answer, scored by a
Gemini 3 Flash Preview judge. Researcher subfield probes use a 4-way
evidence-aware judge (CORRECT_STRONG = subfield + verifiable evidence
item; CORRECT_WEAK = subfield only; REFUSAL; WRONG); other probes use
a 3-way judge (CORRECT / REFUSAL / WRONG). Penalized accuracy scores
each probe in `{+1.0, +0.5, 0, λ}` for the four classes with `λ = -1`
(WRONG); hallucinations are penalized to discourage guessing. The
calibration curve is `log10(params_B) = 6.790 · accuracy − 0.899`
(R² = 0.917 on 89 open models; LOO median fold error 1.59×, 68.5%
within 2× and 87.6% within 3×). For MoE models, *total* parameters
predict accuracy (R² = 0.79) much better than active parameters
(R² = 0.51) — so the curve is fit against total parameter count.

## Requirements

- Python ≥ 3.10
- An API key for the model(s) you want to evaluate (OpenRouter covers
  all 188 evaluated models; OpenAI-compatible endpoints also work)
- An `OPENROUTER_API_KEY` for the judge (always Gemini 3 Flash Preview)
- ~$0.10–$3 per model to score the full 1,400 probes, depending on the
  model priced at OpenRouter rates

## Citing

```bibtex
@misc{li2026ikp,
  title  = {Incompressible Knowledge Probes: Estimating Black-Box LLM
            Parameter Counts via Factual Capacity},
  author = {Bojie Li},
  year   = {2026},
  note   = {Pine AI. \url{https://01.me/research/ikp}}
}
```

## License

Code: MIT. Probe set and per-model results: CC BY 4.0.
