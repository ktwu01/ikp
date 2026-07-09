# IKP paper build + data refresh.
#
# Common targets:
#   make pdf      Quick rebuild (one pdflatex pass)
#   make full     Full rebuild with bibtex (4 passes; needed for new citations)
#   make figs     Regenerate all paper figures
#   make data     Refresh calibration / analysis / website data from data/results/
#   make all      data -> figs -> pdf
#   make eval     Run scripts/run_evaluation.py (auto-skips models with existing results)
#   make budget   Estimate the $ cost of a run (make budget MODEL=openai/gpt-4.1)
#   make clean    Remove LaTeX intermediates
#   make watch    Rebuild PDF on every main.tex / appendix.tex / .bib edit (requires fswatch)
#
# Notes:
# - All commands assume cwd = repo root.
# - Figures and PDFs land in paper/ and paper/figures/.
# - Website data lands in website/public/data/.

PYTHON ?= python3
PAPER_DIR := paper
LATEX := pdflatex -interaction=nonstopmode
BIBTEX := bibtex
MAIN := main

.PHONY: help pdf full figs data all eval budget calibration website website-dev website-build website-preview website-deploy docs clean watch

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) 2>/dev/null \
	  | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}' \
	  || echo "Targets: pdf full figs data all eval clean watch"

# ── Paper build ────────────────────────────────────────────────────────────
# pdflatex returns non-zero on benign warnings (undefined refs etc.); the PDF
# is still produced. We grep for genuine errors and pass through otherwise.
pdf: ## Quick rebuild (single pdflatex pass)
	@cd $(PAPER_DIR) && $(LATEX) $(MAIN).tex > /tmp/ikp-build.log 2>&1; \
	  grep -E "^(! |Output written|LaTeX Error)" /tmp/ikp-build.log | tail -5; \
	  grep -q "^! " /tmp/ikp-build.log && exit 1 || exit 0

full: ## Full rebuild with bibtex (4 passes)
	@cd $(PAPER_DIR) && $(LATEX) $(MAIN).tex > /tmp/ikp-build.log 2>&1 || true
	@cd $(PAPER_DIR) && $(BIBTEX) $(MAIN) > /tmp/ikp-bib.log 2>&1 || true
	@cd $(PAPER_DIR) && $(LATEX) $(MAIN).tex > /tmp/ikp-build.log 2>&1 || true
	@cd $(PAPER_DIR) && $(LATEX) $(MAIN).tex > /tmp/ikp-build.log 2>&1; \
	  grep -E "^(! |Output written|LaTeX Error)" /tmp/ikp-build.log | tail -5; \
	  grep -q "^! " /tmp/ikp-build.log && exit 1 || exit 0

# ── Figures ────────────────────────────────────────────────────────────────
figs: ## Regenerate all paper figures
	$(PYTHON) $(PAPER_DIR)/figures/generate_figures.py
	$(PYTHON) $(PAPER_DIR)/figures/generate_appendix_figures.py

# ── Data refresh from current data/results/ ────────────────────────────────
data: calibration website ## Refresh all derived data + website JSON

calibration: ## Refresh calibration fits (loo_cv + analyze)
	$(PYTHON) scripts/loo_cv_analysis.py
	$(PYTHON) scripts/analyze_results.py

website: ## Refresh website public/data/*.json (run before website-build)
	$(PYTHON) website/scripts/prepare_data.py

website-dev: website ## Local dev server with hot reload (http://localhost:5173)
	cd website && npm install && npm run dev

website-build: website ## Static build → website/dist/
	cd website && npm install && npm run build

website-preview: website-build ## Preview the production build locally
	cd website && npm run preview

# Override DEPLOY_HOST / DEPLOY_PATH for your own server, e.g.
#   make website-deploy DEPLOY_HOST=user@host DEPLOY_PATH=/var/www/research/ikp/
DEPLOY_HOST ?= bojieli@01.me
DEPLOY_PATH ?= /var/www/01.me/research/ikp/
website-deploy: website-build ## rsync website/dist/ to $(DEPLOY_HOST):$(DEPLOY_PATH)
	rsync -avz --delete website/dist/ $(DEPLOY_HOST):$(DEPLOY_PATH)

# ── Full pipeline ──────────────────────────────────────────────────────────
all: data figs pdf ## Full pipeline: data refresh → figures → PDF

# ── Evaluation ─────────────────────────────────────────────────────────────
eval: ## Run model evaluation (auto-skips done models). Logs to data/eval.log.
	$(PYTHON) scripts/run_evaluation.py 2>&1 | tee data/eval.log

# ── Budget ─────────────────────────────────────────────────────────────────
# Estimate a run's $ cost before spending a token (no API key needed).
# Pass a model:  make budget MODEL=openai/gpt-4.1
# Or compare common models:  make budget   (falls back to --list)
MODEL ?=
budget: ## Estimate run cost. `make budget MODEL=openai/gpt-4.1`, else a comparison table.
	@if [ -n "$(MODEL)" ]; then \
	  $(PYTHON) scripts/ikp_budget.py --model $(MODEL); \
	else \
	  $(PYTHON) scripts/ikp_budget.py --list; \
	fi

# ── Maintenance ────────────────────────────────────────────────────────────
clean: ## Remove LaTeX intermediates (keep main.pdf)
	cd $(PAPER_DIR) && rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	  $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk

watch: ## Rebuild on edit (requires fswatch: brew install fswatch)
	@command -v fswatch >/dev/null 2>&1 || { echo "install fswatch (brew install fswatch)"; exit 1; }
	fswatch -o $(PAPER_DIR)/$(MAIN).tex $(PAPER_DIR)/appendix.tex $(PAPER_DIR)/references.bib \
	  | xargs -n1 -I{} make pdf
