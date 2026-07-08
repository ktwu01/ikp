# IKP Estimation Toolkit

`scripts/ikp_estimate.py` is a self-contained CLI that scores a model
against the 1,400-probe IKP benchmark and maps the result to an
effective parameter-count estimate via the calibrated log-linear curve
from the paper. The only external state it reads is
`data/probes/final_probe_set_v8.json`; all calibration constants are
baked into the script.

## Install

```bash
pip install -r requirements.txt
```

Python ≥ 3.10, Linux/macOS. No GPU needed — the tool only consumes API
endpoints.

## One-liner

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/ikp_estimate.py --model openai/gpt-4.1
```

The tool will (1) send a sanity-check probe (``capital of France''),
(2) fan out 16 parallel workers over the 1,400 probes, (3) judge each
answer with Gemini 3 Flash Preview, (4) print a per-tier breakdown and
an estimated parameter count. Typical cost per run: **$0.10–$3** at
OpenRouter list prices, depending on the target model and thinking
budget.

## Budgeting a run (before you spend a token)

Not sure a run fits your wallet? `scripts/ikp_budget.py` prices it
up-front — no API key needed — by multiplying the benchmark's measured
per-probe token footprint by live OpenRouter prices:

```bash
# What does a full run against gpt-4.1 cost?
python scripts/ikp_budget.py --model openai/gpt-4.1

# A thinking model, quick 200-probe sample
python scripts/ikp_budget.py --model anthropic/claude-opus-4.7 --thinking --sample 200

# "I have $10 — what can I run?"
python scripts/ikp_budget.py --model openai/gpt-4.1 --budget 10

# Compare common models at a glance
python scripts/ikp_budget.py --list
```

Example output:

```
  ╔══════════════════════════════════════════════════════════╗
  ║ IKP Budget Estimate                                      ║
  ╠══════════════════════════════════════════════════════════╣
  ║ Target:    openai/gpt-4.1                                ║
  ║ Probes:    1400  (standard mode)                         ║
  ║ Est. cost: $0.890 per run                                ║
  ╚══════════════════════════════════════════════════════════╝

  Breakdown (per run of 1400 probes):
    Target model  (  2.00/  8.00 $/Mtok in/out) :     $0.616
    Judge  (google/gemini-3-flash-preview)      :     $0.274
    Total                                       :     $0.890
```

Prices come live from OpenRouter's public `/models` endpoint; if that
call fails the tool falls back to a built-in early-2026 snapshot and
labels the output as an estimate. Add `--offline` to force the snapshot,
`--json` for machine-readable output. Every dollar figure is derived
from the same prompts and probe set the estimator actually sends, so the
budget and the run stay in sync. Costs scale linearly with `--sample`,
so if a full run is too pricey, the tool suggests a stratified sample
that fits your `--budget`.

## CLI reference

```
python scripts/ikp_estimate.py [options]
```

### Model

| Flag | Default | Purpose |
|---|---|---|
| `--model, -m MODEL` | — | Target model ID for the `/chat/completions` endpoint. E.g. `openai/gpt-4.1`, `anthropic/claude-opus-4.7`, or a custom name on your own server. |
| `--api-base URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint (OpenRouter, OpenAI, vLLM, llama-server, Together, Fireworks, …). |
| `--api-key KEY` | `$OPENROUTER_API_KEY` | Bearer token for `--api-base`. |
| `--thinking` | off | Pass `reasoning: {"effort":"medium"}` to the target model. Use for Claude `-think`, Gemini `-think`, GPT-5 thinking variants, GLM `-think`, etc. (The judge always runs with `reasoning.effort=low` regardless of this flag.) |

### Evaluation

| Flag | Default | Purpose |
|---|---|---|
| `--sample, -n N` | all 1400 | Stratified random sample: `N/7` probes per tier. Use 200–400 for a fast first pass. |
| `--workers, -w N` | 16 | Parallel requests. Lower this if your provider rate-limits. |
| `--sequential, -s` | off | Force `workers=1`. |
| `--output, -o FILE` | — | Dump full per-probe results + calibration metadata to JSON. |

### Inspection / info

| Flag | Purpose |
|---|---|
| `--inspect` | After scoring, print every probe with model answer, gold, and verdict, grouped by tier — useful for qualitative analysis. |
| `--inspect-probes` | Do **not** call the API; print the probe set by tier and exit. |
| `--show-calibration` | Print the calibration formula, reference points, and R² and exit. |

## Environment variables

| Variable | Required | Used for |
|---|---|---|
| `OPENROUTER_API_KEY` | always (for the judge) | `google/gemini-3-flash-preview` judge call |
| — | optional | If `--api-base` is not OpenRouter, also pass `--api-key` for the target endpoint |

The judge is hard-coded to OpenRouter because we want every reported
number in the paper to have been graded by the same judge. To change
the judge model, edit `JUDGE_MODEL` near the top of
`scripts/ikp_estimate.py`.

## Output format

With `--output out.json`:

```json
{
  "model": "openai/gpt-4.1",
  "api_base": "https://openrouter.ai/api/v1",
  "probes_used": 1400,
  "accuracy":      0.639,      // λ=0: correct / total, averaged per tier (no penalty)
  "raw_accuracy":  0.639,      // overall correct / total
  "estimated_params_B": 402.1,
  "tier_accuracy": {"T1": 0.99, …, "T7": 0.04},
  "tier_stats":    {"T1": {"correct":…, "total":…, "refusal":…, "wrong":…}, …},
  "calibration": {"slope": 6.790, "intercept": -0.899,
                   "n_models": 93, "r_squared": 0.910},
  "results":       [ /* 1400 per-probe records */ ]
}
```

Per-probe record:

```json
{
  "probe_id": "IKP_T3_0042",
  "tier": "T3", "domain": "scientist",
  "question": "…", "gold_answer": "…",
  "response": "…(first 500 chars)…",
  "verdict": "CORRECT" | "WRONG" | "REFUSAL"
}
```

## How the estimate is computed

1. Each probe is posted as a single user turn with the system message
   `Answer factual questions directly and concisely. If you don't know,
   say 'I don't know'.`
2. The judge returns one of `CORRECT | WRONG | REFUSAL`. By default
   `HALLUCINATION_PENALTY = 0` (λ=0): a wrong answer scores the same as a
   refusal (0), so accuracy is simply the fraction answered correctly. A
   nonzero penalty is available but is not the default (see paper Appendix
   on λ sensitivity).
3. For each tier, `tier_score = correct / total` (λ=0; not floored, since
   scores are ≥ 0 by construction).
4. `accuracy = mean(tier_score)` across all seven tiers.
5. `log10(params_B) = 6.790 · accuracy − 0.899` (constants fixed;
   see `CALIB_SLOPE`, `CALIB_INTERCEPT`).

These constants are the OLS fit on the full 89-model open-weight
calibration set (135M–1.6T, R² = 0.910, no-penalty λ=0, LOO median fold error 1.48×,
72% within 2× and 86% within 3×). The fit is updated whenever new open-weight models
are added to `data/results/` and `scripts/loo_cv_analysis.py` is
rerun.

## Scope and known limitations

- **Snapshot.** The calibration reflects the web's factual distribution
  circa late 2024 – early 2026. A 2028 model trained on a much newer web
  snapshot will appear systematically over- or under-sized; recalibrate
  on a fresh open-weight cohort before extrapolating.
- **Landmark probes.** A handful of T6/T7 probes were proposed by
  Gemini 3 Pro and are therefore inflated for it; see
  `Fig. 1` and the "Gemini 3.1 Pro landmark" note in the paper. Exclude
  that model from any calibration refit.
- **Aggressive safety tuning.** Models that refuse heavily (e.g., some
  Claude `-think` variants on biographical queries) will be estimated
  *below* their true capacity. See the paper §"Refused-but-known" for
  the gap.
- **Tiny probe samples.** `--sample` < 100 has visibly wider
  prediction intervals. The curve is meant to be read at the full
  1,400 probe budget.
- **Thinking flag.** Passing `--thinking` to a non-reasoning model is a
  no-op on most providers, but some 400 error out. If in doubt,
  run once without.
- **Non-English knowledge.** The benchmark is English only; applying it
  to a model optimized for a single non-English language underestimates
  capacity.

## Quick recipes

```bash
# Fast first pass (~1 min, ~$0.05)
python scripts/ikp_estimate.py --model openai/gpt-4.1 --sample 140

# Self-hosted vLLM with local judge env
OPENROUTER_API_KEY=... python scripts/ikp_estimate.py \
    --api-base http://localhost:8000/v1 --api-key EMPTY \
    --model meta-llama/Meta-Llama-3-70B-Instruct

# Thinking-mode pair (two runs; compare accuracies)
python scripts/ikp_estimate.py -m anthropic/claude-opus-4.7
python scripts/ikp_estimate.py -m anthropic/claude-opus-4.7 --thinking

# Export for post-hoc analysis
python scripts/ikp_estimate.py -m openai/gpt-5 -o runs/gpt-5.json

# Look at the curve before spending any tokens
python scripts/ikp_estimate.py --show-calibration
python scripts/ikp_estimate.py --inspect-probes | head -40
```

## Batch evaluation (for contributing to the roster)

To score many models and produce per-model JSON files compatible with
`data/results/<model>.json`, use the pipeline runner instead:

```bash
python scripts/run_all_models.py --skip-existing            # full roster
python scripts/run_all_models.py --vendor openai            # single vendor
python scripts/run_all_models.py --type open --max-models 10
```

Each run appends a row to `data/results/evaluation_summary.json`, which
is what the figure scripts consume. See `REPRODUCTION.md` for how the
summary flows into the figures.
