#!/usr/bin/env python3
"""Run evaluation for batch 2 of newly added models."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_evaluation import evaluate_model, logger, PROJECT_ROOT, _transcript_files

BATCH2_MODELS = [
    # Gemini
    "gemini-3.1-flash-lite", "gemini-2.5-pro",
    # Claude
    "claude-3.7-sonnet", "claude-3.7-sonnet-think",
    "claude-opus-4.1", "claude-opus-4.6-fast", "claude-sonnet-4.5",
    # OpenAI reasoning
    "o1", "o1-pro", "o3-pro",
    # GPT-4/5
    "gpt-4-turbo",
    "gpt-5", "gpt-5-mini", "gpt-5-pro",
    "gpt-5.1", "gpt-5.2", "gpt-5.2-pro", "gpt-5.3",
    "gpt-5.4-nano", "gpt-5.4-pro",
    # DeepSeek
    "deepseek-v3.1",
    # Grok
    "grok-4-fast",
]


def main():
    probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "final_probe_set_v8.json"))
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / "evaluation_summary.json"
    all_summaries = json.load(open(summary_file)) if summary_file.exists() else []
    done_models = {s['model'] for s in all_summaries}

    logger.info("=" * 60)
    logger.info(f"BATCH 2: {len(BATCH2_MODELS)} models")
    logger.info("=" * 60)

    for model_name in BATCH2_MODELS:
        if model_name in done_models:
            logger.info(f"  {model_name}: already done, skipping")
            continue

        if model_name not in config['models']:
            logger.warning(f"  {model_name}: not in config, skipping")
            continue

        model_info = config['models'][model_name]
        logger.info(f"Evaluating: {model_name} ({model_info['id']})")

        try:
            summary = evaluate_model(model_name, model_info, probes, results_dir)
            entry = {
                "model": model_name,
                "params_B": model_info.get("params_B"),
                "family": model_info.get("family"),
                "vendor": model_info.get("vendor"),
                "accuracy": summary["accuracy"],
                "raw_accuracy": summary["raw_accuracy"],
                "tier_accuracy": summary["tier_accuracy"],
                "tier_stats": summary["tier_stats"],
            }
            all_summaries.append(entry)
            with open(summary_file, "w") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"  Error: {e}", exc_info=True)

    for f in _transcript_files.values():
        f.close()

    logger.info("BATCH 2 COMPLETE")


if __name__ == "__main__":
    main()
