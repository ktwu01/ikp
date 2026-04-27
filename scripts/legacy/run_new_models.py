#!/usr/bin/env python3
"""Run evaluation for newly added models only.

Usage: python scripts/run_new_models.py [model_name ...]
If no model names given, evaluates all models in config without existing results.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_evaluation import (
    evaluate_model, logger, PROJECT_ROOT,
    _transcript_files,
)


def main():
    probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "final_probe_set_v8.json"))
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to evaluate
    if len(sys.argv) > 1:
        target_models = sys.argv[1:]
    else:
        # Find models without results
        target_models = []
        for name in config["models"]:
            result_file = results_dir / f"{name}.json"
            if not result_file.exists():
                target_models.append(name)

    logger.info("=" * 60)
    logger.info(f"Running evaluation for {len(target_models)} models: {target_models}")
    logger.info("=" * 60)

    for model_name in target_models:
        if model_name not in config['models']:
            logger.error(f"Model {model_name} not found in config, skipping")
            continue

        model_info = config['models'][model_name]
        logger.info(f"\nEvaluating: {model_name} ({model_info['id']})")

        try:
            summary = evaluate_model(model_name, model_info, probes, results_dir)
            logger.info(f"  {model_name}: accuracy={summary['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"  Error evaluating {model_name}: {e}", exc_info=True)

    # Close transcript files
    for f in _transcript_files.values():
        f.close()

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
