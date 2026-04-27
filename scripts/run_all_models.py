#!/usr/bin/env python3
"""Run IKP probes against ALL models (calibration + target).

Uses the comprehensive all_models.json config. Processes models cheapest-first
to maximize data early. Fully resumable via caching and checkpoint saves.

Usage:
  python scripts/run_all_models.py                    # Run all 118 models
  python scripts/run_all_models.py --max-models 20    # Run first 20 only
  python scripts/run_all_models.py --vendor openai     # Run only OpenAI models
  python scripts/run_all_models.py --skip-existing     # Skip models with results files
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from probe_runner import run_model_probes, load_all_probes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "all_models_run.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run IKP probes against all models")
    parser.add_argument("--probe-stage", type=str, default="candidates")
    parser.add_argument("--probes-per-tier", type=int, default=50)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--vendor", type=str, default=None, help="Filter by vendor")
    parser.add_argument("--type", type=str, default=None, help="Filter: open or proprietary")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models with existing results")
    parser.add_argument("--rpm", type=int, default=50, help="Requests per minute")
    args = parser.parse_args()

    # Load all models
    with open(PROJECT_ROOT / "configs" / "all_models.json") as f:
        all_models = json.load(f)["models"]

    # Load and subsample probes
    probes = load_all_probes(stage=args.probe_stage)
    if not probes:
        logger.error("No probes found")
        sys.exit(1)

    random.seed(42)
    by_tier = {}
    for p in probes:
        t = p["tier"]
        if t not in by_tier:
            by_tier[t] = []
        by_tier[t].append(p)

    probes = []
    for tier in sorted(by_tier.keys()):
        tp = by_tier[tier]
        n = min(args.probes_per_tier, len(tp))
        probes.extend(random.sample(tp, n))
        logger.info(f"  {tier}: {n}/{len(tp)} probes")

    logger.info(f"Total probes: {len(probes)}")

    # Filter models
    selected = dict(all_models)
    if args.vendor:
        selected = {k: v for k, v in selected.items() if v["vendor"] == args.vendor}
    if args.type:
        selected = {k: v for k, v in selected.items() if v["type"] == args.type}

    # Skip existing
    output_dir = PROJECT_ROOT / "data" / "raw_responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        existing = set()
        for f in output_dir.glob("*_results.json"):
            try:
                r = json.load(open(f))
                if r.get("total_probes", 0) >= len(probes) * 0.9:
                    existing.add(r["model_name"])
            except:
                pass
        before = len(selected)
        selected = {k: v for k, v in selected.items() if k not in existing}
        logger.info(f"Skipping {before - len(selected)} models with existing results")

    # Sort: cheapest models first (to maximize throughput early)
    # Put models with known params first (calibration value), then unknown (estimation)
    def sort_key(item):
        name, info = item
        has_params = 0 if info["params_B"] is not None else 1
        return (has_params, name)

    model_list = sorted(selected.items(), key=sort_key)

    if args.max_models:
        model_list = model_list[:args.max_models]

    logger.info(f"Running {len(probes)} probes against {len(model_list)} models")
    total_calls = len(probes) * 3 * len(model_list)
    logger.info(f"Estimated API calls: {total_calls:,} ({total_calls/args.rpm:.0f} min at {args.rpm} RPM)")

    # Setup client
    client = OpenRouterClient(
        requests_per_minute=args.rpm,
        max_retries=5,
        retry_delay=10,
        timeout=120,
    )

    completed = 0
    failed = []
    start_time = time.time()

    for model_name, model_info in model_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"  [{completed+1}/{len(model_list)}] {model_name} ({model_info['vendor']}, {model_info['type']})")
        if model_info["params_B"]:
            logger.info(f"  Params: {model_info['params_B']}B ({model_info['arch']})")
        logger.info(f"{'='*60}")

        try:
            summary = run_model_probes(
                client=client,
                model_name=model_name,
                model_id=model_info["id"],
                probes=probes,
                output_dir=output_dir,
            )

            # Enrich summary with model metadata
            summary["params_billion"] = model_info["params_B"]
            summary["family"] = model_info["family"]
            summary["architecture"] = model_info["arch"]
            summary["active_params_billion"] = model_info["active_B"]
            summary["vendor"] = model_info["vendor"]
            summary["model_type"] = model_info["type"]

            # Save enriched results
            results_file = output_dir / f"{model_name}_results.json"
            with open(results_file, "w") as f:
                json.dump(summary, f, indent=2)

            completed += 1
            elapsed = time.time() - start_time
            rate = completed / (elapsed / 3600) if elapsed > 0 else 0
            remaining = len(model_list) - completed
            eta_hours = remaining / rate if rate > 0 else 0

            logger.info(
                f"  Completed {completed}/{len(model_list)} "
                f"({elapsed/3600:.1f}h elapsed, ~{eta_hours:.1f}h remaining)"
            )

        except Exception as e:
            logger.error(f"  FAILED: {model_name}: {e}")
            failed.append((model_name, str(e)))
            continue

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"  COMPLETED: {completed}/{len(model_list)} models")
    logger.info(f"  FAILED: {len(failed)} models")
    for name, err in failed:
        logger.info(f"    {name}: {err}")
    logger.info(f"  Total time: {(time.time()-start_time)/3600:.1f} hours")
    logger.info(f"  API stats: {json.dumps(client.get_stats(), indent=2)}")


if __name__ == "__main__":
    main()
