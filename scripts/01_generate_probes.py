#!/usr/bin/env python3
"""Phase 1: Generate probe candidates for Tiers 1-5 using LLM.

Usage: python scripts/01_generate_probes.py [--tiers T1,T2,...] [--dry-run]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from probe_generator import (
    generate_tier_probes, validate_probe_format,
    save_probes, load_probes, TIER_DEFINITIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "generation.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Generate IKP probe candidates")
    parser.add_argument("--tiers", type=str, default="T1,T2,T3,T4,T5",
                        help="Comma-separated list of tiers to generate")
    parser.add_argument("--candidates", type=int, default=400,
                        help="Number of candidates per tier")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Probes per LLM generation call")
    parser.add_argument("--model", type=str, default=None,
                        help="Override generator model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts but don't call API")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(PROJECT_ROOT / "configs" / "experiment.json") as f:
        config = json.load(f)

    tiers = [t.strip() for t in args.tiers.split(",")]
    generator_model = args.model or config["probe_generation"]["generator_model"]

    logger.info(f"=== IKP Probe Generation ===")
    logger.info(f"Tiers: {tiers}")
    logger.info(f"Candidates per tier: {args.candidates}")
    logger.info(f"Generator model: {generator_model}")
    logger.info(f"Seed: {args.seed}")

    if args.dry_run:
        from probe_generator import build_generation_prompt
        for tier in tiers:
            prompt = build_generation_prompt(tier, 0, args.batch_size)
            print(f"\n{'='*60}")
            print(f"TIER {tier} - Batch 0 Prompt:")
            print(f"{'='*60}")
            print(prompt[:2000])
            print("...")
        return

    client = OpenRouterClient(
        requests_per_minute=config["api"]["requests_per_minute"],
        max_retries=config["api"]["max_retries"],
        retry_delay=config["api"]["retry_delay_seconds"],
        timeout=120,  # Longer timeout for generation
    )

    for tier in tiers:
        existing = load_probes(tier, "candidates")
        if len(existing) >= args.candidates * 0.8:
            logger.info(f"Skipping {tier}: already have {len(existing)} candidates")
            continue

        logger.info(f"\n--- Generating {tier} ---")
        probes = generate_tier_probes(
            client=client,
            tier=tier,
            generator_model=generator_model,
            candidates_target=args.candidates,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # Validate
        valid = []
        invalid_count = 0
        for p in probes:
            ok, msg = validate_probe_format(p)
            if ok:
                valid.append(p)
            else:
                invalid_count += 1
                logger.debug(f"Invalid probe: {msg}")

        save_probes(valid, tier, "candidates")
        logger.info(f"{tier}: {len(valid)} valid / {invalid_count} invalid / {len(probes)} total")

    stats = client.get_stats()
    logger.info(f"\nAPI stats: {json.dumps(stats, indent=2)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
