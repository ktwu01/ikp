#!/usr/bin/env python3
"""Assemble the final IKP probe dataset from all sources.

Sources:
1. LLM probes (batch 1-3): Calibrated against 4-model ladder (T1-T4)
2. Researcher probes (v3, v4): Citation-verified tiers from OpenAlex (T3-T7)
3. Wikidata probes: Sitelink-calibrated tiers (T5-T7)

Quality controls:
- Remove duplicate questions
- Remove T5+ LLM probes (answers may be wrong — no model confirmed them)
- Trim oversized tiers to target sizes
- Verify consistent schema
- Check for answer-in-question leaks
"""

import json
import random
import hashlib
import logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Target sizes per tier
TIER_TARGETS = {
    "T1": 200,
    "T2": 200,
    "T3": 200,
    "T4": 200,
    "T5": 200,
    "T6": 200,
    "T7": 200,
}


def normalize_probe(probe, source_name, tier=None):
    """Normalize a probe to a consistent schema."""
    # Determine the question field
    question = probe.get("question") or probe.get("question_direct", "")
    answer = probe.get("answer", "")

    if not question or not answer:
        return None

    # Determine source type
    if "researcher_name" in probe:
        source_type = "researcher"
    elif "sitelinks" in probe or "sitelink_count" in probe or "wikidata_id" in probe:
        source_type = "wikidata"
    else:
        source_type = "llm"

    tier = tier or probe.get("tier", "unassigned")

    normalized = {
        "question": question.strip(),
        "answer": answer.strip(),
        "tier": tier,
        "source": source_name,
        "source_type": source_type,
        "answer_type": probe.get("answer_type", "text"),
        "domain": probe.get("domain", probe.get("category", "general")),
    }

    # Copy relevant metadata
    if source_type == "researcher":
        normalized["researcher_name"] = probe.get("researcher_name", "")
        normalized["citation_count"] = probe.get("citation_count", 0)
        normalized["question_fill_blank"] = probe.get("question_fill_blank", "")
    elif source_type == "wikidata":
        normalized["wikidata_id"] = probe.get("wikidata_id", "")
        sitelinks = probe.get("sitelinks", probe.get("sitelink_count", 0))
        normalized["sitelink_count"] = sitelinks
        normalized["entity_type"] = probe.get("entity_type", "")

    return normalized


def check_answer_in_question(q, a):
    """Check if the answer leaks into the question."""
    q_lower = q.lower()
    a_lower = a.lower().strip()
    # Check for direct inclusion (but allow short answers like numbers or abbreviations)
    if len(a_lower) > 3 and a_lower in q_lower:
        return True
    return False


def deduplicate(probes):
    """Remove duplicate questions (keep first occurrence)."""
    seen = set()
    unique = []
    for p in probes:
        key = p["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    all_probes = []

    # Load LLM probes (batches 1-3, calibrated)
    for batch_file, batch_name in [
        ("llm_probes_calibrated.json", "llm_batch1"),
        ("llm_probes_batch2_calibrated.json", "llm_batch2"),
        ("llm_probes_batch3_calibrated.json", "llm_batch3"),
    ]:
        try:
            probes = json.load(open(PROJECT_ROOT / "data" / "probes" / batch_file))
            for p in probes:
                tier = p.get("tier", "unassigned")
                if tier == "T5+":
                    continue  # Skip unconfirmed probes
                if tier == "unassigned":
                    continue
                norm = normalize_probe(p, batch_name)
                if norm:
                    all_probes.append(norm)
            logger.info(f"  {batch_name}: loaded {len(probes)} -> {sum(1 for p in probes if p.get('tier') not in ['T5+','unassigned'])} usable")
        except Exception as e:
            logger.warning(f"  {batch_name}: {e}")

    # Load researcher probes
    for res_file, res_name in [
        ("researcher_field_probes_v3.json", "researcher_v3"),
        ("researcher_field_probes_v4.json", "researcher_v4"),
    ]:
        try:
            probes = json.load(open(PROJECT_ROOT / "data" / "probes" / res_file))
            for p in probes:
                norm = normalize_probe(p, res_name)
                if norm:
                    all_probes.append(norm)
            logger.info(f"  {res_name}: loaded {len(probes)}")
        except Exception as e:
            logger.warning(f"  {res_name}: {e}")

    # Load Wikidata probes
    try:
        probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "wikidata_diverse_probes.json"))
        for p in probes:
            norm = normalize_probe(p, "wikidata")
            if norm:
                all_probes.append(norm)
        logger.info(f"  wikidata: loaded {len(probes)}")
    except Exception as e:
        logger.warning(f"  wikidata: {e}")

    logger.info(f"\nTotal raw probes: {len(all_probes)}")

    # Deduplicate
    all_probes = deduplicate(all_probes)
    logger.info(f"After dedup: {len(all_probes)}")

    # Remove answer-in-question leaks
    clean_probes = []
    leaks = 0
    for p in all_probes:
        if check_answer_in_question(p["question"], p["answer"]):
            leaks += 1
        else:
            clean_probes.append(p)
    all_probes = clean_probes
    logger.info(f"Removed {leaks} answer-in-question leaks, now {len(all_probes)}")

    # Tier distribution before trimming
    tier_counts = Counter(p["tier"] for p in all_probes)
    logger.info(f"\nPre-trim tier distribution:")
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        logger.info(f"  {t}: {tier_counts.get(t, 0)}")

    # Trim oversized tiers (random sample to maintain diversity)
    final_probes = []
    for tier in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        tier_probes = [p for p in all_probes if p["tier"] == tier]
        target = TIER_TARGETS.get(tier, 200)

        if len(tier_probes) > target:
            # Stratified trim: keep proportional representation of source types
            by_source_type = defaultdict(list)
            for p in tier_probes:
                by_source_type[p["source_type"]].append(p)

            selected = []
            remaining_target = target
            source_types = list(by_source_type.keys())

            # First pass: allocate proportionally
            for st in source_types:
                n = int(target * len(by_source_type[st]) / len(tier_probes))
                random.shuffle(by_source_type[st])
                selected.extend(by_source_type[st][:n])
                by_source_type[st] = by_source_type[st][n:]
                remaining_target -= n

            # Second pass: fill remaining from largest pool
            all_remaining = []
            for st in source_types:
                all_remaining.extend(by_source_type[st])
            random.shuffle(all_remaining)
            selected.extend(all_remaining[:remaining_target])

            tier_probes = selected[:target]

        final_probes.extend(tier_probes)

    # Assign sequential IDs
    for i, p in enumerate(final_probes):
        p["id"] = f"IKP_{p['tier']}_{i:04d}"

    # Final statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL DATASET")
    logger.info(f"{'='*60}")
    logger.info(f"Total probes: {len(final_probes)}")

    tier_counts = Counter(p["tier"] for p in final_probes)
    source_counts = Counter(p["source_type"] for p in final_probes)

    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        tp = [p for p in final_probes if p["tier"] == t]
        sources = Counter(p["source_type"] for p in tp)
        logger.info(f"  {t}: {len(tp):4d}  (llm={sources.get('llm',0)}, researcher={sources.get('researcher',0)}, wikidata={sources.get('wikidata',0)})")

    logger.info(f"\nSource types: {dict(source_counts)}")

    # Domain diversity
    domains = Counter(p.get("domain", "general") for p in final_probes)
    logger.info(f"\nDomains (top 15):")
    for d, n in domains.most_common(15):
        logger.info(f"  {d}: {n}")

    # Quality checks
    logger.info(f"\nQuality checks:")

    # Check answer lengths
    short_answers = sum(1 for p in final_probes if len(p["answer"]) <= 1)
    long_answers = sum(1 for p in final_probes if len(p["answer"]) > 50)
    logger.info(f"  Very short answers (<=1 char): {short_answers}")
    logger.info(f"  Long answers (>50 chars): {long_answers}")

    # Check for duplicate answers across tiers (same answer in multiple tiers)
    answer_tiers = defaultdict(set)
    for p in final_probes:
        answer_tiers[p["answer"].lower()].add(p["tier"])
    cross_tier = {a: ts for a, ts in answer_tiers.items() if len(ts) > 1}
    logger.info(f"  Cross-tier duplicate answers: {len(cross_tier)}")
    for a, ts in list(cross_tier.items())[:5]:
        logger.info(f"    '{a}' in {sorted(ts)}")

    # Save
    output = PROJECT_ROOT / "data" / "probes" / "final_probe_set_v6.json"
    with open(output, "w") as f:
        json.dump(final_probes, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved {len(final_probes)} probes to {output}")

    # Also save a clean CSV for analysis
    import csv
    csv_output = PROJECT_ROOT / "data" / "probes" / "final_probe_set_v6.csv"
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tier", "source_type", "question", "answer", "domain"])
        writer.writeheader()
        for p in final_probes:
            writer.writerow({k: p.get(k, "") for k in ["id", "tier", "source_type", "question", "answer", "domain"]})
    logger.info(f"Saved CSV to {csv_output}")


if __name__ == "__main__":
    main()
