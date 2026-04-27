#!/usr/bin/env python3
"""Fill researcher probe gaps by querying OpenAlex for authors from the 24K pool.

Strategy:
1. Sample authors from proceedings_authors_v2.json (diverse by venue)
2. Query OpenAlex for each: citation count + research topics
3. Assign tiers based on citation count:
   T3: 5000+ citations (well-known)
   T4: 1000-4999 citations (moderately known)
   T5: 200-999 citations (niche)
   T6: 50-199 citations (obscure)
   T7: 1-49 citations (very obscure)
4. Create probes asking "What is the research field of X?"
5. Fill until we have enough per tier

Uses ThreadPoolExecutor for parallel OpenAlex queries.
"""

import json
import os
import sys
import time
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Map from venue to our standardized field names
VENUE_TO_FIELD = {
    "NSDI": "computer networking",
    "SIGCOMM": "computer networking",
    "ISCA": "computer architecture",
    "MICRO": "computer architecture",
    "ASPLOS": "computer architecture",
    "NeurIPS": "machine learning",
    "ICML": "machine learning",
    "CCS": "computer security",
    "USENIX Sec": "computer security",
    "VLDB": "database systems",
    "SIGMOD": "database systems",
    "KDD": "data mining",
    "CHI": "human-computer interaction",
    "STOC": "theoretical computer science",
    "PLDI": "programming languages",
    "POPL": "programming languages",
    "OSDI": "operating systems",
    "SOSP": "operating systems",
    "ATC": "operating systems",
    "EuroSys": "distributed systems",
    "FAST": "storage systems",
}

# OpenAlex concept-to-field mapping (level 1 concepts)
OPENALEX_CONCEPT_MAP = {
    "computer network": "computer networking",
    "computer networking": "computer networking",
    "networking": "computer networking",
    "computer architecture": "computer architecture",
    "machine learning": "machine learning",
    "artificial intelligence": "machine learning",
    "deep learning": "machine learning",
    "computer security": "computer security",
    "information security": "computer security",
    "cryptography": "computer security",
    "database": "database systems",
    "data management": "database systems",
    "data mining": "data mining",
    "information retrieval": "data mining",
    "knowledge discovery": "data mining",
    "human-computer interaction": "human-computer interaction",
    "human computer interaction": "human-computer interaction",
    "user interface": "human-computer interaction",
    "algorithm": "theoretical computer science",
    "theoretical computer science": "theoretical computer science",
    "computational complexity": "theoretical computer science",
    "combinatorics": "theoretical computer science",
    "programming language": "programming languages",
    "compiler": "programming languages",
    "software verification": "programming languages",
    "formal methods": "programming languages",
    "operating system": "operating systems",
    "distributed computing": "distributed systems",
    "distributed system": "distributed systems",
    "parallel computing": "distributed systems",
    "cloud computing": "distributed systems",
    "file system": "storage systems",
    "storage": "storage systems",
    "computer vision": "computer vision",
    "image processing": "computer vision",
    "natural language processing": "natural language processing",
    "computational linguistics": "natural language processing",
    "computer graphics": "computer graphics",
    "visualization": "computer graphics",
    "software engineering": "software engineering",
    "robotics": "robotics",
    "embedded system": "embedded systems",
    "real-time computing": "embedded systems",
    "bioinformatics": "computational biology",
    "computational biology": "computational biology",
}

# Tier boundaries based on citation count
TIER_BOUNDARIES = {
    "T3": (5000, float("inf")),
    "T4": (1000, 4999),
    "T5": (200, 999),
    "T6": (50, 199),
    "T7": (1, 49),
}

# Bad names to filter
BAD_NAME_PATTERNS = [
    "university", "institute", "laboratory", "laboratories", "inc.",
    "corp", "ltd", "gmbh", "research", "department", "school",
    "college", "center", "centre", "group", "team", "microsoft",
    "google", "facebook", "meta", "amazon", "apple", "nvidia",
    "bytedance", "tencent", "alibaba", "huawei", "samsung",
    "purple mountain", "adobe",
]


def is_valid_author_name(name: str) -> bool:
    """Filter out institution names and invalid entries."""
    if not name or len(name) < 3:
        return False
    name_lower = name.lower()
    for bad in BAD_NAME_PATTERNS:
        if bad in name_lower:
            return False
    # Must have at least 2 parts (first + last name)
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    # No single-letter parts except initials
    if any(len(p) == 1 and not p.isupper() for p in parts):
        return False
    return True


def query_openalex_author(name: str, retry_count: int = 0) -> dict:
    """Query OpenAlex for author info. Max 2 retries with backoff."""
    if retry_count >= 3:
        return None
    http = httpx.Client(timeout=30)
    try:
        r = http.get(
            "https://api.openalex.org/authors",
            params={
                "search": name,
                "per_page": 1,
                "mailto": "research@example.org",
            },
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                author = results[0]
                cited = author.get("cited_by_count", 0)
                works = author.get("works_count", 0)
                display_name = author.get("display_name", name)

                # Get top concepts for field
                concepts = author.get("x_concepts", [])
                if not concepts:
                    concepts = author.get("topics", [])

                top_field = None
                for c in concepts:
                    concept_name = c.get("display_name", "").lower()
                    if concept_name in OPENALEX_CONCEPT_MAP:
                        top_field = OPENALEX_CONCEPT_MAP[concept_name]
                        break

                return {
                    "name": display_name,
                    "citations": cited,
                    "works": works,
                    "field": top_field,
                    "concepts": [c.get("display_name", "") for c in concepts[:5]],
                }
        elif r.status_code == 429:
            time.sleep(3 * (retry_count + 1))
            return query_openalex_author(name, retry_count + 1)
        return None
    except Exception as e:
        logger.warning(f"OpenAlex error for {name}: {e}")
        return None
    finally:
        http.close()


def assign_tier(citations: int) -> str:
    for tier, (low, high) in TIER_BOUNDARIES.items():
        if low <= citations <= high:
            return tier
    return None


def main():
    # Load author pool
    data = json.load(open(PROJECT_ROOT / "data" / "probes" / "proceedings_authors_v2.json"))
    papers = data["papers"]

    # Extract unique authors with their venue-based field
    author_venues = defaultdict(set)
    for p in papers:
        venue = p.get("venue", "").split()[0]  # "NSDI 2023" -> "NSDI"
        field = VENUE_TO_FIELD.get(venue, p.get("field", "unknown"))
        for a in p.get("authors", []):
            if is_valid_author_name(a):
                author_venues[a].add(field)

    logger.info(f"Valid authors in pool: {len(author_venues)}")

    # Load existing researcher probes to avoid duplicates
    existing_names = set()
    try:
        existing = json.load(open(PROJECT_ROOT / "data" / "probes" / "researcher_field_probes_v3.json"))
        for p in existing:
            name = p.get("researcher_name", "")
            if name:
                existing_names.add(name.lower())
    except:
        pass
    logger.info(f"Existing researcher probes: {len(existing_names)} names")

    # Sample authors stratified by venue field
    candidates = []
    for name, fields in author_venues.items():
        if name.lower() in existing_names:
            continue
        # Pick the most specific field
        field = sorted(fields, key=lambda f: len(f), reverse=True)[0]
        candidates.append((name, field))

    random.shuffle(candidates)
    logger.info(f"Candidates after dedup: {len(candidates)}")

    # Sample up to 800 (we'll query OpenAlex for all, then filter by tier)
    sample = candidates[:800]
    logger.info(f"Sampling {len(sample)} for OpenAlex queries")

    # Query OpenAlex in parallel (3 workers to stay under 10 req/sec rate limit)
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(query_openalex_author, name): (name, field) for name, field in sample}
        done = 0
        for future in as_completed(futures):
            name, venue_field = futures[future]
            try:
                info = future.result()
                if info and info["citations"] > 0:
                    # Use OpenAlex field if available, fallback to venue field
                    field = info["field"] or venue_field
                    tier = assign_tier(info["citations"])
                    if tier:
                        results.append({
                            "name": info["name"],
                            "citations": info["citations"],
                            "works": info["works"],
                            "field": field,
                            "venue_field": venue_field,
                            "concepts": info["concepts"],
                            "tier": tier,
                        })
                done += 1
                if done % 50 == 0:
                    tier_counts = Counter(r["tier"] for r in results)
                    logger.info(f"  {done}/{len(sample)} queried, tiers so far: {dict(tier_counts)}")
            except Exception as e:
                logger.warning(f"Error: {e}")
                done += 1

    logger.info(f"\nTotal authors with citation data: {len(results)}")
    tier_counts = Counter(r["tier"] for r in results)
    field_counts = Counter(r["field"] for r in results)
    logger.info(f"Tier distribution: {dict(sorted(tier_counts.items()))}")
    logger.info(f"Field distribution: {dict(sorted(field_counts.items()))}")

    # Build probes, ensuring field diversity per tier
    probes = []
    for tier in ["T3", "T4", "T5", "T6", "T7"]:
        tier_authors = [r for r in results if r["tier"] == tier]
        # Sort by field to get diversity, then sample
        random.shuffle(tier_authors)

        # Take up to 50 per tier, ensuring field diversity
        used_fields = Counter()
        selected = []
        for a in tier_authors:
            if len(selected) >= 60:
                break
            # Limit any single field to 30% of tier
            if used_fields[a["field"]] >= max(3, len(selected) * 0.3 + 1):
                continue
            selected.append(a)
            used_fields[a["field"]] += 1

        for i, a in enumerate(selected):
            probe = {
                "id": f"{tier}_researcher_new_{i:03d}",
                "question_direct": f"What is the research field of {a['name']}?",
                "question_fill_blank": f"{a['name']} is a researcher in the field of ___.",
                "question_contextual": f"In computer science, {a['name']}'s research focuses on ___.",
                "answer": a["field"],
                "answer_type": "field",
                "domain": "computer_science",
                "region": "global",
                "tier": tier,
                "frequency_metric": f"OpenAlex citations: {a['citations']}",
                "source": "DBLP+OpenAlex",
                "researcher_name": a["name"],
                "citation_count": a["citations"],
                "paper_count": a["works"],
                "top_concepts": a["concepts"][:3],
                "scoring_note": "Use LLM-as-judge for field matching",
            }
            probes.append(probe)

        logger.info(f"  {tier}: {len(selected)} probes, fields: {dict(used_fields)}")

    # Save
    output = PROJECT_ROOT / "data" / "probes" / "researcher_field_probes_v4.json"
    with open(output, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved {len(probes)} researcher probes to {output}")

    # Also save raw OpenAlex results for analysis
    raw_output = PROJECT_ROOT / "data" / "probes" / "openalex_author_results.json"
    with open(raw_output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved raw results to {raw_output}")


if __name__ == "__main__":
    main()
