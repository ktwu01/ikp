#!/usr/bin/env python3
"""Generate mid-tier researcher probes by querying OpenAlex for researchers
with specific citation ranges.

Problem: Our current researcher pool is dominated by low-citation authors
who all end up in T7. We need researchers in the 1K-100K citation range
to fill T3-T6.

Strategy: Query OpenAlex directly for CS researchers by citation count range,
not from our existing DBLP pool. This gives us access to the full OpenAlex
database of ~200M authors.

OpenAlex API supports filtering by cited_by_count range and concept.
"""

import json
import os
import time
import random
import logging
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OPENALEX_API_KEY = "B9py5POrsA9X9ldIj8yZn8"

# CS concept ID in OpenAlex
CS_CONCEPT = "C41008148"  # Computer Science

# Citation ranges targeting T3-T6
CITATION_RANGES = [
    (50000, 200000, "very_high"),    # T3 candidates - very famous
    (20000, 50000, "high"),          # T3-T4 candidates
    (5000, 20000, "medium_high"),    # T4-T5 candidates
    (2000, 5000, "medium"),          # T5 candidates
    (500, 2000, "medium_low"),       # T5-T6 candidates
    (100, 500, "low"),               # T6 candidates
]

# Map OpenAlex topics to our standardized fields
TOPIC_TO_FIELD = {
    "artificial intelligence": "machine learning",
    "machine learning": "machine learning",
    "natural language processing": "natural language processing",
    "computer vision": "computer vision",
    "computer security": "computer security",
    "computer network": "computer networking",
    "database": "database systems",
    "data mining": "data mining",
    "information retrieval": "information retrieval",
    "human-computer interaction": "human-computer interaction",
    "algorithm": "theoretical computer science",
    "computational complexity": "theoretical computer science",
    "programming language": "programming languages",
    "compiler": "programming languages",
    "operating system": "operating systems",
    "distributed computing": "distributed systems",
    "cloud computing": "distributed systems",
    "computer architecture": "computer architecture",
    "embedded system": "embedded systems",
    "computer graphics": "computer graphics",
    "software engineering": "software engineering",
    "robotics": "robotics",
}


def query_openalex_by_citations(min_cite: int, max_cite: int, per_page: int = 200, page: int = 1) -> list[dict]:
    """Query OpenAlex for authors in a citation range (filter CS client-side)."""
    with httpx.Client(timeout=60) as http:
        r = http.get("https://api.openalex.org/authors",
            params={
                "filter": f"cited_by_count:{min_cite}-{max_cite}",
                "per_page": per_page,
                "sample": per_page,
                "seed": random.randint(1, 99999),
            },
            headers={"Authorization": f"Bearer {OPENALEX_API_KEY}"})
        if r.status_code == 200:
            return r.json().get("results", [])
        else:
            logger.warning(f"OpenAlex {r.status_code}: {r.text[:100]}")
            return []


def extract_field(author: dict) -> str | None:
    """Extract standardized CS field from OpenAlex author concepts."""
    concepts = author.get("x_concepts", []) or []
    for c in concepts:
        name = c.get("display_name", "").lower()
        if name in TOPIC_TO_FIELD:
            return TOPIC_TO_FIELD[name]
    return None


def generate_all(target_per_range: int = 200) -> list[dict]:
    """Generate researcher probes across citation ranges."""

    # Load existing names to deduplicate
    existing_names = set()
    try:
        store = json.loads((PROJECT_ROOT / "data" / "pipeline_store.json").read_text())
        for p in store:
            if p.get("source_type") == "researcher":
                name = p.get("researcher_name", "").lower()
                if name:
                    existing_names.add(name)
    except:
        pass
    logger.info(f"Existing researcher names: {len(existing_names)}")

    all_probes = []

    for min_cite, max_cite, label in CITATION_RANGES:
        logger.info(f"\nQuerying citations {min_cite:,}-{max_cite:,} ({label})...")

        # Query multiple samples to get diverse results
        # Most won't be CS, so we need many samples
        authors = []
        for i in range(10):  # 10 random samples × 200 = 2000 candidates per range
            results = query_openalex_by_citations(min_cite, max_cite, per_page=200)
            authors.extend(results)
            time.sleep(0.2)

        logger.info(f"  Got {len(authors)} authors from OpenAlex")

        probes = []
        for a in authors:
            name = a.get("display_name", "")
            if not name or len(name) < 4:
                continue
            if name.lower() in existing_names:
                continue

            # Must have at least 2 name parts (not institution)
            parts = name.strip().split()
            if len(parts) < 2:
                continue

            # Skip institution-like names
            lower = name.lower()
            if any(bad in lower for bad in ["university", "institute", "college", "laboratory",
                                             "inc.", "ltd", "center", "centre", "department"]):
                continue

            citations = a.get("cited_by_count", 0)
            field = extract_field(a)
            if not field:
                continue

            existing_names.add(name.lower())
            probes.append({
                "question": f"What is the research field of {name}?",
                "answer": field,
                "source_type": "researcher",
                "researcher_name": name,
                "citation_count": citations,
                "paper_count": a.get("works_count", 0),
                "domain": "computer_science",
                "citation_range": label,
            })

        logger.info(f"  Valid probes: {len(probes)}")
        fields = Counter(p["answer"] for p in probes)
        logger.info(f"  Fields: {dict(fields.most_common(5))}")
        all_probes.extend(probes)

    logger.info(f"\nTotal: {len(all_probes)} new researcher probes")
    fields = Counter(p["answer"] for p in all_probes)
    logger.info(f"Fields ({len(fields)} unique):")
    for f, n in fields.most_common():
        logger.info(f"  {f}: {n}")

    ranges = Counter(p["citation_range"] for p in all_probes)
    logger.info(f"\nBy citation range:")
    for r, n in sorted(ranges.items()):
        logger.info(f"  {r}: {n}")

    return all_probes


if __name__ == "__main__":
    probes = generate_all(target_per_range=200)

    # Save
    output = PROJECT_ROOT / "data" / "probes" / "researcher_probes_mid_tier.json"
    output.write_text(json.dumps(probes, indent=2, ensure_ascii=False))
    logger.info(f"\nSaved to {output}")

    # Ingest into store
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.store import ProbeStore
    store = ProbeStore(PROJECT_ROOT / "data" / "pipeline_store.json")
    added = store.add_batch(probes)
    store.save()
    logger.info(f"Added {added} new probes to pipeline store")
