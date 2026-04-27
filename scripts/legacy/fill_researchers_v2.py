#!/usr/bin/env python3
"""Fill researcher probe gaps using OpenAlex. Sequential queries with rate limiting.

Strategy: Sample 400 diverse authors from the 24K pool, query OpenAlex ONE AT A TIME
with a 0.15s delay (well under 10 req/sec limit). Assign tiers based on citation count.
"""

import json
import os
import time
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

VENUE_TO_FIELD = {
    "NSDI": "computer networking", "SIGCOMM": "computer networking",
    "ISCA": "computer architecture", "MICRO": "computer architecture",
    "ASPLOS": "computer architecture",
    "NeurIPS": "machine learning", "ICML": "machine learning",
    "CCS": "computer security", "USENIX": "computer security",
    "VLDB": "database systems", "SIGMOD": "database systems",
    "KDD": "data mining", "CHI": "human-computer interaction",
    "STOC": "theoretical computer science",
    "PLDI": "programming languages", "POPL": "programming languages",
    "OSDI": "operating systems", "SOSP": "operating systems",
    "ATC": "operating systems", "EuroSys": "distributed systems",
    "FAST": "storage systems",
}

# Map OpenAlex topics to our standardized fields
TOPIC_TO_FIELD = {
    "artificial intelligence": "machine learning",
    "machine learning": "machine learning",
    "deep learning": "machine learning",
    "natural language processing": "natural language processing",
    "computer vision": "computer vision",
    "computer security": "computer security",
    "information security": "computer security",
    "cryptography": "computer security",
    "computer network": "computer networking",
    "networking": "computer networking",
    "database": "database systems",
    "data mining": "data mining",
    "information retrieval": "data mining",
    "human-computer interaction": "human-computer interaction",
    "algorithm": "theoretical computer science",
    "computational complexity": "theoretical computer science",
    "programming language": "programming languages",
    "compiler": "programming languages",
    "operating system": "operating systems",
    "distributed computing": "distributed systems",
    "distributed system": "distributed systems",
    "cloud computing": "distributed systems",
    "computer architecture": "computer architecture",
    "computer graphics": "computer graphics",
    "software engineering": "software engineering",
    "robotics": "robotics",
    "embedded system": "embedded systems",
}

BAD_PATTERNS = [
    "university", "institute", "laboratory", "inc.", "corp", "ltd",
    "gmbh", "research", "department", "school", "college", "center",
    "centre", "group", "team", "microsoft", "google", "facebook",
    "meta", "amazon", "apple", "nvidia", "bytedance", "tencent",
    "alibaba", "huawei", "samsung", "purple mountain", "adobe",
]


def is_valid_name(name: str) -> bool:
    if not name or len(name) < 3:
        return False
    lower = name.lower()
    for bad in BAD_PATTERNS:
        if bad in lower:
            return False
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    return True


def query_openalex(name: str) -> dict:
    """Query OpenAlex for a single author. Returns dict or None."""
    try:
        with httpx.Client(timeout=30) as http:
            r = http.get("https://api.openalex.org/authors",
                params={"search": name, "per_page": 1, "mailto": "research@example.org"})
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    a = results[0]
                    concepts = a.get("x_concepts", []) or a.get("topics", [])
                    field = None
                    for c in concepts:
                        cn = c.get("display_name", "").lower()
                        if cn in TOPIC_TO_FIELD:
                            field = TOPIC_TO_FIELD[cn]
                            break
                    return {
                        "name": a.get("display_name", name),
                        "citations": a.get("cited_by_count", 0),
                        "works": a.get("works_count", 0),
                        "field": field,
                        "concepts": [c.get("display_name", "") for c in concepts[:5]],
                    }
            elif r.status_code == 429:
                time.sleep(5)
                # One retry
                r2 = http.get("https://api.openalex.org/authors",
                    params={"search": name, "per_page": 1, "mailto": "research@example.org"})
                if r2.status_code == 200:
                    results = r2.json().get("results", [])
                    if results:
                        a = results[0]
                        concepts = a.get("x_concepts", []) or a.get("topics", [])
                        field = None
                        for c in concepts:
                            cn = c.get("display_name", "").lower()
                            if cn in TOPIC_TO_FIELD:
                                field = TOPIC_TO_FIELD[cn]
                                break
                        return {
                            "name": a.get("display_name", name),
                            "citations": a.get("cited_by_count", 0),
                            "works": a.get("works_count", 0),
                            "field": field,
                            "concepts": [c.get("display_name", "") for c in concepts[:5]],
                        }
    except Exception as e:
        logger.warning(f"Error for {name}: {e}")
    return None


def assign_tier(citations: int) -> str:
    if citations >= 5000: return "T3"
    if citations >= 1000: return "T4"
    if citations >= 200: return "T5"
    if citations >= 50: return "T6"
    if citations >= 1: return "T7"
    return None


def main():
    data = json.load(open(PROJECT_ROOT / "data" / "probes" / "proceedings_authors_v2.json"))
    papers = data["papers"]

    # Build author -> venue mapping
    author_venues = defaultdict(set)
    for p in papers:
        venue = p.get("venue", "").split()[0]
        field = VENUE_TO_FIELD.get(venue, p.get("field", "unknown"))
        for a in p.get("authors", []):
            if is_valid_name(a):
                author_venues[a].add(field)

    logger.info(f"Valid authors: {len(author_venues)}")

    # Load existing to avoid duplicates
    existing_names = set()
    try:
        for f in ["researcher_field_probes_v3.json", "researcher_field_probes_v4.json"]:
            try:
                existing = json.load(open(PROJECT_ROOT / "data" / "probes" / f))
                for p in existing:
                    name = p.get("researcher_name", "")
                    if name:
                        existing_names.add(name.lower())
            except:
                pass
    except:
        pass
    logger.info(f"Existing names to skip: {len(existing_names)}")

    # Sample 400 authors, stratified by venue to ensure field diversity
    candidates = [(name, sorted(fields, key=len, reverse=True)[0])
                  for name, fields in author_venues.items()
                  if name.lower() not in existing_names]
    random.shuffle(candidates)

    # Stratified: take proportional samples from each field
    by_field = defaultdict(list)
    for name, field in candidates:
        by_field[field].append(name)

    sample = []
    per_field = max(30, 400 // len(by_field))
    for field, names in by_field.items():
        for name in names[:per_field]:
            sample.append((name, field))
    random.shuffle(sample)
    sample = sample[:400]
    logger.info(f"Sampling {len(sample)} authors")

    # Query sequentially
    results = []
    for i, (name, venue_field) in enumerate(sample):
        info = query_openalex(name)
        if info and info["citations"] > 0:
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

        if (i + 1) % 50 == 0:
            tier_counts = Counter(r["tier"] for r in results)
            logger.info(f"  {i+1}/{len(sample)} queried, found {len(results)}, tiers: {dict(tier_counts)}")

        time.sleep(0.15)  # Rate limit: ~6-7 req/sec

    logger.info(f"\nTotal authors with data: {len(results)}")
    tier_counts = Counter(r["tier"] for r in results)
    field_counts = Counter(r["field"] for r in results)
    logger.info(f"Tiers: {dict(sorted(tier_counts.items()))}")
    logger.info(f"Fields: {dict(sorted(field_counts.items()))}")

    # Build probes with field diversity
    probes = []
    for tier in ["T3", "T4", "T5", "T6", "T7"]:
        tier_authors = [r for r in results if r["tier"] == tier]
        random.shuffle(tier_authors)

        used_fields = Counter()
        selected = []
        for a in tier_authors:
            if len(selected) >= 60:
                break
            if used_fields[a["field"]] >= max(5, len(selected) * 0.3 + 1):
                continue
            selected.append(a)
            used_fields[a["field"]] += 1

        for i, a in enumerate(selected):
            probe = {
                "id": f"{tier}_researcher_v4_{i:03d}",
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

    output = PROJECT_ROOT / "data" / "probes" / "researcher_field_probes_v4.json"
    with open(output, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved {len(probes)} researcher probes to {output}")


if __name__ == "__main__":
    main()
