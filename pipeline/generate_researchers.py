#!/usr/bin/env python3
"""Generate ~1000 diverse researcher probes using OpenAlex.

Strategy:
1. Sample authors from the 24K pool, stratified by venue (field diversity)
2. Query OpenAlex sequentially (1 req/sec) for citation count + research topics
3. Map OpenAlex topics to standardized CS field names
4. Filter out institution names, duplicates, and ambiguous fields
5. Create probes: "What is the research field of X?"

Target: ~1000 probes with 20+ diverse fields, spread across citation ranges
(tiers will be assigned by the landmark calibration pipeline, not by us).
"""

import json
import time
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Venue -> standardized field
VENUE_TO_FIELD = {
    "NSDI": "computer networking", "SIGCOMM": "computer networking",
    "ISCA": "computer architecture", "MICRO": "computer architecture",
    "ASPLOS": "computer systems",
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

# OpenAlex concept -> our standardized field (broader mapping)
CONCEPT_TO_FIELD = {
    "artificial intelligence": "machine learning",
    "machine learning": "machine learning",
    "deep learning": "machine learning",
    "neural network": "machine learning",
    "natural language processing": "natural language processing",
    "computer vision": "computer vision",
    "image processing": "computer vision",
    "pattern recognition": "computer vision",
    "computer security": "computer security",
    "cryptography": "computer security",
    "information security": "computer security",
    "computer network": "computer networking",
    "wireless network": "computer networking",
    "internet": "computer networking",
    "database": "database systems",
    "data mining": "data mining",
    "information retrieval": "information retrieval",
    "web mining": "data mining",
    "human-computer interaction": "human-computer interaction",
    "user interface": "human-computer interaction",
    "algorithm": "theoretical computer science",
    "computational complexity": "theoretical computer science",
    "graph theory": "theoretical computer science",
    "combinatorics": "theoretical computer science",
    "programming language": "programming languages",
    "compiler": "programming languages",
    "software verification": "programming languages",
    "formal methods": "formal methods",
    "operating system": "operating systems",
    "distributed computing": "distributed systems",
    "cloud computing": "distributed systems",
    "parallel computing": "distributed systems",
    "computer architecture": "computer architecture",
    "microprocessor": "computer architecture",
    "embedded system": "embedded systems",
    "computer graphics": "computer graphics",
    "visualization": "computer graphics",
    "rendering": "computer graphics",
    "software engineering": "software engineering",
    "software testing": "software engineering",
    "robotics": "robotics",
    "control theory": "robotics",
    "bioinformatics": "computational biology",
    "computational biology": "computational biology",
    "simulation": "simulation",
    "real-time computing": "real-time systems",
}

BAD_PATTERNS = [
    "university", "institute", "laboratory", "inc.", "corp", "ltd",
    "gmbh", "department", "school", "college", "center", "centre",
    "group", "team", "microsoft", "google", "facebook", "meta",
    "amazon", "apple", "nvidia", "bytedance", "tencent", "alibaba",
    "huawei", "samsung", "purple mountain", "adobe", "research",
]


def is_valid_name(name: str) -> bool:
    if not name or len(name) < 4:
        return False
    lower = name.lower()
    for bad in BAD_PATTERNS:
        if bad in lower:
            return False
    parts = name.strip().split()
    return len(parts) >= 2


def query_openalex(name: str) -> dict | None:
    """Query OpenAlex for one author. Returns dict or None."""
    with httpx.Client(timeout=30) as http:
        try:
            r = http.get("https://api.openalex.org/authors",
                params={"search": name, "per_page": 1},
                headers={"Authorization": "Bearer B9py5POrsA9X9ldIj8yZn8"})
            if r.status_code == 200:
                results = r.json().get("results", [])
                if not results:
                    return None
                a = results[0]
                concepts = a.get("x_concepts", []) or a.get("topics", [])

                # Find best matching field
                field = None
                for c in concepts:
                    cn = c.get("display_name", "").lower()
                    if cn in CONCEPT_TO_FIELD:
                        field = CONCEPT_TO_FIELD[cn]
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
                return None
            return None
        except:
            return None


def generate_all(target: int = 1000) -> list[dict]:
    # Load author pool
    data = json.load(open(PROJECT_ROOT / "data" / "probes" / "proceedings_authors_v2.json"))
    papers = data["papers"]

    # Build author -> venues mapping
    author_venues = defaultdict(set)
    for p in papers:
        venue = p.get("venue", "").split()[0]
        for a in p.get("authors", []):
            if is_valid_name(a):
                author_venues[a].add(venue)

    logger.info(f"Valid authors in pool: {len(author_venues)}")

    # Load existing researcher names to avoid duplicates
    existing_names = set()
    try:
        store = json.loads((PROJECT_ROOT / "data" / "pipeline_store.json").read_text())
        for p in store:
            if p.get("source_type") == "researcher":
                rn = p.get("researcher_name", "").lower()
                if rn:
                    existing_names.add(rn)
    except:
        pass
    logger.info(f"Existing researcher names: {len(existing_names)}")

    # Build candidate list, stratified by venue for field diversity
    by_venue = defaultdict(list)
    for name, venues in author_venues.items():
        if name.lower() in existing_names:
            continue
        for v in venues:
            by_venue[v].append(name)

    # Sample proportionally from each venue, up to ~2x target for safety
    sample_target = target * 2
    per_venue = max(50, sample_target // len(by_venue))
    candidates = []
    for venue, names in by_venue.items():
        random.shuffle(names)
        for name in names[:per_venue]:
            field_guess = VENUE_TO_FIELD.get(venue, "computer science")
            candidates.append((name, field_guess))

    random.shuffle(candidates)
    candidates = candidates[:sample_target]
    logger.info(f"Candidates to query: {len(candidates)}")

    # Resume from checkpoint if available
    output = PROJECT_ROOT / "data" / "probes" / "researcher_probes_v5.json"
    results = []
    done_names = set()
    if output.exists():
        results = json.loads(output.read_text())
        done_names = {r["researcher_name"].lower() for r in results}
        logger.info(f"Resuming from checkpoint: {len(results)} probes, {len(done_names)} names done")

    # Filter out already-done candidates
    candidates = [(n, f) for n, f in candidates if n.lower() not in done_names]
    logger.info(f"Remaining candidates: {len(candidates)}")

    # Query OpenAlex in parallel with 16 workers (API key removes rate limit)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def query_one(args):
        name, venue_field = args
        info = query_openalex(name)
        if not info or info["citations"] <= 0:
            return None
        field = info["field"] or venue_field
        return {
            "question": f"What is the research field of {info['name']}?",
            "answer": field,
            "source_type": "researcher",
            "researcher_name": info["name"],
            "citation_count": info["citations"],
            "paper_count": info["works"],
            "top_concepts": info["concepts"][:3],
            "domain": "computer_science",
        }

    done_count = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(query_one, c): c for c in candidates}
        for future in as_completed(futures):
            if len(results) >= target:
                break
            try:
                result = future.result()
                if result:
                    results.append(result)
                done_count += 1
                if done_count % 100 == 0:
                    fields = Counter(r["answer"] for r in results)
                    logger.info(f"  {done_count} queried, {len(results)} valid, fields: {len(fields)} unique")
                    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                    logger.info(f"  Checkpoint saved: {len(results)} probes")
            except:
                done_count += 1

    # Final save
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    logger.info(f"\nGenerated {len(results)} researcher probes")

    # Field distribution
    fields = Counter(r["answer"] for r in results)
    logger.info(f"Fields ({len(fields)} unique):")
    for f, n in fields.most_common():
        logger.info(f"  {f}: {n}")

    # Citation distribution
    cites = sorted(r["citation_count"] for r in results)
    logger.info(f"\nCitation range: {cites[0]} - {cites[-1]}")
    logger.info(f"Median: {cites[len(cites)//2]}")

    return results


if __name__ == "__main__":
    probes = generate_all(target=1000)
    output = PROJECT_ROOT / "data" / "probes" / "researcher_probes_v5.json"
    output.write_text(json.dumps(probes, indent=2, ensure_ascii=False))
    logger.info(f"Saved to {output}")
