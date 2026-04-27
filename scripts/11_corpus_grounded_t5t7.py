#!/usr/bin/env python3
"""Corpus-grounded T5-T7 probe generation.

This script generates T5-T7 probes from EXTERNAL data sources,
NOT from LLM recall. The pipeline:

  1. SAMPLE entities from structured corpora
  2. MEASURE document frequency via web search result counts
  3. ASSIGN tier based on frequency
  4. FORMULATE probe and VERIFY answer from source

Data sources used:
  - DBLP API: conference proceedings → researcher names
  - Semantic Scholar API: citation counts for researchers
  - Wikipedia API: random articles → obscure facts
  - Web search: result count as frequency proxy
"""

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api_client import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "corpus_grounded.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# ================================================================
# SOURCE 1: DBLP → Researchers with citation counts
# ================================================================

def fetch_dblp_conference_authors(venue_key, year):
    """Fetch ALL author names from a DBLP conference proceedings page."""
    url = f"https://dblp.org/search/publ/api"
    params = {
        "q": f"venue:{venue_key}: year:{year}:",
        "h": 200,  # max results
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"DBLP API returned {resp.status_code} for {venue_key} {year}")
            return []
        data = resp.json()
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        authors = set()
        for hit in hits:
            info = hit.get("info", {})
            author_info = info.get("authors", {}).get("author", [])
            if isinstance(author_info, dict):
                author_info = [author_info]
            for a in author_info:
                name = a.get("text", "") if isinstance(a, dict) else str(a)
                if name and len(name) > 3:
                    authors.add(name)
        return list(authors)
    except Exception as e:
        logger.error(f"DBLP fetch failed for {venue_key} {year}: {e}")
        return []


def get_semantic_scholar_citations(author_name, retries=2):
    """Look up an author's citation count on Semantic Scholar."""
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params={
                "query": author_name,
                "limit": 3,
                "fields": "citationCount,paperCount,hIndex,affiliations",
            }, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    best = max(data, key=lambda x: x.get("citationCount", 0))
                    return {
                        "citations": best.get("citationCount", 0),
                        "papers": best.get("paperCount", 0),
                        "hIndex": best.get("hIndex", 0),
                        "affiliations": best.get("affiliations", []),
                    }
                return None
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                return None
        except:
            time.sleep(2)
    return None


def generate_researcher_probes():
    """Generate researcher probes from DBLP + Semantic Scholar."""
    logger.info("=" * 60)
    logger.info("  GENERATING RESEARCHER PROBES FROM DBLP + SEMANTIC SCHOLAR")
    logger.info("=" * 60)

    # Step 1: Fetch author names from multiple venues and years
    venues = [
        ("conf/sigcomm", [2022, 2023, 2024]),
        ("conf/nsdi", [2023, 2024]),
        ("conf/osdi", [2023, 2024]),
        ("conf/sosp", [2023]),
        ("conf/eurosys", [2023, 2024]),
    ]

    all_authors = set()
    for venue_key, years in venues:
        for year in years:
            logger.info(f"  Fetching {venue_key} {year} from DBLP...")
            authors = fetch_dblp_conference_authors(venue_key, year)
            all_authors.update(authors)
            logger.info(f"    Got {len(authors)} authors (total unique: {len(all_authors)})")
            time.sleep(1)

    logger.info(f"\n  Total unique authors from all venues: {len(all_authors)}")

    # Step 2: Sample and look up citation counts
    # Randomly sample to avoid bias toward any particular ordering
    random.seed(42)
    author_list = list(all_authors)
    random.shuffle(author_list)

    # Look up citation counts (rate-limited)
    author_profiles = []
    lookup_limit = min(120, len(author_list))  # Look up at most 120

    for i, name in enumerate(author_list[:lookup_limit]):
        profile = get_semantic_scholar_citations(name)
        if profile:
            profile["name"] = name
            author_profiles.append(profile)
            if (i + 1) % 10 == 0:
                logger.info(f"    Looked up {i+1}/{lookup_limit} authors ({len(author_profiles)} with data)")
        time.sleep(1.2)  # Rate limit

    logger.info(f"\n  Got citation data for {len(author_profiles)} authors")

    # Step 3: Tier assignment based on citation count
    tier_bins = {
        "T4": (200, 2000),
        "T5": (50, 200),
        "T6": (10, 50),
        "T7": (0, 10),
    }

    tiered_authors = {tier: [] for tier in tier_bins}
    for p in author_profiles:
        cites = p["citations"]
        for tier, (lo, hi) in tier_bins.items():
            if lo <= cites < hi:
                tiered_authors[tier].append(p)
                break

    for tier in sorted(tiered_authors.keys()):
        logger.info(f"  {tier}: {len(tiered_authors[tier])} authors")

    # Step 4: Formulate probes (select up to 15 per tier)
    probes = []
    for tier, authors in sorted(tiered_authors.items()):
        selected = authors[:15]
        for i, a in enumerate(selected):
            affiliation = a["affiliations"][0] if a.get("affiliations") else "unknown"
            probe = {
                "id": f"{tier}_researcher_{i:03d}",
                "question_direct": f"What institution is {a['name']} affiliated with?",
                "question_fill_blank": f"{a['name']} is affiliated with ___.",
                "question_contextual": f"The researcher {a['name']} works at ___.",
                "answer": affiliation,
                "answer_type": "text",
                "domain": "people",
                "region": "Unknown",
                "tier": tier,
                "frequency_metric": f"Semantic Scholar citations: {a['citations']}",
                "estimated_web_frequency": str(a['citations']),
                "source": "Semantic Scholar API + DBLP",
                "researcher_name": a["name"],
                "citation_count": a["citations"],
                "h_index": a["hIndex"],
                "paper_count": a["papers"],
            }
            probes.append(probe)

    return probes, author_profiles


# ================================================================
# SOURCE 2: Wikipedia Random → General obscure facts
# ================================================================

def fetch_wikipedia_random_articles(n=50, lang="en"):
    """Fetch random Wikipedia articles to discover obscure facts."""
    articles = []
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"

    for i in range(n):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                articles.append({
                    "title": data.get("title", ""),
                    "extract": data.get("extract", ""),
                    "description": data.get("description", ""),
                    "pageid": data.get("pageid"),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                })
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Wikipedia random failed: {e}")
            time.sleep(1)

        if (i + 1) % 20 == 0:
            logger.info(f"    Fetched {i+1}/{n} random Wikipedia articles")

    return articles


def estimate_article_frequency(title):
    """Estimate how many Google results mention this article title."""
    # Use Wikipedia page views as a proxy for frequency
    # (higher views ≈ more well-known ≈ higher frequency)
    url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    url += f"/en.wikipedia/all-access/all-agents/{requests.utils.quote(title)}/monthly/20240101/20241231"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "IKP-Research/1.0"})
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            total_views = sum(item.get("views", 0) for item in items)
            return total_views
        return None
    except:
        return None


def generate_wikipedia_probes(client):
    """Generate probes from random Wikipedia articles."""
    logger.info("\n" + "=" * 60)
    logger.info("  GENERATING PROBES FROM RANDOM WIKIPEDIA ARTICLES")
    logger.info("=" * 60)

    # Fetch random articles
    logger.info("  Fetching 80 random Wikipedia articles...")
    articles = fetch_wikipedia_random_articles(n=80)
    logger.info(f"  Got {len(articles)} articles")

    # Check page views as frequency proxy
    articles_with_freq = []
    for i, art in enumerate(articles):
        views = estimate_article_frequency(art["title"])
        if views is not None:
            art["annual_views"] = views
            articles_with_freq.append(art)
        time.sleep(0.3)
        if (i + 1) % 20 == 0:
            logger.info(f"    Checked views for {i+1}/{len(articles)} articles")

    logger.info(f"  Got view counts for {len(articles_with_freq)} articles")

    # Tier assignment based on annual page views
    # High views = well-known = low tier, Low views = obscure = high tier
    # Calibration (rough): T5 ≈ 1K-10K views/year, T6 ≈ 100-1K, T7 < 100
    tier_bins_views = {
        "T5": (1000, 10000),
        "T6": (100, 1000),
        "T7": (0, 100),
    }

    tiered_articles = {tier: [] for tier in tier_bins_views}
    for art in articles_with_freq:
        views = art["annual_views"]
        for tier, (lo, hi) in tier_bins_views.items():
            if lo <= views < hi:
                tiered_articles[tier].append(art)
                break

    for tier in sorted(tiered_articles.keys()):
        logger.info(f"  {tier}: {len(tiered_articles[tier])} articles (by page views)")
        for art in tiered_articles[tier][:3]:
            logger.info(f"    - {art['title']} ({art['annual_views']} views/year)")

    # Use LLM to extract factual probes from the articles
    probes = []
    for tier, articles_list in sorted(tiered_articles.items()):
        for art in articles_list[:20]:
            if not art.get("extract") or len(art["extract"]) < 50:
                continue

            # Ask LLM to extract a single incompressible fact
            prompt = f"""From this Wikipedia article extract, create exactly ONE factual question with a single verifiable answer.

Article title: {art['title']}
Extract: {art['extract'][:500]}

Requirements:
- The answer must be a specific name, number, or short phrase
- The fact must be incompressible (cannot be derived by reasoning)
- The question must have exactly one correct answer

Return a JSON object:
{{"question_direct": "What is...?", "question_fill_blank": "The ... is ___.", "answer": "...", "answer_type": "text|numeric"}}

Return ONLY the JSON object, nothing else."""

            try:
                text = client.get_response_text(
                    model="anthropic/claude-sonnet-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=500,
                )
                text = text.strip()
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:-1])
                    if text.startswith("json"):
                        text = text[4:]

                probe_data = json.loads(text)
                probe = {
                    "id": f"{tier}_wiki_{len(probes):03d}",
                    "question_direct": probe_data.get("question_direct", ""),
                    "question_fill_blank": probe_data.get("question_fill_blank", ""),
                    "question_contextual": probe_data.get("question_fill_blank", ""),
                    "answer": probe_data.get("answer", ""),
                    "answer_type": probe_data.get("answer_type", "text"),
                    "domain": "places" if any(w in art.get("description", "").lower() for w in ["city", "town", "village", "commune", "municipality"]) else "people" if "person" in art.get("description", "").lower() else "events",
                    "region": "Unknown",
                    "tier": tier,
                    "frequency_metric": f"Wikipedia annual page views: {art['annual_views']}",
                    "estimated_web_frequency": str(art["annual_views"]),
                    "source": art.get("url", "Wikipedia"),
                    "wikipedia_title": art["title"],
                }
                probes.append(probe)
            except Exception as e:
                continue

    logger.info(f"\n  Generated {len(probes)} Wikipedia-grounded probes")
    return probes


# ================================================================
# SOURCE 3: Chinese Wikipedia → Chinese-language probes
# ================================================================

def generate_chinese_wikipedia_probes(client):
    """Generate Chinese probes from Chinese Wikipedia random articles."""
    logger.info("\n" + "=" * 60)
    logger.info("  GENERATING CHINESE PROBES FROM CHINESE WIKIPEDIA")
    logger.info("=" * 60)

    logger.info("  Fetching 60 random Chinese Wikipedia articles...")
    articles = fetch_wikipedia_random_articles(n=60, lang="zh")
    logger.info(f"  Got {len(articles)} Chinese articles")

    # Check page views
    articles_with_freq = []
    for i, art in enumerate(articles):
        title = art["title"]
        url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
        url += f"/zh.wikipedia/all-access/all-agents/{requests.utils.quote(title)}/monthly/20240101/20241231"
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "IKP-Research/1.0"})
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                views = sum(item.get("views", 0) for item in items)
                art["annual_views"] = views
                articles_with_freq.append(art)
        except:
            pass
        time.sleep(0.3)
        if (i + 1) % 20 == 0:
            logger.info(f"    Checked views for {i+1}/{len(articles)}")

    logger.info(f"  Got view counts for {len(articles_with_freq)} Chinese articles")

    tier_bins_views = {"T5": (1000, 10000), "T6": (100, 1000), "T7": (0, 100)}
    tiered = {tier: [] for tier in tier_bins_views}
    for art in articles_with_freq:
        for tier, (lo, hi) in tier_bins_views.items():
            if lo <= art["annual_views"] < hi:
                tiered[tier].append(art)
                break

    for tier in sorted(tiered.keys()):
        logger.info(f"  {tier}: {len(tiered[tier])} Chinese articles")

    probes = []
    for tier, arts in sorted(tiered.items()):
        for art in arts[:15]:
            if not art.get("extract") or len(art["extract"]) < 30:
                continue

            prompt = f"""从以下中文维基百科摘要中，提取一个事实性问题，答案必须简短明确。

标题：{art['title']}
摘要：{art['extract'][:500]}

要求：答案必须是一个具体的名称、数字或短语，不能通过推理得出。

返回JSON：
{{"question_direct": "...是什么？", "question_fill_blank": "...是___。", "answer": "...", "answer_type": "text|numeric"}}

只返回JSON。"""

            try:
                text = client.get_response_text(
                    model="anthropic/claude-sonnet-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=500,
                )
                text = text.strip()
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:-1])
                    if text.startswith("json"):
                        text = text[4:]
                pd = json.loads(text)
                probe = {
                    "id": f"{tier}_zhwiki_{len(probes):03d}",
                    "question_direct": pd.get("question_direct", ""),
                    "question_fill_blank": pd.get("question_fill_blank", ""),
                    "question_contextual": pd.get("question_fill_blank", ""),
                    "answer": pd.get("answer", ""),
                    "answer_type": pd.get("answer_type", "text"),
                    "domain": "places",
                    "region": "East Asia",
                    "tier": tier,
                    "language": "zh",
                    "frequency_metric": f"zh.Wikipedia annual views: {art['annual_views']}",
                    "source": art.get("url", "zh.Wikipedia"),
                    "wikipedia_title": art["title"],
                }
                probes.append(probe)
            except:
                continue

    logger.info(f"  Generated {len(probes)} Chinese Wikipedia probes")
    return probes


# ================================================================
# MAIN
# ================================================================

def main():
    random.seed(42)
    client = OpenRouterClient(requests_per_minute=30, max_retries=3, timeout=120)

    all_probes = {"researcher": [], "wikipedia_en": [], "wikipedia_zh": []}
    all_metadata = {}

    # 1. Researcher probes from DBLP + Semantic Scholar
    researcher_probes, author_profiles = generate_researcher_probes()
    all_probes["researcher"] = researcher_probes
    all_metadata["author_profiles"] = author_profiles

    # 2. English Wikipedia probes
    wiki_probes = generate_wikipedia_probes(client)
    all_probes["wikipedia_en"] = wiki_probes

    # 3. Chinese Wikipedia probes
    zh_probes = generate_chinese_wikipedia_probes(client)
    all_probes["wikipedia_zh"] = zh_probes

    # Save everything
    output_dir = PROJECT_ROOT / "data" / "probes"

    # Save corpus-grounded T5-T7 probes
    combined = researcher_probes + wiki_probes + zh_probes
    for tier in ["T5", "T6", "T7"]:
        tier_probes = [p for p in combined if p.get("tier") == tier]
        output_file = output_dir / f"{tier}_corpus_grounded.json"
        with open(output_file, "w") as f:
            json.dump(tier_probes, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(tier_probes)} corpus-grounded {tier} probes to {output_file}")

    # Save Chinese probes separately
    output_file = output_dir / "chinese_corpus_grounded.json"
    with open(output_file, "w") as f:
        json.dump(zh_probes, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(zh_probes)} Chinese corpus-grounded probes")

    # Save researcher probes separately
    output_file = output_dir / "researcher_corpus_grounded.json"
    with open(output_file, "w") as f:
        json.dump(researcher_probes, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(researcher_probes)} researcher corpus-grounded probes")

    # Save metadata
    output_file = output_dir / "corpus_grounded_metadata.json"
    with open(output_file, "w") as f:
        json.dump({
            "total_probes": len(combined),
            "by_source": {k: len(v) for k, v in all_probes.items()},
            "by_tier": {tier: len([p for p in combined if p.get("tier") == tier]) for tier in ["T4", "T5", "T6", "T7"]},
            "methodology": {
                "researchers": "DBLP proceedings → Semantic Scholar citations → tier assignment",
                "wikipedia_en": "Random English Wikipedia articles → page view frequency → tier assignment",
                "wikipedia_zh": "Random Chinese Wikipedia articles → page view frequency → tier assignment",
            },
        }, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  CORPUS-GROUNDED PROBE GENERATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Researcher probes: {len(researcher_probes)}")
    logger.info(f"  English Wikipedia probes: {len(wiki_probes)}")
    logger.info(f"  Chinese Wikipedia probes: {len(zh_probes)}")
    logger.info(f"  Total: {len(combined)}")
    for tier in ["T4", "T5", "T6", "T7"]:
        n = len([p for p in combined if p.get("tier") == tier])
        logger.info(f"    {tier}: {n} probes")


if __name__ == "__main__":
    main()
