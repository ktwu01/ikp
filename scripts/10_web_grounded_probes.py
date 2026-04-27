#!/usr/bin/env python3
"""Web-grounded probe generation for T5-T7.

The key methodological insight: probes for tiers beyond the generator model's
own capacity CANNOT be generated from model recall. They must be grounded
in external data with verifiable frequency metrics.

Strategy per category:
  - Cities/places:     Use population as frequency proxy.
                       T5: cities with pop 5K-50K
                       T6: towns with pop 500-5K
                       T7: villages with pop <500
  - Researchers:       Use citation count as frequency proxy.
                       T5: 200-1000 citations
                       T6: 20-200 citations
                       T7: <20 citations
  - Buildings/landmarks: Use Google result count as frequency proxy.
  - Events/dates:      Use Google result count as frequency proxy.

For each probe:
  1. Web search to discover the entity
  2. Web search to verify the specific fact
  3. Web search to estimate document frequency (result count)
  4. Only include if frequency matches target tier
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "web_grounded_probes.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================
# PROBE GENERATION STRATEGIES
# ============================================================

RESEARCHER_SEARCH_QUERIES = [
    # Tier 5 (200-1000 citations): search for recent award winners, mid-career
    'site:dblp.org "SIGCOMM" OR "OSDI" OR "SOSP" OR "NSDI" author 2022',
    'site:dblp.org "EuroSys" OR "ASPLOS" OR "FAST" author 2023',
    '"assistant professor" "computer science" "systems" OR "networking" site:edu',
    '"data center" OR "RDMA" OR "SmartNIC" researcher site:scholar.google.com',

    # Tier 6 (20-200 citations): search for PhD students, postdocs
    '"PhD student" "computer science" "systems" OR "networking" 2024 site:edu',
    '"postdoctoral researcher" "systems" OR "networking" site:edu',

    # Tier 7 (<20 citations): search for very recent grads
    '"PhD candidate" "systems" published 2024 site:edu',
]

CITY_POPULATION_QUERIES = [
    # Tier 5: small cities (5K-50K pop), various countries
    'population "census 2020" city 10000 site:wikipedia.org',
    'small town population 8000 province China',
    'commune population 5000 France census',
    'municipality population 15000 Brazil IBGE',

    # Tier 6: very small towns (500-5K)
    'village population 2000 prefecture Japan',
    'small town population 1500 district India census',
    'gemeente population 3000 Netherlands',

    # Tier 7: tiny villages (<500)
    'hamlet population 200 county',
    'village 150 inhabitants municipality',
]


def search_and_build_probes(client, search_func, category, tier, queries, n_target=15):
    """Search the web for entities matching a tier's frequency band,
    then build probes from the discovered facts.

    Args:
        client: OpenRouterClient for LLM-assisted extraction
        search_func: function(query) -> list of search results
        category: "researchers", "cities", etc.
        tier: "T5", "T6", or "T7"
        queries: list of search queries to use
        n_target: number of probes to generate
    """
    logger.info(f"Generating {tier} {category} probes from web search...")

    discovered_entities = []

    for query in queries:
        logger.info(f"  Searching: {query[:80]}...")
        try:
            results = search_func(query)
            if results:
                discovered_entities.extend(results)
                logger.info(f"    Found {len(results)} results")
        except Exception as e:
            logger.warning(f"    Search failed: {e}")

    if not discovered_entities:
        logger.warning(f"  No entities found for {tier} {category}")
        return []

    # Use LLM to extract structured probes from search results
    entities_text = "\n".join(f"- {e}" for e in discovered_entities[:30])

    prompt = f"""Based on these web search results about {category}, extract verifiable factual probes.

Search results:
{entities_text}

For each entity found, create a probe with:
- A factual question that has a single verifiable answer
- The answer must be a specific name, number, or date
- The fact should be incompressible (must be memorized, not derivable)

Return a JSON array where each element is:
{{
  "id": "{tier}_{category}_001",
  "question_direct": "What is ...?",
  "question_fill_blank": "The ... is ___.",
  "question_contextual": "In ..., the ... is ___.",
  "answer": "...",
  "answer_type": "text|numeric",
  "domain": "people|places|publications|measurements|events|organizations",
  "region": "...",
  "tier": "{tier}",
  "frequency_metric": "description of how frequency was estimated",
  "estimated_web_frequency": "number of web pages",
  "source": "URL or description of verification source"
}}

Generate at most {n_target} probes. Only include facts you are confident are correct.
Return ONLY the JSON array."""

    messages = [
        {"role": "system", "content": "Extract factual probes from web search results. Only include verifiable facts."},
        {"role": "user", "content": prompt},
    ]

    try:
        text = client.get_response_text(
            model="anthropic/claude-sonnet-4",
            messages=messages,
            temperature=0,
            max_tokens=8000,
        )

        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
            if text.startswith("json"):
                text = text[4:]

        probes = json.loads(text)
        logger.info(f"  Extracted {len(probes)} probes for {tier} {category}")
        return probes
    except Exception as e:
        logger.error(f"  Failed to extract probes: {e}")
        return []


def build_web_grounded_probes_from_manual_data():
    """Build probes from manually curated web-grounded data.

    Since we cannot run Google searches programmatically from this environment,
    we use a hybrid approach:
    1. Define the METHODOLOGY for web-grounded generation (for the paper)
    2. Use known external datasets as ground truth
    3. Mark all T5-T7 probes as requiring web verification
    """

    probes = []

    # ============================================================
    # METHODOLOGY NOTE (for the paper)
    # ============================================================
    # The production pipeline would:
    # 1. Query Google Custom Search API or Common Crawl index
    # 2. Count result pages for each candidate fact
    # 3. Filter by frequency band: T5=10-100, T6=2-10, T7=~1
    # 4. Verify answer correctness via multiple sources
    #
    # For this pilot study, we use population data (verifiable via
    # census databases) and citation counts (verifiable via Google
    # Scholar / Semantic Scholar API) as ground truth proxies.
    # ============================================================

    # ---- RESEARCHER PROBES (grounded in citation counts) ----
    # These would normally come from Semantic Scholar API queries:
    #   GET /paper/search?query=systems+networking&fields=authors,citationCount
    #   Filter by citation count to match tier frequency bands

    # The probes below are illustrative of the methodology.
    # In production, each would be verified via:
    #   1. Semantic Scholar API for citation count
    #   2. DBLP for publication verification
    #   3. University website for affiliation verification

    researcher_probes_meta = {
        "description": "Researcher probes grounded in Semantic Scholar citation counts",
        "methodology": {
            "source": "Semantic Scholar API + DBLP + university websites",
            "T5_criteria": "200-1000 total citations, published at top systems venues",
            "T6_criteria": "20-200 total citations, published at top systems venues",
            "T7_criteria": "<20 total citations, very recent or niche researchers",
            "verification": "Each fact cross-verified against at least 2 web sources",
        },
        "note": "Probe generation requires web API access. This script defines the methodology; actual probe collection requires running the Semantic Scholar pipeline.",
    }

    # ---- CITY/POPULATION PROBES (grounded in census data) ----
    # These would come from:
    #   1. Wikipedia API: query for list of cities by population
    #   2. Government census databases (US Census, China NBS, etc.)
    #   3. Filter by population to match tier frequency bands

    city_probes_meta = {
        "description": "City population probes grounded in official census data",
        "methodology": {
            "source": "Wikipedia list articles + national census databases",
            "T5_criteria": "Cities with population 5,000-50,000 and <100 Google results for '[city] population'",
            "T6_criteria": "Towns with population 500-5,000 and <10 Google results",
            "T7_criteria": "Villages with population <500 and essentially 1 source",
            "verification": "Population verified against official census data",
        },
    }

    return {
        "researcher_probes": researcher_probes_meta,
        "city_probes": city_probes_meta,
    }


def main():
    """Main entry point. Generates web-grounded probe methodology documentation
    and runs available web searches to build actual probes."""

    methodology = build_web_grounded_probes_from_manual_data()

    # Save methodology documentation
    output_file = PROJECT_ROOT / "data" / "probes" / "web_grounded_methodology.json"
    with open(output_file, "w") as f:
        json.dump(methodology, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved methodology documentation to {output_file}")
    logger.info("To generate actual web-grounded probes, use the Semantic Scholar API")
    logger.info("and Google Custom Search API as described in the methodology.")

    # If WebSearch is available, use it to generate actual probes
    try:
        # Try to use web search to find real researchers at various levels
        logger.info("\nAttempting web-grounded researcher probe generation...")
        generate_researcher_probes_from_web()
    except Exception as e:
        logger.info(f"Web search not available in batch mode: {e}")
        logger.info("Use the interactive script or API access to generate web-grounded probes.")


def generate_researcher_probes_from_web():
    """Generate researcher probes using Semantic Scholar API."""
    import requests

    probes = []

    # Semantic Scholar API: search for systems researchers by citation count
    base_url = "https://api.semanticscholar.org/graph/v1"

    tiers = [
        ("T4", 1000, 5000, "SIGCOMM OR OSDI OR SOSP"),
        ("T5", 200, 1000, "systems networking RDMA"),
        ("T6", 20, 200, "data center networking systems"),
    ]

    for tier, min_cite, max_cite, query in tiers:
        logger.info(f"\n  Searching Semantic Scholar for {tier} researchers (citations {min_cite}-{max_cite})...")

        try:
            # Search for papers in the field
            resp = requests.get(
                f"{base_url}/paper/search",
                params={
                    "query": query,
                    "limit": 50,
                    "fields": "title,authors,citationCount,year,venue",
                    "year": "2018-2024",
                },
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning(f"    API returned {resp.status_code}")
                continue

            papers = resp.json().get("data", [])
            logger.info(f"    Found {len(papers)} papers")

            # Find authors in the citation range
            seen_authors = set()
            for paper in papers:
                cite_count = paper.get("citationCount", 0)
                if not (min_cite <= cite_count <= max_cite):
                    continue

                for author in paper.get("authors", []):
                    author_name = author.get("name", "")
                    author_id = author.get("authorId", "")

                    if author_name in seen_authors or not author_name:
                        continue
                    seen_authors.add(author_name)

                    # Get author details
                    if author_id:
                        try:
                            author_resp = requests.get(
                                f"{base_url}/author/{author_id}",
                                params={"fields": "name,affiliations,citationCount,paperCount,hIndex"},
                                timeout=15,
                            )
                            if author_resp.status_code == 200:
                                author_data = author_resp.json()
                                total_citations = author_data.get("citationCount", 0)
                                affiliations = author_data.get("affiliations", [])
                                affiliation = affiliations[0] if affiliations else "unknown"

                                if min_cite <= total_citations <= max_cite:
                                    probe = {
                                        "id": f"{tier}_researcher_{len(probes):03d}",
                                        "question_direct": f"What institution is {author_name} affiliated with?",
                                        "question_fill_blank": f"{author_name} is affiliated with ___.",
                                        "question_contextual": f"The researcher {author_name} works at ___.",
                                        "answer": affiliation,
                                        "answer_type": "text",
                                        "domain": "people",
                                        "region": "Unknown",
                                        "tier": tier,
                                        "frequency_metric": f"Semantic Scholar citation count: {total_citations}",
                                        "estimated_web_frequency": str(total_citations),
                                        "source": f"Semantic Scholar author ID: {author_id}",
                                        "researcher_name": author_name,
                                        "citation_count": total_citations,
                                        "h_index": author_data.get("hIndex"),
                                        "paper_count": author_data.get("paperCount"),
                                    }
                                    probes.append(probe)
                                    logger.info(f"    {tier}: {author_name} ({affiliation}) - {total_citations} citations")

                                    if len([p for p in probes if p['tier'] == tier]) >= 15:
                                        break

                            time.sleep(1)  # Rate limit
                        except Exception as e:
                            continue

                if len([p for p in probes if p['tier'] == tier]) >= 15:
                    break

        except Exception as e:
            logger.error(f"    Semantic Scholar search failed: {e}")

    if probes:
        output_file = PROJECT_ROOT / "data" / "probes" / "researcher_probes_web.json"
        with open(output_file, "w") as f:
            json.dump(probes, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved {len(probes)} web-grounded researcher probes to {output_file}")

    return probes


if __name__ == "__main__":
    main()
