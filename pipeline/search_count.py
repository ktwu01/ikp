#!/usr/bin/env python3
"""Get Google search result counts for probes.

Supports multiple backends:
  1. Serper.dev (google.serper.dev) — needs SERPER_API_KEY
  2. Brave Search API — needs BRAVE_API_KEY
  3. Google Custom Search JSON API — needs GOOGLE_API_KEY + GOOGLE_CSE_ID

Usage:
  python -m pipeline.search_count --test        Test which backend works
  python -m pipeline.search_count --run          Run on all valid probes
"""

import os
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# API keys from environment
SEARCHAPI_KEY = os.environ.get("SEARCHAPI_KEY", "RkEfseTtkEvoCrLdhYsZf11H")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")


def search_serper(query: str) -> int | None:
    """Get total result count from Serper.dev (Google)."""
    with httpx.Client(timeout=30) as http:
        r = http.post("https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 1})
        if r.status_code == 200:
            total = r.json().get("searchInformation", {}).get("totalResults")
            if total is not None:
                return int(total)
    return None


def search_brave(query: str) -> int | None:
    """Get total result count from Brave Search API."""
    with httpx.Client(timeout=30) as http:
        r = http.get("https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 1},
            headers={
                "X-Subscription-Token": BRAVE_API_KEY,
                "Accept": "application/json",
            })
        if r.status_code == 200:
            data = r.json()
            # Brave returns estimated total in web.totalResults or query.totalResults
            total = data.get("web", {}).get("totalResults")
            if total is None:
                total = data.get("query", {}).get("totalResults")
            if total is not None:
                return int(total)
    return None


def search_google_cse(query: str) -> int | None:
    """Get total result count from Google Custom Search JSON API."""
    with httpx.Client(timeout=30) as http:
        r = http.get("https://www.googleapis.com/customsearch/v1",
            params={
                "q": query,
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "num": 1,
            })
        if r.status_code == 200:
            total = r.json().get("searchInformation", {}).get("totalResults")
            if total is not None:
                return int(total)
    return None


def search_searchapi(query: str) -> int | None:
    """Get total result count from SearchAPI.io (Google)."""
    with httpx.Client(timeout=30) as http:
        r = http.get("https://www.searchapi.io/api/v1/search",
            params={"engine": "google", "q": query, "api_key": SEARCHAPI_KEY})
        if r.status_code == 200:
            total = r.json().get("search_information", {}).get("total_results")
            if total is not None:
                return int(total)
    return None


def get_search_count(query: str) -> int | None:
    """Try all available backends, return first success."""
    if SEARCHAPI_KEY:
        result = search_searchapi(query)
        if result is not None:
            return result

    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        result = search_google_cse(query)
        if result is not None:
            return result

    if SERPER_API_KEY:
        result = search_serper(query)
        if result is not None:
            return result

    if BRAVE_API_KEY:
        result = search_brave(query)
        if result is not None:
            return result

    return None


def test_backends():
    """Test which backends are available and working."""
    test_query = "University of Bologna founded year"

    print("Testing search backends...")

    if SEARCHAPI_KEY:
        result = search_searchapi(test_query)
        print(f"  SearchAPI.io: {result:,}" if result else "  SearchAPI.io: FAILED")
    else:
        print("  SearchAPI.io: no API key (set SEARCHAPI_KEY)")

    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        result = search_google_cse(test_query)
        print(f"  Google CSE: {result:,}" if result else "  Google CSE: FAILED")
    else:
        print("  Google CSE: no API key (set GOOGLE_API_KEY + GOOGLE_CSE_ID)")

    if SERPER_API_KEY:
        result = search_serper(test_query)
        print(f"  Serper: {result:,}" if result else "  Serper: FAILED")
    else:
        print("  Serper: no API key (set SERPER_API_KEY)")

    if BRAVE_API_KEY:
        result = search_brave(test_query)
        print(f"  Brave: {result:,}" if result else "  Brave: FAILED")
    else:
        print("  Brave: no API key (set BRAVE_API_KEY)")

    # Test with a well-known and an obscure query
    if any([GOOGLE_API_KEY, SERPER_API_KEY, BRAVE_API_KEY]):
        print("\nSample queries:")
        samples = [
            "University of Bologna founded year",
            "Ksar Beni Ghedir founded year",
            "What is the chemical symbol for gold",
            "What is the research field of Scott Shenker",
        ]
        for q in samples:
            count = get_search_count(q)
            print(f"  '{q[:50]}': {count:,}" if count else f"  '{q[:50]}': None")


def run_on_store(workers: int = 4):
    """Run search counts on all valid probes in the store."""
    from .store import ProbeStore

    store = ProbeStore(PROJECT_ROOT / "data" / "pipeline_store.json")
    valid = [p for p in store.probes.values()
             if p["status"] == "valid" and "search_count" not in p]

    if not valid:
        print("No valid probes without search counts.")
        return

    print(f"Getting search counts for {len(valid)} probes ({workers} workers)...")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(get_search_count, p["question"]): p for p in valid}
        for future in as_completed(futures):
            probe = futures[future]
            try:
                count = future.result()
                probe["search_count"] = count
                done += 1
                if done % 50 == 0:
                    store.save()
                    print(f"  {done}/{len(valid)} done", flush=True)
            except Exception as e:
                probe["search_count"] = None
                done += 1

    store.save()
    print(f"Done. {done} probes updated.")

    # Quick analysis
    probes_with_counts = [p for p in store.probes.values()
                          if p.get("search_count") is not None and p.get("tier")]
    if probes_with_counts:
        from collections import defaultdict
        import statistics

        tier_counts = defaultdict(list)
        for p in probes_with_counts:
            tier_counts[p["tier"]].append(p["search_count"])

        print("\nSearch count by tier (median):")
        for tier in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
            counts = tier_counts.get(tier, [])
            if counts:
                med = statistics.median(counts)
                print(f"  {tier}: median={med:>12,.0f}  (n={len(counts)})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test which backends work")
    parser.add_argument("--run", action="store_true", help="Run on all valid probes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.test:
        test_backends()
    elif args.run:
        run_on_store(workers=args.workers)
    else:
        test_backends()
