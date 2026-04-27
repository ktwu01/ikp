"""Harvest candidates for the 3 remaining subfields (OS, arch, theory) using
better, more specific OpenAlex concept IDs that don't get diluted by
non-CS authors.

For OS:    Linux kernel (C553261973), Hypervisor (C112904061),
           Memory hierarchy (C2778100165)
For arch:  Computer architecture (C118524514) — level-1 concept,
           filter strictly by CS top-field
For theory: Graph algorithms (C2986651925), Approximation algorithm (C148764684),
           Computational complexity theory (C179799912)
"""

import json
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path("/Users/boj/ikp-paper")
sys.path.insert(0, str(ROOT))

EXISTING_CIT = ROOT / "data" / "researcher_citations.json"
CANDIDATES = ROOT / "data" / "probes" / "t7_candidates.json"

# Specific concept IDs that index real CS authors
SUBFIELD_CONCEPTS = {
    "operating systems":           ["C553261973", "C112904061", "C2778100165"],
    "computer architecture":       ["C118524514", "C80469333"],
    "theoretical computer science":["C2986651925", "C148764684", "C179799912"],
}

DROPPED_BY_SUBFIELD = {
    "operating systems":           ["IKP_T7_1252"],
    "computer architecture":       ["IKP_T7_1283"],
    "theoretical computer science":["IKP_T7_1286"],
}

KEYWORD_MAP = {
    "operating systems": ["operating system", "kernel", "virtualization", "hypervisor",
                           "scheduler", "memory hierarchy", "memory management",
                           "concurren", "scheduling", "system software", "linux",
                           "filesystem", "file system", "container", "embedded system"],
    "computer architecture": ["architecture", "processor", "microprocessor", "vlsi",
                              "cache", "gpu", "memory", "fpga", "asic", "hardware",
                              "circuit", "low-power"],
    "theoretical computer science":["algorithm", "complexity", "graph theor", "approximation",
                                    "combinatorial", "discrete", "polynomial",
                                    "computational", "logic"],
}

CS_TOKENS = ["computer", "computational", "computing", "informatics", "acm",
             "ieee transactions", "lncs", "lecture notes in computer",
             "sigplan", "sigmod", "sigops", "sigcomm", "vldb", "popl", "pldi",
             "ndss", "usenix", "neurips", "icml", "iclr", "aaai", "ijcai",
             "kdd", "icde", "cvpr", "iccv", "eccv", "chi", "uist", "cscw",
             "siggraph", "imc", "mobicom", "isca", "asplos", "micro", "hpca",
             "fast", "ccs", "oakland", "stoc", "focs", "soda", "icalp", "esa", "arxiv"]


def is_cs_venue(v):
    if not v: return False
    return any(t in v.lower() for t in CS_TOKENS)


def harvest_authors(http, concept_id, year_range=(2021, 2024), n_works=120):
    url = "https://api.openalex.org/works"
    yr_min, yr_max = year_range
    params = {
        "filter": f"concepts.id:{concept_id},cited_by_count:5-30,publication_year:{yr_min}-{yr_max}",
        "per-page": n_works,
        "select": "authorships,publication_year,cited_by_count",
    }
    for _ in range(3):
        try:
            r = http.get(url, params=params, timeout=30)
            if r.status_code == 200:
                authors = []
                for w in r.json().get("results", []):
                    for a in (w.get("authorships") or []):
                        au = a.get("author") or {}
                        aid = au.get("id"); name = au.get("display_name", "").strip()
                        if aid and name:
                            authors.append((aid, name))
                return authors
        except Exception:
            time.sleep(1)
    return []


def fetch_author(http, aid):
    short = aid.split("/")[-1]
    for _ in range(3):
        try:
            r = http.get(f"https://api.openalex.org/authors/{short}", timeout=30)
            if r.status_code == 200: return r.json()
        except Exception: time.sleep(1)
    return {}


def fetch_works(http, aid, limit=10):
    short = aid.split("/")[-1]
    for _ in range(3):
        try:
            r = http.get("https://api.openalex.org/works",
                          params={"filter": f"author.id:{short}", "per-page": limit,
                                  "sort": "cited_by_count:desc"}, timeout=30)
            if r.status_code == 200: return r.json().get("results", [])
        except Exception: time.sleep(1)
    return []


def vet(http, author, target_subfield, debug=False):
    aid = author.get("id");
    if not aid: return None
    name = author.get("display_name", "").strip()
    if not name: return None
    h = (author.get("summary_stats") or {}).get("h_index", 0)
    cited = author.get("cited_by_count", 0)
    if not (3 <= h <= 16): return None
    if not (30 <= cited <= 1500): return None
    topics = author.get("topics") or []
    if not topics: return None
    top_field = (topics[0].get("field") or {}).get("display_name", "").lower()
    cs_adjacent = ["computer", "informatics", "engineering", "mathematics"]
    if not any(k in top_field for k in cs_adjacent):
        any_cs = any("computer" in (t.get("field") or {}).get("display_name", "").lower()
                     for t in topics[:3])
        if not any_cs:
            if debug: print(f"      reject {name}: top_field='{top_field}'", flush=True)
            return None
    works = fetch_works(http, aid, limit=10)
    if len(works) < 3: return None
    cs_count = 0; venues, titles = [], []; coauth = {}
    for w in works:
        primary_loc = w.get("primary_location") or {}
        src = primary_loc.get("source") or {}
        venue = src.get("display_name") or ""
        if is_cs_venue(venue): cs_count += 1
        if venue: venues.append(venue)
        title = w.get("title") or w.get("display_name") or ""
        if title: titles.append(title)
        for a in (w.get("authorships") or []):
            disp = (a.get("author") or {}).get("display_name")
            other_id = ((a.get("author") or {}).get("id") or "").split("/")[-1]
            if disp and other_id != aid.split("/")[-1]:
                coauth[disp] = coauth.get(disp, 0) + 1
    if cs_count < 1:
        if debug: print(f"      reject {name}: cs_count=0, venues={venues[:2]}", flush=True)
        return None
    topic_names = " ".join((t.get("display_name") or "").lower() for t in topics[:5])
    if not any(k in topic_names for k in KEYWORD_MAP[target_subfield]):
        if debug: print(f"      reject {name}: subfield mismatch", flush=True)
        return None
    co_authors = [n for n, _ in sorted(coauth.items(), key=lambda x: -x[1])[:5]]
    affils = []
    for inst in (author.get("last_known_institutions") or []):
        n = inst.get("display_name");
        if n: affils.append(n)
    return {
        "openalex_id": aid, "name": name, "primary_subfield": target_subfield,
        "h_index": h, "cited_by_count": cited,
        "topics": [(t.get("display_name") or "") for t in topics[:3]],
        "top_titles": titles[:8], "top_venues": venues[:5],
        "cs_venue_count": cs_count, "affiliations": affils[:2],
        "co_authors": co_authors,
    }


MAX_ATTEMPTS = 250


def main():
    with open(EXISTING_CIT) as f: existing = json.load(f)
    used_ids = {c["openalex_id"].split("/")[-1] for c in existing if c.get("openalex_id")}
    used_names = {c["name"].lower() for c in existing if c.get("name")}

    pools = json.load(open(CANDIDATES))
    http = httpx.Client(timeout=30)

    for subfield, concepts in SUBFIELD_CONCEPTS.items():
        if pools.get(subfield):
            print(f"  {subfield}: already has {len(pools[subfield])}, skipping", flush=True)
            continue
        n_needed = len(DROPPED_BY_SUBFIELD[subfield])
        target_pool = max(8, n_needed * 8)
        print(f"\n  {subfield}: target {target_pool}", flush=True)

        author_seen = set(); author_counts = Counter()
        for cid in concepts:
            print(f"    fetching {cid}...", flush=True)
            authors = harvest_authors(http, cid)
            print(f"    got {len(authors)} entries", flush=True)
            for aid, name in authors:
                short = aid.split("/")[-1]
                if short in used_ids or name.lower() in used_names: continue
                if short in author_seen:
                    author_counts[short] += 1; continue
                author_seen.add(short); author_counts[short] += 1
        print(f"    {len(author_seen)} unique candidates", flush=True)

        ordered = sorted(author_seen, key=lambda x: -author_counts[x])
        kept = []
        n_attempted = 0
        for short in ordered:
            if len(kept) >= target_pool: break
            if n_attempted >= MAX_ATTEMPTS: break
            n_attempted += 1
            full_id = f"https://openalex.org/{short}"
            au = fetch_author(http, full_id)
            if not au: continue
            cand = vet(http, au, subfield, debug=(n_attempted <= 5))
            if cand:
                kept.append(cand)
                print(f"    + {cand['name']} (h={cand['h_index']}, cited={cand['cited_by_count']})", flush=True)
            if n_attempted % 25 == 0:
                print(f"    [{subfield}] vetted {n_attempted}, kept {len(kept)}", flush=True)
        print(f"  → {subfield}: {len(kept)} candidates ({n_attempted} attempted)", flush=True)
        pools[subfield] = kept
        with open(CANDIDATES, "w") as f:
            json.dump(pools, f, indent=2, ensure_ascii=False)
    http.close()
    print(f"\nFinal pools: {[(sf, len(v)) for sf, v in pools.items()]}", flush=True)


if __name__ == "__main__":
    main()
