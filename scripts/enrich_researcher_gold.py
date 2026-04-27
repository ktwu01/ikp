"""Enrich researcher probe gold answers with OpenAlex evidence.

For each of the 345 researcher probes, pull from OpenAlex:
  - Top works (titles, venues, publication years, citation counts)
  - Last known institutional affiliations
  - Top concepts/topics with scores
  - Top co-authors (most-collaborated)

Detect likely name collisions:
  - If majority of top-N venues are non-CS (e.g., medicine, chemistry, physics journals)
  - If primary concept is non-CS

Output:
  data/probes/researcher_gold_enriched.json — list of enriched gold bundles
  data/probes/researcher_collisions.json    — list of probe_ids flagged as collisions
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path("/Users/boj/ikp-paper")
SRC = ROOT / "data" / "researcher_citations.json"
PROBES = ROOT / "data" / "probes" / "final_probe_set_v8.json"
OUT_ENRICHED = ROOT / "data" / "probes" / "researcher_gold_enriched.json"
OUT_COLLISIONS = ROOT / "data" / "probes" / "researcher_collisions.json"

N_WORKERS = 8
TOP_N_WORKS = 10

# CS venue keyword set (lowercase). If at least 2/top-10 venues match, treat as CS researcher.
CS_VENUE_KEYWORDS = [
    # Top-tier CS conferences and abbreviations
    "sigcomm", "nsdi", "sigmod", "vldb", "sosp", "osdi", "asplos", "isca", "micro",
    "hpca", "fast", "atc", "eurosys", "usenix security", "ndss", "ccs", "oakland",
    "s&p", "ieee s&p", "icml", "neurips", "iclr", "aaai", "ijcai", "cvpr", "iccv",
    "eccv", "kdd", "icde", "pldi", "popl", "oopsla", "icse", "fse", "ase",
    "stoc", "focs", "soda", "podc", "spaa", "lics", "cav", "tacas", "chi",
    "uist", "siggraph", "infocom", "icnp", "imc", "conext", "mobicom", "mobisys",
    "sensys", "ipsn", "rss", "icra", "iros",
    # ACM/IEEE journals (CS)
    "ieee transactions on computers", "ieee transactions on parallel and distributed",
    "acm transactions on", "ieee/acm transactions on networking",
    "ieee transactions on knowledge and data", "ieee transactions on software",
    "ieee transactions on information forensics", "ieee transactions on dependable",
    "ieee transactions on cloud", "ieee transactions on services computing",
    "ieee transactions on mobile computing", "ieee transactions on wireless",
    "ieee transactions on vehicular technology", "ieee transactions on cybernetics",
    "ieee transactions on neural networks", "ieee transactions on image processing",
    "ieee transactions on pattern analysis", "ieee transactions on multimedia",
    "ieee transactions on visualization", "ieee transactions on robotics",
    "ieee transactions on circuits and systems", "ieee transactions on vlsi",
    "ieee transactions on industrial informatics", "ieee transactions on instrumentation",
    "ieee transactions on automation",
    "communications of the acm", "journal of the acm", "jacm",
    "computer networks", "computers & security", "future generation computer systems",
    "journal of systems and software", "software practice and experience",
    "information sciences", "information systems",
    "performance evaluation", "ad hoc networks",
    # Generic CS terms in venue names
    "computer science", "computer systems", "computer security", "computer networking",
    "operating systems", "distributed systems", "parallel computing",
    "machine learning", "artificial intelligence", "deep learning",
    "data mining", "database", "programming language",
    "software engineering", "software architecture",
    "embedded systems", "real-time systems", "cyber-physical",
    "computer vision", "natural language processing",
    "information security", "network security",
    "high performance computing", "supercomputing",
    "human computer interaction", "human-computer interaction", "user interface",
    "robotics", "autonomous", "internet of things", "iot",
    # Workshops and proceedings
    "proceedings of the", "acm symposium", "ieee symposium",
    "international conference on", "international workshop on",
]

# Non-CS keywords that indicate likely name collision
NON_CS_KEYWORDS = [
    "chemistry", "biochemistry", "biology", "biological", "medicine", "medical",
    "clinical", "surgery", "surgical", "oncology", "cancer", "cardiology",
    "cardiac", "neurology", "psychiatry", "pediatric", "physiology",
    "physics", "physical review", "astrophysical", "astronomy",
    "geology", "geological", "earth science", "environmental",
    "agriculture", "agricultural", "veterinary",
    "law journal", "social science", "psychology", "sociology",
    "economics", "finance",
    "metrologia", "angewandte chemie", "chemical society",
    "nature", "science",  # ambiguous but check majority
]


def is_cs_venue(venue_name: str) -> bool:
    if not venue_name:
        return False
    v = venue_name.lower()
    if any(k in v for k in NON_CS_KEYWORDS) and not any(k in v for k in [
        "computational", "computer", "informatics", "computing"]):
        return False
    return any(k in v for k in CS_VENUE_KEYWORDS)


def classify_venue(venue_name: str) -> str:
    """Returns 'cs', 'noncs', or 'unknown'."""
    if not venue_name:
        return "unknown"
    v = venue_name.lower()
    if any(k in v for k in NON_CS_KEYWORDS):
        # Override: 'nature communications' / 'nature machine intelligence' / etc. can still be CS
        if "computational" in v or "machine intelligence" in v:
            return "cs"
        return "noncs"
    if any(k in v for k in CS_VENUE_KEYWORDS):
        return "cs"
    return "unknown"


def extract_named_systems(titles: list[str]) -> list[str]:
    """Heuristically extract named systems/artifacts from paper titles.

    Looks for ALLCAPS short tokens (2-8 chars) or CamelCase names that appear
    to be system names (typically appear as the first token followed by a colon
    or a noun phrase like 'a framework' / 'an OS' / 'a system').
    """
    systems = set()
    for title in titles:
        if not title:
            continue
        # Pattern: "NAME: rest of title" — common system-paper convention
        m = re.match(r"^([A-Z][A-Za-z0-9\-_+]{1,15}|[A-Z]{2,10}):\s+", title)
        if m:
            cand = m.group(1)
            # Skip if it's a common English word
            if cand.lower() not in {"the", "a", "an", "on", "in", "to", "from", "with",
                                    "using", "toward", "towards", "improving",
                                    "understanding", "learning", "rethinking",
                                    "scaling", "is", "do", "can", "why", "how", "what"}:
                systems.add(cand)
        # Pattern: ALLCAPS standalone token of 3-10 chars in title
        for tok in re.findall(r"\b([A-Z]{3,10})\b", title):
            if tok not in {"GPU", "CPU", "GPUs", "CPUs", "HPC", "ACM", "IEEE",
                           "USA", "PDF", "API", "OS", "RAM", "ROM", "USB", "DSL",
                           "AI", "ML", "DL", "NN", "RL", "RNN", "CNN", "LLM", "NLP",
                           "DNN", "DNS", "TCP", "UDP", "HTTP", "HTTPS", "TLS",
                           "VLSI", "RTL", "FPGA", "ASIC", "ARM", "x86", "IO", "I/O"}:
                systems.add(tok)
    return sorted(systems)


def fetch_works(http: httpx.Client, openalex_id: str) -> list[dict]:
    aid = openalex_id.split("/")[-1]
    url = f"https://api.openalex.org/works?filter=author.id:{aid}&per-page={TOP_N_WORKS}&sort=cited_by_count:desc"
    for attempt in range(3):
        try:
            r = http.get(url, timeout=30)
            if r.status_code == 200:
                return r.json().get("results", [])
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return []
        except Exception:
            time.sleep(1)
    return []


def fetch_author(http: httpx.Client, openalex_id: str) -> dict:
    aid = openalex_id.split("/")[-1]
    url = f"https://api.openalex.org/authors/{aid}"
    for attempt in range(3):
        try:
            r = http.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return {}
        except Exception:
            time.sleep(1)
    return {}


def enrich_one(probe_id, name, openalex_url, gold_subfield):
    http = httpx.Client(timeout=30)
    try:
        author = fetch_author(http, openalex_url)
        works = fetch_works(http, openalex_url)

        # Affiliations
        affiliations = []
        for inst in author.get("last_known_institutions", []) or []:
            n = inst.get("display_name")
            if n:
                affiliations.append(n)
        if not affiliations:
            for inst_record in (author.get("affiliations") or [])[:3]:
                inst = inst_record.get("institution") or {}
                n = inst.get("display_name")
                if n and n not in affiliations:
                    affiliations.append(n)

        # Top concepts/topics
        topics = []
        for t in (author.get("topics") or [])[:5]:
            disp = t.get("display_name") or ""
            field = (t.get("field") or {}).get("display_name", "")
            topics.append({"name": disp, "field": field})

        # Works → titles + venues + years
        work_records = []
        cs_count = 0
        noncs_count = 0
        for w in works:
            title = w.get("title") or w.get("display_name") or ""
            yr = w.get("publication_year")
            cited = w.get("cited_by_count", 0)
            venue = ""
            primary_loc = w.get("primary_location") or {}
            src = primary_loc.get("source") or {}
            venue = src.get("display_name") or ""
            if not venue:
                # Fallback: host_organization
                venue = src.get("host_organization_name") or ""
            cls = classify_venue(venue)
            if cls == "cs":
                cs_count += 1
            elif cls == "noncs":
                noncs_count += 1
            work_records.append({
                "title": title,
                "year": yr,
                "venue": venue,
                "cited": cited,
                "venue_cls": cls,
            })

        # Co-authors: aggregate over top works
        coauthor_counts = {}
        for w in works:
            for auth in (w.get("authorships") or []):
                a = auth.get("author") or {}
                disp = a.get("display_name")
                aid = (a.get("id") or "").split("/")[-1]
                if not disp or aid == openalex_url.split("/")[-1]:
                    continue
                coauthor_counts[disp] = coauthor_counts.get(disp, 0) + 1
        co_authors = sorted(coauthor_counts.items(), key=lambda x: -x[1])[:5]

        # Named systems from titles
        named_systems = extract_named_systems([w["title"] for w in work_records])

        # Venues list (deduplicated, top by frequency among CS venues)
        venue_counter = {}
        for w in work_records:
            v = w["venue"]
            if not v: continue
            venue_counter[v] = venue_counter.get(v, 0) + 1
        top_venues = sorted(venue_counter.items(), key=lambda x: -x[1])[:8]

        # Collision detection: noncs venues dominate
        collision_flag = False
        collision_reason = None
        if (cs_count + noncs_count) >= 5:
            if noncs_count > cs_count:
                collision_flag = True
                collision_reason = f"non-CS venue majority ({noncs_count} non-CS vs {cs_count} CS in top {len(work_records)})"
        if cs_count == 0 and noncs_count >= 3:
            collision_flag = True
            collision_reason = f"no CS venues in top {len(work_records)}"
        # Topic check
        if topics:
            top_field = topics[0].get("field", "").lower()
            if top_field and top_field not in ("computer science",) and not collision_flag:
                # Only flag if the top field is clearly non-CS (chemistry, medicine, physics, etc.)
                if any(k in top_field for k in ("chem", "biol", "medic", "phys",
                                                  "geol", "agricul", "law",
                                                  "social", "econom", "psycho")):
                    collision_flag = True
                    collision_reason = f"top OpenAlex field is '{topics[0]['field']}', not CS"

        return {
            "probe_id": probe_id,
            "name": name,
            "openalex_id": openalex_url,
            "primary_subfield": gold_subfield,
            "secondary_subfields": [],  # filled in by post-process from topics
            "affiliations": affiliations,
            "venues": [v for v, _ in top_venues],
            "named_systems": named_systems,
            "co_authors": [c for c, _ in co_authors],
            "top_works": [{"title": w["title"], "year": w["year"], "cited": w["cited"]}
                          for w in work_records[:5]],
            "topics": topics,
            "cs_venue_count": cs_count,
            "noncs_venue_count": noncs_count,
            "collision_flag": collision_flag,
            "collision_reason": collision_reason,
        }
    finally:
        http.close()


def main():
    with open(SRC) as f:
        cit = json.load(f)
    with open(PROBES) as f:
        probes = json.load(f)

    researcher = [p for p in probes if p.get("source_type") == "researcher"]
    by_pid = {c["probe_id"]: c for c in cit}

    targets = []
    for p in researcher:
        c = by_pid.get(p["id"])
        if not c or not c.get("openalex_id"):
            print(f"  [SKIP] {p['id']}: no OpenAlex id")
            continue
        targets.append((p["id"], c["name"], c["openalex_id"], p["answer"]))

    print(f"Enriching {len(targets)} researcher probes via OpenAlex...")

    enriched = []
    collisions = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(enrich_one, pid, name, oid, gold): pid
                for pid, name, oid, gold in targets}
        for i, fut in enumerate(as_completed(futs)):
            try:
                rec = fut.result()
            except Exception as e:
                pid = futs[fut]
                print(f"  [ERR] {pid}: {e}")
                continue
            enriched.append(rec)
            if rec["collision_flag"]:
                collisions.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(targets)}  collisions so far: {len(collisions)}")

    enriched.sort(key=lambda r: r["probe_id"])
    collisions.sort(key=lambda r: r["probe_id"])

    with open(OUT_ENRICHED, "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    with open(OUT_COLLISIONS, "w") as f:
        json.dump([{"probe_id": c["probe_id"], "name": c["name"],
                    "reason": c["collision_reason"],
                    "venues_sample": c["venues"][:3]}
                   for c in collisions], f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUT_ENRICHED} ({len(enriched)} entries)")
    print(f"Wrote {OUT_COLLISIONS} ({len(collisions)} collision flags)")

    # Tier breakdown of collisions
    from collections import Counter
    tier_coll = Counter()
    for c in collisions:
        # tier from probe_id e.g. IKP_T6_1003
        parts = c["probe_id"].split("_")
        if len(parts) >= 2:
            tier_coll[parts[1]] += 1
    print(f"Collisions by tier: {dict(tier_coll)}")


if __name__ == "__main__":
    main()
