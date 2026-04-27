"""Find replacement T7 CS researchers for the 16 Case-B probes.

Pipeline:
  1. For each needed subfield, harvest authors from RECENT (2021-2024) low-cited
     CS papers tagged with the right concept ID. This biases toward early-career
     / niche researchers, which is the T7 demographic.
  2. For each unique author, fetch profile (h-index, citation, topics, works).
     Keep only h-index 4-14, cited 80-800, top_field includes "Computer Science",
     ≥2 CS venues in top 10 works, name not already in probe set.
  3. For each surviving candidate, build the probe question and run it through
     the 6 landmark models (pipeline/landmarks.py). T7 = no landmark answers.
  4. Keep first n_needed qualifying T7 candidates per subfield.

Reads:  data/researcher_citations.json
Writes: data/probes/t7_candidates.json   (full pool with calibration verdicts)
        data/probes/t7_replacements.json (final n=16 substitutions)
"""

import json
import os
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path("/Users/boj/ikp-paper")
sys.path.insert(0, str(ROOT))
from pipeline.landmarks import LANDMARKS, query_landmark
from pipeline.judge import judge

EXISTING_CIT = ROOT / "data" / "researcher_citations.json"
ENRICHED = ROOT / "data" / "probes" / "researcher_gold_enriched.json"
OUT_CANDIDATES = ROOT / "data" / "probes" / "t7_candidates.json"
OUT_REPLACEMENTS = ROOT / "data" / "probes" / "t7_replacements.json"

# OpenAlex concept IDs for each CS subfield
SUBFIELD_CONCEPTS = {
    "human-computer interaction":  ["C107457646"],
    "computer security":           ["C38652104"],   # Computer security
    "programming languages":       ["C169590947",   # Compiler
                                     "C18701968",   # Programming language theory
                                     "C111498074"], # Formal verification
    "computer networking":         ["C31258907"],   # Computer network
    "computer vision":             ["C31972630"],   # Computer vision
    "operating systems":           ["C108713360"],  # Operating system
    "computer architecture":       ["C9390403"],    # Computer hardware (broader)
    "theoretical computer science":["C2522767166"], # Algorithm
}

# Cap attempts per subfield to avoid stalling on broken concepts
MAX_ATTEMPTS_PER_SUBFIELD = 220

# Subfield → list of dropped probe IDs to fill
DROPPED_BY_SUBFIELD = {
    "human-computer interaction":  ["IKP_T7_1200", "IKP_T7_1202", "IKP_T7_1264"],
    "computer security":           ["IKP_T7_1227", "IKP_T7_1289", "IKP_T7_1299"],
    "programming languages":       ["IKP_T7_1244", "IKP_T7_1250", "IKP_T7_1261", "IKP_T7_1273"],
    "computer networking":         ["IKP_T7_1214", "IKP_T7_1218"],
    "computer vision":             ["IKP_T7_1206"],
    "operating systems":           ["IKP_T7_1252"],
    "computer architecture":       ["IKP_T7_1283"],
    "theoretical computer science":["IKP_T7_1286"],
}

KEYWORD_MAP = {
    "human-computer interaction": ["human-computer", "interaction", "interface", "user experience", "accessibility", "hci"],
    "computer security":           ["security", "cryptography", "privacy", "malware", "intrusion"],
    "programming languages":       ["programming language", "compiler", "type system", "verification", "formal method"],
    "computer networking":         ["network", "wireless", "routing", "protocol", "internet"],
    "computer vision":             ["vision", "image", "video", "recognition", "segmentation", "object detection"],
    "operating systems":           ["operating system", "kernel", "virtualization", "hypervisor", "scheduler"],
    "computer architecture":       ["architecture", "processor", "microprocessor", "vlsi", "cache", "gpu"],
    "theoretical computer science":["algorithm", "complexity", "graph theor", "approximation", "combinatorial"],
}

CS_TOKENS = ["computer", "computational", "computing", "informatics", "acm",
             "ieee transactions", "lncs", "lecture notes in computer",
             "sigplan", "sigmod", "sigops", "sigcomm", "vldb", "popl", "pldi",
             "ndss", "usenix", "neurips", "icml", "iclr", "aaai", "ijcai",
             "kdd", "icde", "cvpr", "iccv", "eccv", "chi", "uist", "cscw",
             "siggraph", "imc", "mobicom", "isca", "asplos", "micro", "hpca",
             "fast", "ccs", "oakland", "arxiv"]


def is_cs_venue(v):
    if not v: return False
    return any(t in v.lower() for t in CS_TOKENS)


def harvest_authors_from_works(http, concept_id, n_works=120, year_min=2021, year_max=2024):
    """Harvest unique author IDs from low-cited CS works tagged with concept_id."""
    url = "https://api.openalex.org/works"
    # Bias toward niche: low cited (5-30), recent
    params = {
        "filter": f"concepts.id:{concept_id},cited_by_count:5-30,publication_year:{year_min}-{year_max}",
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
                        author = a.get("author") or {}
                        aid = author.get("id")
                        name = author.get("display_name", "").strip()
                        if aid and name:
                            authors.append((aid, name))
                return authors
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return []


def fetch_author(http, openalex_id):
    aid = openalex_id.split("/")[-1]
    for _ in range(3):
        try:
            r = http.get(f"https://api.openalex.org/authors/{aid}", timeout=30)
            if r.status_code == 200:
                return r.json()
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return {}


def fetch_works(http, openalex_id, limit=10):
    aid = openalex_id.split("/")[-1]
    url = "https://api.openalex.org/works"
    params = {"filter": f"author.id:{aid}", "per-page": limit, "sort": "cited_by_count:desc"}
    for _ in range(3):
        try:
            r = http.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("results", [])
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return []


def vet_author(http, author, target_subfield, debug=False):
    """Returns enriched candidate dict or None."""
    aid = author.get("id")
    if not aid: return None
    name = author.get("display_name", "").strip()
    if not name: return None
    h = (author.get("summary_stats") or {}).get("h_index", 0)
    cited = author.get("cited_by_count", 0)
    if not (3 <= h <= 16):
        if debug: print(f"      reject {name}: h={h} out of range", flush=True)
        return None
    if not (30 <= cited <= 1500):
        if debug: print(f"      reject {name}: cited={cited} out of range", flush=True)
        return None
    topics = author.get("topics") or []
    if not topics:
        if debug: print(f"      reject {name}: no topics", flush=True)
        return None
    top_field = (topics[0].get("field") or {}).get("display_name", "").lower()
    # Accept CS-adjacent fields. HCI authors often classified under Social Sciences
    # or Psychology. Vision can be tagged as Engineering. Theory under Mathematics.
    cs_adjacent = ["computer", "informatics", "engineering", "mathematics"]
    if target_subfield == "human-computer interaction":
        cs_adjacent += ["social", "psycholog"]
    if target_subfield == "computer vision":
        cs_adjacent += ["medicine"]  # medical imaging often
    if not any(k in top_field for k in cs_adjacent):
        # Also accept any author where CS appears anywhere in top-3 topics' fields
        any_cs = False
        for t in topics[:3]:
            f = (t.get("field") or {}).get("display_name", "").lower()
            if any(k in f for k in ["computer", "informatics"]):
                any_cs = True; break
        if not any_cs:
            if debug: print(f"      reject {name}: top_field='{top_field}' not CS-adjacent", flush=True)
            return None
    works = fetch_works(http, aid, limit=10)
    if len(works) < 3:
        if debug: print(f"      reject {name}: only {len(works)} works", flush=True)
        return None
    cs_count = 0
    venues, titles = [], []
    coauth = {}
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
        if debug: print(f"      reject {name}: cs_count={cs_count}, venues sample={venues[:3]}", flush=True)
        return None
    topic_names = " ".join((t.get("display_name") or "").lower() for t in topics[:5])
    if not any(k in topic_names for k in KEYWORD_MAP[target_subfield]):
        if debug: print(f"      reject {name}: subfield mismatch, topics={topic_names[:80]}", flush=True)
        return None
    co_authors = [n for n, _ in sorted(coauth.items(), key=lambda x: -x[1])[:5]]
    affils = []
    for inst in author.get("last_known_institutions", []) or []:
        n = inst.get("display_name")
        if n: affils.append(n)
    return {
        "openalex_id": aid,
        "name": name,
        "primary_subfield": target_subfield,
        "h_index": h,
        "cited_by_count": cited,
        "topics": [(t.get("display_name") or "") for t in topics[:3]],
        "top_titles": titles[:8],
        "top_venues": venues[:5],
        "cs_venue_count": cs_count,
        "affiliations": affils[:2],
        "co_authors": co_authors,
    }


PROBE_TEMPLATE = (
    "In computer science, what is the research subfield of {name}, and name "
    "one paper, system, institution, or co-author associated with their work? "
    "If you don't know who this person is, say so."
)


def calibrate_with_landmarks(question: str, gold_subfield: str) -> dict:
    out = {"per_landmark": {}, "tier": None}
    with ThreadPoolExecutor(max_workers=len(LANDMARKS)) as ex:
        futures = {ex.submit(query_landmark, lm, question): lm for lm in LANDMARKS}
        for fut in as_completed(futures):
            lm = futures[fut]
            try:
                resp = fut.result()
            except Exception:
                resp = ""
            correct = bool(resp and judge(question, gold_subfield, resp))
            out["per_landmark"][lm["name"]] = {
                "correct": correct,
                "response": (resp or "")[:200],
            }
    tier = "T7"
    for lm in LANDMARKS:
        if out["per_landmark"][lm["name"]]["correct"]:
            tier = lm["tier"]
            break
    out["tier"] = tier
    return out


def main():
    with open(EXISTING_CIT) as f:
        existing_cit = json.load(f)
    used_ids = {c["openalex_id"].split("/")[-1] for c in existing_cit if c.get("openalex_id")}
    used_names = {c["name"].lower() for c in existing_cit if c.get("name")}

    http = httpx.Client(timeout=30)

    # PHASE 1: harvest candidates from low-cited CS works
    print("=== Phase 1: harvesting candidates from CS works ===", flush=True)
    pools = defaultdict(list)
    for subfield, dropped_pids in DROPPED_BY_SUBFIELD.items():
        n_needed = len(dropped_pids)
        target_pool = max(20, n_needed * 8)
        print(f"\n  {subfield}: need {n_needed}, target {target_pool}", flush=True)
        author_seen = set()
        author_counts = Counter()
        for cid in SUBFIELD_CONCEPTS[subfield]:
            print(f"    fetching works for concept {cid}...", flush=True)
            authors = harvest_authors_from_works(http, cid, n_works=120)
            print(f"    got {len(authors)} author entries", flush=True)
            for aid, name in authors:
                short_id = aid.split("/")[-1]
                if short_id in used_ids: continue
                if name.lower() in used_names: continue
                if short_id in author_seen:
                    author_counts[short_id] += 1
                    continue
                author_seen.add(short_id)
                author_counts[short_id] += 1
        print(f"    {len(author_seen)} unique authors to vet", flush=True)

        ordered = sorted(author_seen, key=lambda x: -author_counts[x])
        n_attempted = 0
        # Need at least n_needed*2 to leave room for landmark-T7 filtering
        min_acceptable = max(n_needed * 3, 8)
        for short_id in ordered:
            if len(pools[subfield]) >= target_pool:
                break
            if n_attempted >= MAX_ATTEMPTS_PER_SUBFIELD and len(pools[subfield]) >= min_acceptable:
                print(f"    [{subfield}] hit max attempts cap ({MAX_ATTEMPTS_PER_SUBFIELD}); have {len(pools[subfield])} ≥ min_acceptable={min_acceptable}", flush=True)
                break
            n_attempted += 1
            full_id = f"https://openalex.org/{short_id}"
            au = fetch_author(http, full_id)
            if not au: continue
            cand = vet_author(http, au, subfield, debug=(n_attempted <= 5))
            if cand:
                pools[subfield].append(cand)
                print(f"    + {cand['name']} (h={cand['h_index']}, cited={cand['cited_by_count']})", flush=True)
            if n_attempted % 25 == 0:
                print(f"    [{subfield}] vetted {n_attempted} authors, {len(pools[subfield])} kept", flush=True)
        print(f"  → {subfield}: {len(pools[subfield])} candidates ({n_attempted} attempted)", flush=True)
        # Save partial state incrementally
        OUT_CANDIDATES.write_text(json.dumps(dict(pools), indent=2, ensure_ascii=False))

    http.close()

    OUT_CANDIDATES.write_text(json.dumps(dict(pools), indent=2, ensure_ascii=False))
    print(f"\nPool sizes: {[(sf, len(v)) for sf, v in pools.items()]}")

    # PHASE 2: 6-landmark calibration
    print("\n=== Phase 2: 6-landmark calibration to verify T7 ===", flush=True)
    replacements = {}
    failed_subfields = []
    for subfield, dropped_pids in DROPPED_BY_SUBFIELD.items():
        n_needed = len(dropped_pids)
        cands = pools[subfield]
        kept = []
        rejected = []
        for c in cands:
            if len(kept) >= n_needed: break
            q = PROBE_TEMPLATE.format(name=c["name"])
            cal = calibrate_with_landmarks(q, c["primary_subfield"])
            c["landmark_results"] = cal["per_landmark"]
            c["assigned_tier"] = cal["tier"]
            tag = "✓ T7" if cal["tier"] == "T7" else f"× {cal['tier']}"
            print(f"  {tag}: {subfield} | {c['name']} (h={c['h_index']}, cited={c['cited_by_count']})", flush=True)
            if cal["tier"] == "T7":
                kept.append(c)
            else:
                rejected.append(c)
        for pid, cand in zip(dropped_pids, kept):
            replacements[pid] = cand
        if len(kept) < n_needed:
            failed_subfields.append((subfield, n_needed, len(kept)))
            print(f"  WARN: {subfield} only got {len(kept)}/{n_needed}", flush=True)

    OUT_REPLACEMENTS.write_text(json.dumps(replacements, indent=2, ensure_ascii=False))
    print(f"\nFinal replacements: {len(replacements)}/16")
    for pid, c in replacements.items():
        print(f"  {pid} → {c['name']} ({c['primary_subfield']}, h={c['h_index']})")
    if failed_subfields:
        print(f"\nFailed subfields needing more candidates:")
        for sf, n, got in failed_subfields:
            print(f"  {sf}: {got}/{n}")


if __name__ == "__main__":
    main()
