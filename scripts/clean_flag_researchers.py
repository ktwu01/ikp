#!/usr/bin/env python3
"""Flag researcher probes whose name collides with another distinct researcher.

A probe is flagged AMBIGUOUS if OpenAlex has >=2 distinct authors with the exact
display name and >=100 citations at different institutions (so the model cannot
know which person the question means). Uses the intended openalex_id stored in
data/researcher_citations.json to distinguish the target from collisions and to
avoid counting OpenAlex author-splits of the same person (same institution).
"""
import json, os, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
CIT = json.load(open(ROOT / "data" / "researcher_citations.json"))
KEY = os.environ.get("OPENALEX_API_KEY", "")
OUT = ROOT / "data" / "probes" / "researcher_collisions.json"
MIN_CITES = 100

def norm_inst(a):
    inst = (a.get("last_known_institutions") or [{}])
    return (inst[0].get("display_name") or "").lower().strip() if inst else ""

def check(entry):
    name = entry["name"]; intended = entry.get("openalex_id","")
    params = {"search": name, "per-page": 50}
    if KEY: params["api_key"] = KEY
    for attempt in range(3):
        try:
            r = httpx.get("https://api.openalex.org/authors", params=params, timeout=30)
            if r.status_code == 200: break
            import time; time.sleep(2*(attempt+1))
        except Exception:
            import time; time.sleep(2)
    else:
        return {**entry, "collision": None, "note": "api_error"}
    res = r.json().get("results", [])
    exact = [a for a in res
             if a.get("display_name","").lower().strip() == name.lower().strip()
             and a.get("cited_by_count",0) >= MIN_CITES]
    intended_inst = ""
    others = []
    for a in exact:
        if a["id"] == intended or a["id"].split("/")[-1] == intended.split("/")[-1]:
            intended_inst = norm_inst(a); continue
        others.append(a)
    # keep collisions at a DIFFERENT institution than the target (avoid same-person splits)
    true_coll = [a for a in others if norm_inst(a) and norm_inst(a) != intended_inst]
    return {
        "probe_id": entry["probe_id"], "name": name, "tier": entry["tier"],
        "intended_cites": entry.get("cited_by_count"),
        "n_exact_ge100": len(exact),
        "n_collisions": len(true_coll),
        "collision": len(true_coll) >= 1,
        "collision_insts": [norm_inst(a) for a in true_coll][:4],
        "collision_cites": [a.get("cited_by_count") for a in true_coll][:4],
    }

results = []; lock = threading.Lock(); done = [0]
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = [ex.submit(check, e) for e in CIT]
    for f in as_completed(futs):
        r = f.result()
        with lock:
            results.append(r); done[0]+=1
            if done[0] % 50 == 0: print(f"  {done[0]}/{len(CIT)}", flush=True)

results.sort(key=lambda x: x["probe_id"])
json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
flagged = [r for r in results if r.get("collision")]
errs = [r for r in results if r.get("collision") is None]
print(f"\n{len(flagged)}/{len(results)} researcher probes flagged as name-collisions "
      f"({100*len(flagged)/len(results):.1f}%); {len(errs)} api errors")
print("examples:")
for r in flagged[:8]:
    print(f"  {r['probe_id']} {r['name']} (cites={r['intended_cites']}) vs {r['n_collisions']} other(s) {r['collision_cites']}")
print(f"wrote {OUT}")
