"""Apply T7 substitutions to the probe set + enriched gold.

Reads:
  data/probes/t7_replacements.json — map from old probe_id → new researcher candidate
  data/probes/final_probe_set_v9.json
  data/probes/researcher_gold_enriched.json

Writes (in place):
  data/probes/final_probe_set_v9.json — replaces the probe question for each
                                          dropped probe ID
  data/probes/researcher_gold_enriched.json — replaces the enriched bundle for each
                                                dropped probe ID with the new researcher's
                                                manually-vetted info
"""

import json
from pathlib import Path

ROOT = Path("/Users/boj/ikp-paper")
REPL = ROOT / "data" / "probes" / "t7_replacements.json"
PROBES = ROOT / "data" / "probes" / "final_probe_set_v9.json"
ENRICHED = ROOT / "data" / "probes" / "researcher_gold_enriched.json"

NEW_TEMPLATE = (
    "In computer science, what is the research subfield of {name}, and name "
    "one paper, system, institution, or co-author associated with their work? "
    "If you don't know who this person is, say so."
)


def main():
    with open(REPL) as f:
        replacements = json.load(f)
    with open(PROBES) as f:
        probes = json.load(f)
    with open(ENRICHED) as f:
        enriched = json.load(f)
    enriched_by_pid = {r["probe_id"]: r for r in enriched}

    n_swap = 0
    for p in probes:
        pid = p["id"]
        if pid in replacements:
            new = replacements[pid]
            new_name = new["name"]
            p["question_v1"] = p.get("question_v1") or p["question"]
            p["question"] = NEW_TEMPLATE.format(name=new_name)
            p["researcher_name"] = new_name
            p["answer"] = new["primary_subfield"]
            p["_replaced_via_t7_substitution"] = True
            p["_replaced_old_name"] = enriched_by_pid.get(pid, {}).get("name")
            n_swap += 1

    # Update enriched bundles
    for pid, new in replacements.items():
        e = enriched_by_pid.get(pid)
        if e is None:
            continue
        e["name"] = new["name"]
        e["openalex_id"] = new["openalex_id"]
        e["primary_subfield"] = new["primary_subfield"]
        e["secondary_subfields"] = []
        e["affiliations"] = new.get("affiliations", [])
        e["named_systems"] = []
        e["venues"] = new.get("top_venues", [])[:6]
        e["co_authors"] = new.get("co_authors", [])
        e["top_works"] = [{"title": t} for t in new.get("top_titles", [])[:6]]
        e["topics"] = [{"name": t} for t in new.get("topics", [])[:3]]
        e["cs_venue_count"] = new.get("cs_venue_count", 0)
        e["noncs_venue_count"] = 0
        e["collision_flag"] = False
        e["collision_reason"] = None
        e["no_cs_match"] = False  # this is now a real CS researcher
        if "expected_verdict" in e: del e["expected_verdict"]
        e["manually_verified"] = True
        e["manual_notes"] = (f"T7 substitution from "
                             f"{e.get('_old_name','dropped probe')} to {new['name']}; "
                             f"verified T7 via 6-landmark calibration "
                             f"(no landmark answered correctly).")
        e["_t7_substitution"] = True

    with open(PROBES, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    with open(ENRICHED, "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"Substituted {n_swap}/{len(replacements)} probes")
    print(f"Wrote {PROBES} and {ENRICHED}")


if __name__ == "__main__":
    main()
