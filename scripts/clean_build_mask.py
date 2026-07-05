#!/usr/bin/env python3
"""Combine probe-quality flags into a single clean-probe mask.

Sources:
  - researcher_collisions.json  (OpenAlex name collisions)
  - wikidata_ambiguity.json     (same-type same-label ambiguity, ungrounded)
  - manual known-wrong answers  (from the LessWrong critique)
Junk Wikidata "collisions" (fan-wiki / fictional / media homonyms) are dropped
so real ambiguities (two journals named 'Analysis') are kept but 'capital of
Norway' (vs a Star Trek entity) is not.
"""
import json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
probes = json.load(open(ROOT / "data" / "probes" / "final_probe_set_v8.json"))
rc = json.load(open(ROOT / "data" / "probes" / "researcher_collisions.json"))
wa = json.load(open(ROOT / "data" / "probes" / "wikidata_ambiguity.json"))

JUNK = re.compile(r"star trek|as depicted|fictional|video game|\bsong\b|\balbum\b|\bfilm\b|"
                  r"\bnovel\b|comic|manga|episode|character|band\b", re.I)

flagged = {}   # probe_id -> reason

# researcher collisions
for r in rc:
    if r.get("collision"):
        flagged[r["probe_id"]] = f"researcher name-collision (x{r['n_collisions']}, cites {r.get('collision_cites')})"

# wikidata: keep only if a NON-junk same-type collision exists
for r in wa:
    if not r.get("ambiguous"):
        continue
    real = [d for d in (r.get("collision_desc") or []) if not JUNK.search(d or "")]
    if real:
        flagged[r["probe_id"]] = f"wikidata label-ambiguity ({r['label']!r}: {real[:2]})"

# known manual-wrong from the critique (highest peak in Bangladesh: disputed/updated)
for x in probes:
    if re.search(r"highest (peak|point|mountain).*bangladesh", x["question"], re.I):
        flagged[x["id"]] = "manual known-wrong/disputed answer (critique: Bangladesh peak)"

clean_ids = [x["id"] for x in probes if x["id"] not in flagged]
mask = {"n_total": len(probes), "n_flagged": len(flagged), "n_clean": len(clean_ids),
        "flagged": flagged, "clean_ids": clean_ids}
(ROOT / "data" / "probes" / "clean_mask.json").write_text(json.dumps(mask, indent=2, ensure_ascii=False))

byid = {x["id"]: x for x in probes}
print(f"Flagged {len(flagged)}/{len(probes)} ({100*len(flagged)/len(probes):.1f}%); clean set = {len(clean_ids)}")
print("flagged by tier:", dict(Counter(byid[i]["tier"] for i in flagged)))
print("flagged by source:", dict(Counter(byid[i]["source_type"] for i in flagged)))
print("clean by tier:", dict(Counter(byid[i]["tier"] for i in clean_ids)))
print("\nsample flags:")
for i, r in list(flagged.items())[:10]:
    print(f"  {i} [{byid[i]['tier']}] {r}")
