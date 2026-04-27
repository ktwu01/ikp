"""Refine collision detection — fewer false positives.

A probe is flagged as a collision only if BOTH:
  (a) The OpenAlex top-field is non-CS (chemistry, medicine, physics, etc.)
      OR there are zero CS venues AND venues clearly indicate non-CS
      (no occurrence of 'computer', 'computational', 'informatics', 'computing'
       in any of the top-10 venue names).
  (b) Top topics' field is not "Computer Science".

Reads:  data/probes/researcher_gold_enriched.json
Writes: data/probes/researcher_collisions_v2.json (final list)
        Also updates the in-place collision_flag/collision_reason fields.
"""

import json
from pathlib import Path
from collections import Counter

ROOT = Path("/Users/boj/ikp-paper")
SRC = ROOT / "data" / "probes" / "researcher_gold_enriched.json"
OUT_COLLISIONS = ROOT / "data" / "probes" / "researcher_collisions_v2.json"

CS_VENUE_TOKENS = ["computer", "computational", "computing", "informatics",
                   "acm ", "ieee transactions", "lecture notes in computer",
                   "lncs", "sigplan", "sigmod", "sigops", "sigcomm",
                   "vldb", "popl", "pldi", "ndss", "usenix",
                   "logical methods", "functional programming",
                   "designs codes and cryptography",
                   "theoretical computer science",
                   "formal methods", "automated reasoning",
                   "discrete mathematics & theoretical computer",
                   "ieee access",  # broad CS/EE catch-all
                   "journal of cryptology", "designs, codes",
                   "data mining and knowledge discovery"]

# Strong non-CS markers — if these appear in top venues, very likely a name collision
STRONG_NON_CS = ["chemical society", "angewandte chemie", "journal of the american chemical",
                 "nature chemistry", "nature physics", "nature medicine",
                 "physical review letters", "physical review b", "physical review a",
                 "astrophysical", "metrologia",
                 "journal of clinical", "lancet", "new england journal of medicine",
                 "world journal of surgery", "european journal of radiology",
                 "cell", "cell research", "nano letters",
                 "laboratory investigation", "plos one",
                 "current opinion in environmental",
                 "global environmental",
                 "wiley interdisciplinary reviews climate",
                 "journal of infection", "j infect",
                 "biochemistry", "biology", "biological",
                 "journal of agricultural", "agricultural",
                 "annals of"]


def is_cs_venue(v: str) -> bool:
    if not v: return False
    vl = v.lower()
    if any(tok in vl for tok in STRONG_NON_CS):
        return False
    return any(tok in vl for tok in CS_VENUE_TOKENS)


def is_strong_non_cs_venue(v: str) -> bool:
    if not v: return False
    return any(tok in v.lower() for tok in STRONG_NON_CS)


def has_cs_topic(topics: list) -> bool:
    for t in topics or []:
        f = (t.get("field") or "").lower()
        if "computer" in f or "informatics" in f:
            return True
    return False


def main():
    with open(SRC) as f:
        recs = json.load(f)

    new_collisions = []
    cs_topic_overrides = 0
    cs_venue_overrides = 0

    for r in recs:
        venues = r.get("venues", []) or []
        topics = r.get("topics", []) or []
        cs_v = sum(1 for v in venues if is_cs_venue(v))
        noncs_v = sum(1 for v in venues if is_strong_non_cs_venue(v))
        top_field = (topics[0].get("field", "") if topics else "").lower()

        # Collision if:
        # - top field is clearly non-CS (medicine, chemistry, etc.)
        # - AND no CS topic anywhere
        # - AND no CS-keyword venues in top 10
        is_noncs_top_field = bool(top_field) and any(k in top_field for k in
                                                      ("chem", "biol", "medic",
                                                       "phys", "geol", "agricul",
                                                       "law", "social", "econom", "psycho"))
        has_cs_t = has_cs_topic(topics)
        # Venue strength signal
        venue_signal_noncs = (noncs_v >= 2 and cs_v == 0) or (noncs_v >= 3 and noncs_v > cs_v)

        if is_noncs_top_field and not has_cs_t and not (cs_v >= 2):
            # Definite collision
            new_collisions.append({
                "probe_id": r["probe_id"],
                "name": r["name"],
                "reason": f"top OpenAlex field='{topics[0]['field']}'; no CS topics; cs_venues={cs_v}",
                "venues_sample": venues[:3],
                "openalex_id": r["openalex_id"],
            })
            r["collision_flag"] = True
            r["collision_reason"] = f"top field non-CS, no CS topic"
        elif venue_signal_noncs and not has_cs_t:
            new_collisions.append({
                "probe_id": r["probe_id"],
                "name": r["name"],
                "reason": f"venue signal: {noncs_v} non-CS, {cs_v} CS, no CS topic",
                "venues_sample": venues[:3],
                "openalex_id": r["openalex_id"],
            })
            r["collision_flag"] = True
            r["collision_reason"] = f"non-CS venues dominate, no CS topic"
        elif r.get("collision_flag"):
            # Was flagged before but now overruled by CS topic / CS venues
            if has_cs_t:
                cs_topic_overrides += 1
            if cs_v >= 2:
                cs_venue_overrides += 1
            r["collision_flag"] = False
            r["collision_reason"] = None

    # Save
    with open(SRC, "w") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    with open(OUT_COLLISIONS, "w") as f:
        json.dump(new_collisions, f, indent=2, ensure_ascii=False)

    # Per-tier
    by_tier = Counter()
    for c in new_collisions:
        parts = c["probe_id"].split("_")
        if len(parts) >= 2: by_tier[parts[1]] += 1

    print(f"After refinement:")
    print(f"  Collisions: {len(new_collisions)} (was 82)")
    print(f"  Reverted by CS topic match: {cs_topic_overrides}")
    print(f"  Reverted by ≥2 CS venues:   {cs_venue_overrides}")
    print(f"  Per tier: {dict(by_tier)}")
    print(f"\nSample remaining collisions:")
    for c in new_collisions[:10]:
        print(f"  [{c['probe_id']}] {c['name']} — {c['reason']}")


if __name__ == "__main__":
    main()
