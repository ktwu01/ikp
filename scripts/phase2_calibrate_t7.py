"""Phase 2 only: load t7_candidates.json and run 6-landmark calibration.

For each candidate in the pool, build the probe question, query the 6 landmarks
in parallel, and accept the candidate as T7 ONLY IF no landmark answers correctly.
Stops when n_needed T7 candidates are accepted per subfield.

Reads:
  data/probes/t7_candidates.json (per-subfield candidate pool)
Writes:
  data/probes/t7_replacements.json (final approved substitutions)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.landmarks import LANDMARKS, query_landmark
from pipeline.judge import judge

CANDIDATES = ROOT / "data" / "probes" / "t7_candidates.json"
REPLACEMENTS = ROOT / "data" / "probes" / "t7_replacements.json"

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

PROBE_TEMPLATE = (
    "In computer science, what is the research subfield of {name}, and name "
    "one paper, system, institution, or co-author associated with their work? "
    "If you don't know who this person is, say so."
)


def calibrate(question: str, gold_subfield: str) -> dict:
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
            out["per_landmark"][lm["name"]] = {"correct": correct,
                                                "response": (resp or "")[:200]}
    tier = "T7"
    for lm in LANDMARKS:
        if out["per_landmark"][lm["name"]]["correct"]:
            tier = lm["tier"]; break
    out["tier"] = tier
    return out


def main():
    with open(CANDIDATES) as f:
        pools = json.load(f)
    existing = {}
    if REPLACEMENTS.exists():
        existing = json.load(open(REPLACEMENTS))
    print(f"Loaded {sum(len(v) for v in pools.values())} candidates; "
          f"already have {len(existing)} confirmed replacements", flush=True)

    for subfield, dropped_pids in DROPPED_BY_SUBFIELD.items():
        n_needed = len(dropped_pids)
        already_filled = sum(1 for pid in dropped_pids if pid in existing)
        if already_filled >= n_needed:
            print(f"  {subfield}: already filled {already_filled}/{n_needed}, skipping", flush=True)
            continue
        cands = pools.get(subfield, [])
        if not cands:
            print(f"  {subfield}: NO CANDIDATES IN POOL, skipping (handle separately)", flush=True)
            continue

        print(f"\n  {subfield}: need {n_needed - already_filled} more T7 candidates", flush=True)
        unfilled_pids = [pid for pid in dropped_pids if pid not in existing]
        kept = []
        for c in cands:
            if len(kept) >= len(unfilled_pids): break
            q = PROBE_TEMPLATE.format(name=c["name"])
            cal = calibrate(q, c["primary_subfield"])
            c["landmark_results"] = cal["per_landmark"]
            c["assigned_tier"] = cal["tier"]
            tag = "✓ T7" if cal["tier"] == "T7" else f"× {cal['tier']}"
            print(f"    {tag}: {c['name']} (h={c['h_index']}, cited={c['cited_by_count']})", flush=True)
            if cal["tier"] == "T7":
                kept.append(c)
        for pid, cand in zip(unfilled_pids, kept):
            existing[pid] = cand

        # Save incrementally
        with open(REPLACEMENTS, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        if len(kept) < len(unfilled_pids):
            print(f"  WARN: {subfield} only got {len(kept)}/{len(unfilled_pids)}", flush=True)

    print(f"\nDone. {len(existing)}/16 substitutions confirmed.", flush=True)
    for pid, c in existing.items():
        print(f"  {pid} → {c['name']} ({c['primary_subfield']})", flush=True)


if __name__ == "__main__":
    main()
