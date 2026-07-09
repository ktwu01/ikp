#!/usr/bin/env python3
"""Generate a deterministic public/private split of the IKP probe set.

IKP v2 defense against *contamination* (an operator memorizing the public
probe answers to look larger): score on a held-out **private** split that
was never published. This script deterministically partitions the 1,400
probes into a `public` half (for demos, tutorials, the interactive CLI) and
a `private` half (for adjudicated estimates), balanced within every tier so
both halves are representative.

The split is a pure function of each probe's stable `id` plus a fixed salt,
so it is reproducible anywhere and needs no stored RNG state. Rotating the
salt (``--salt``) yields a fresh, non-overlapping partition when a private
split is suspected of having leaked.

Usage:
  python scripts/make_probe_split.py                 # writes split_manifest_v2.json
  python scripts/make_probe_split.py --public-frac 0.5 --salt ikp-v2-2026
  python scripts/make_probe_split.py --check         # verify balance, no write
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROBE_FILE = PROJECT_ROOT / "data" / "probes" / "final_probe_set_v8.json"
OUT_FILE = PROJECT_ROOT / "data" / "probes" / "split_manifest_v2.json"
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def bucket(probe_id: str, salt: str) -> float:
    """Map an id to a stable float in [0, 1) via SHA-256(salt|id)."""
    h = hashlib.sha256(f"{salt}|{probe_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def build_split(probes, public_frac: float, salt: str):
    """Assign each probe to 'public' or 'private', balanced per tier.

    Within each tier we sort by hash and take the lowest `public_frac`
    fraction as public. This keeps the split exactly balanced per tier
    (not just in expectation) while staying a deterministic function of id.
    """
    by_tier = defaultdict(list)
    for p in probes:
        by_tier[p["tier"]].append(p)

    assignment = {}
    for tier, items in by_tier.items():
        items_sorted = sorted(items, key=lambda p: bucket(p["id"], salt))
        n_public = round(len(items_sorted) * public_frac)
        for i, p in enumerate(items_sorted):
            assignment[p["id"]] = "public" if i < n_public else "private"
    return assignment


def summarize(probes, assignment):
    per_tier = defaultdict(Counter)
    for p in probes:
        per_tier[p["tier"]][assignment[p["id"]]] += 1
    print(f"\n  {'Tier':<5} {'public':>7} {'private':>8} {'total':>6}")
    print(f"  {'─' * 30}")
    tot = Counter()
    for t in TIERS:
        c = per_tier[t]
        tot += c
        print(f"  {t:<5} {c['public']:>7} {c['private']:>8} "
              f"{c['public'] + c['private']:>6}")
    print(f"  {'─' * 30}")
    print(f"  {'all':<5} {tot['public']:>7} {tot['private']:>8} "
          f"{tot['public'] + tot['private']:>6}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public-frac", type=float, default=0.5,
                    help="Fraction of each tier assigned to the public split (default 0.5)")
    ap.add_argument("--salt", default="ikp-v2",
                    help="Salt for the hash; rotate to regenerate a fresh split")
    ap.add_argument("--check", action="store_true",
                    help="Print the balance summary without writing the manifest")
    args = ap.parse_args()

    probes = json.load(open(PROBE_FILE))
    assignment = build_split(probes, args.public_frac, args.salt)
    summarize(probes, assignment)

    if args.check:
        print("  (--check: manifest not written)\n")
        return

    manifest = {
        "version": "v2",
        "salt": args.salt,
        "public_frac": args.public_frac,
        "n_probes": len(probes),
        "assignment": assignment,
    }
    OUT_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote {OUT_FILE.relative_to(PROJECT_ROOT)} "
          f"({sum(1 for v in assignment.values() if v == 'private')} private probes)\n")


if __name__ == "__main__":
    main()
