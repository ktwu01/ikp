"""Probe store — JSON-backed storage with incremental updates.

Each probe lives in the store as a dict with:
  - question, answer, source_type, metadata
  - calibration results per landmark (added incrementally)
  - tier (assigned after calibration)
  - status: pending → calibrating → valid | dropped

The store is a single JSON file that can be loaded, queried, and updated.
"""

import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Optional

from .landmarks import LANDMARKS, TIER_NAMES

# Expected tier ranges per source type
TIER_RANGES = {
    "llm":        {"T1", "T2", "T3", "T4"},
    "researcher": {"T3", "T4", "T5", "T6", "T7"},
    "wikidata":   {"T3", "T4", "T5", "T6", "T7"},
}

# Target probe counts per tier per source type
# LLM: T1-T2=200 each, T3-T4=50 each (LLM generation is inefficient for T3+)
# Researcher: 50 for T3-T4, 100 for T5-T7
# Wikidata: 100 for T3-T4 (compensates for reduced LLM), 100 for T5-T7
# Totals: 200 per tier across all sources
TIER_TARGETS = {
    "llm":        {"T1": 200, "T2": 200, "T3": 50, "T4": 50},
    "researcher": {"T3": 50, "T4": 50, "T5": 100, "T6": 100, "T7": 100},
    "wikidata":   {"T3": 100, "T4": 100, "T5": 100, "T6": 100, "T7": 100},
}


def probe_id(question: str) -> str:
    """Deterministic ID from question text."""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()[:12]


class ProbeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.probes: dict[str, dict] = {}  # id -> probe
        if self.path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text())
        if isinstance(data, list):
            for p in data:
                pid = p.get("id") or probe_id(p["question"])
                p["id"] = pid
                self.probes[pid] = p
        elif isinstance(data, dict):
            self.probes = data

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(self.probes.values()), indent=2, ensure_ascii=False))

    def add(self, question: str, answer: str, source_type: str, **metadata) -> Optional[str]:
        """Add a probe. Returns its ID, or None if duplicate."""
        pid = probe_id(question)
        if pid in self.probes:
            return None
        self.probes[pid] = {
            "id": pid,
            "question": question.strip(),
            "answer": answer.strip(),
            "source_type": source_type,
            "status": "pending",
            "tier": None,
            **metadata,
        }
        return pid

    def add_batch(self, probes: list[dict]) -> int:
        """Add multiple probes. Returns count of new (non-duplicate) probes added."""
        added = 0
        for p in probes:
            q = p.get("question") or p.get("question_direct", "")
            a = p.get("answer", "")
            st = p.get("source_type", "llm")
            meta = {k: v for k, v in p.items() if k not in ("question", "question_direct", "answer", "source_type")}
            if q and a and self.add(q, a, st, **meta) is not None:
                added += 1
        return added

    def get_pending(self, landmark_name: str) -> list[dict]:
        """Get probes that haven't been tested on this landmark yet."""
        key = f"cal_{landmark_name}"
        return [p for p in self.probes.values() if key not in p and p["status"] != "dropped"]

    def set_calibration(self, probe_id: str, landmark_name: str, correct: bool, response: str = ""):
        """Record a calibration result for one probe on one landmark."""
        p = self.probes.get(probe_id)
        if p:
            p[f"cal_{landmark_name}"] = correct
            p[f"resp_{landmark_name}"] = response[:150]

    def assign_tiers(self):
        """Assign tiers and apply monotonicity + range filtering to all calibrated probes."""
        for p in self.probes.values():
            # Check if fully calibrated
            results = []
            fully_calibrated = True
            for lm in LANDMARKS:
                key = f"cal_{lm['name']}"
                if key not in p:
                    fully_calibrated = False
                    break
                results.append(p[key])

            if not fully_calibrated:
                continue

            # Monotonicity check
            if not self._is_monotonic(results):
                p["status"] = "dropped"
                p["drop_reason"] = "non-monotonic"
                p["tier"] = None
                continue

            # Assign tier
            tier = "T7"
            for i, (correct, lm) in enumerate(zip(results, LANDMARKS)):
                if correct:
                    tier = lm["tier"]
                    break
            p["tier"] = tier

            # Range check
            allowed = TIER_RANGES.get(p.get("source_type", "llm"), set(TIER_NAMES))
            if tier not in allowed:
                p["status"] = "dropped"
                p["drop_reason"] = f"out_of_range ({p['source_type']} should be {allowed})"
                p["tier"] = None
                continue

            p["status"] = "valid"

    @staticmethod
    def _is_monotonic(results: list[bool]) -> bool:
        seen_true = False
        for r in results:
            if r:
                seen_true = True
            elif seen_true:
                return False
        return True

    def status(self) -> dict:
        """Return a summary of the current store state."""
        total = len(self.probes)
        by_status = Counter(p["status"] for p in self.probes.values())
        by_source = Counter(p.get("source_type", "?") for p in self.probes.values())

        # Valid probes per tier per source
        valid = [p for p in self.probes.values() if p["status"] == "valid"]
        tier_source = {}
        for t in TIER_NAMES:
            tier_source[t] = Counter(p.get("source_type", "?") for p in valid if p["tier"] == t)

        # Gaps
        gaps = {}
        for src, targets in TIER_TARGETS.items():
            for tier, target in targets.items():
                have = tier_source.get(tier, Counter()).get(src, 0)
                gap = max(0, target - have)
                if gap > 0:
                    gaps[(src, tier)] = {"have": have, "target": target, "gap": gap}

        # Calibration progress per landmark
        cal_progress = {}
        for lm in LANDMARKS:
            key = f"cal_{lm['name']}"
            done = sum(1 for p in self.probes.values() if key in p)
            cal_progress[lm["name"]] = done

        return {
            "total": total,
            "by_status": dict(by_status),
            "by_source": dict(by_source),
            "tier_source": {t: dict(c) for t, c in tier_source.items()},
            "cal_progress": cal_progress,
            "gaps": gaps,
        }

    def print_status(self):
        """Print a human-readable status report."""
        s = self.status()
        print(f"\n{'='*70}")
        print(f"PROBE STORE: {s['total']} probes")
        print(f"{'='*70}")
        print(f"Status:  {s['by_status']}")
        print(f"Sources: {s['by_source']}")

        print(f"\nCalibration progress:")
        for lm in LANDMARKS:
            done = s["cal_progress"].get(lm["name"], 0)
            print(f"  {lm['name']:20s}: {done}/{s['total']}")

        print(f"\nValid probes per tier:")
        header = f"  {'Tier':<5s}"
        sources = sorted(set(p.get("source_type", "?") for p in self.probes.values()))
        for src in sources:
            header += f" {src:>12s}"
        header += f" {'Total':>8s}"
        print(header)

        for t in TIER_NAMES:
            row = f"  {t:<5s}"
            total = 0
            for src in sources:
                n = s["tier_source"].get(t, {}).get(src, 0)
                total += n
                # Mark with * if below target
                target = TIER_TARGETS.get(src, {}).get(t, 0)
                marker = "*" if target > 0 and n < target else " "
                row += f" {n:>11d}{marker}"
            row += f" {total:>8d}"
            print(row)

        if s["gaps"]:
            print(f"\nGaps to fill:")
            for (src, tier), info in sorted(s["gaps"].items()):
                print(f"  {src:12s} {tier}: have {info['have']}, need {info['target']}, gap = {info['gap']}")
        else:
            print(f"\nAll targets met!")

    def export_valid(self, output_path: str | Path, tier_total: int = 200):
        """Export a balanced final dataset by sampling from the valid pool.

        Two-pass sampling:
        1. Sample up to per-source targets for each tier
        2. If a tier is under tier_total, fill the gap uniformly from
           remaining (unsampled) probes of other sources in that tier

        The full pool (pipeline_store.json) is never modified.
        """
        import random
        random.seed(42)  # reproducible sampling

        valid = [p for p in self.probes.values() if p["status"] == "valid"]

        # Group by (source_type, tier)
        from collections import defaultdict
        groups = defaultdict(list)
        for p in valid:
            groups[(p["source_type"], p["tier"])].append(p)

        sampled_ids = set()
        tier_sampled = defaultdict(list)  # tier -> list of probes

        # Pass 1: sample per-source targets
        print("Pass 1: Sample per-source targets")
        for tier in TIER_NAMES:
            for src, targets in TIER_TARGETS.items():
                target = targets.get(tier, 0)
                if target == 0:
                    continue
                pool = groups.get((src, tier), [])
                n = min(target, len(pool))
                chosen = random.sample(pool, n) if len(pool) > n else list(pool)
                tier_sampled[tier].extend(chosen)
                for p in chosen:
                    sampled_ids.add(p["id"])
                status = "OK" if n >= target else f"short ({n}/{target})"
                print(f"  {tier} {src:12s}: {n:4d}/{target} ({status})")

        # Pass 2: fill tiers under tier_total from remaining pool
        print(f"\nPass 2: Fill to {tier_total} per tier from remaining pool")
        for tier in TIER_NAMES:
            current = len(tier_sampled[tier])
            gap = tier_total - current
            if gap <= 0:
                print(f"  {tier}: {current}/{tier_total} (OK)")
                continue

            # Collect remaining unsampled probes for this tier across all sources
            remaining = []
            for src in TIER_TARGETS:
                for p in groups.get((src, tier), []):
                    if p["id"] not in sampled_ids:
                        remaining.append(p)

            fill = min(gap, len(remaining))
            chosen = random.sample(remaining, fill) if len(remaining) > fill else remaining
            tier_sampled[tier].extend(chosen)
            for p in chosen:
                sampled_ids.add(p["id"])
            print(f"  {tier}: {current} + {fill} fill = {current + fill}/{tier_total}"
                  f" ({'OK' if current + fill >= tier_total else f'short, only {len(remaining)} available'})")

        # Flatten
        sampled = []
        for tier in TIER_NAMES:
            sampled.extend(tier_sampled[tier])

        # Sort and assign sequential IDs
        sampled.sort(key=lambda p: (
            TIER_NAMES.index(p["tier"]) if p["tier"] in TIER_NAMES else 99,
            p["source_type"],
        ))

        final = []
        for i, p in enumerate(sampled):
            clean = {
                "id": f"IKP_{p['tier']}_{i:04d}",
                "question": p["question"],
                "answer": p["answer"],
                "tier": p["tier"],
                "source_type": p["source_type"],
                "domain": p.get("domain", "general"),
            }
            if p["source_type"] == "researcher":
                clean["researcher_name"] = p.get("researcher_name", "")
                clean["citation_count"] = p.get("citation_count", 0)
            elif p["source_type"] == "wikidata":
                clean["wikidata_id"] = p.get("wikidata_id", "")
                clean["sitelink_count"] = p.get("sitelink_count", 0)
                clean["entity_type"] = p.get("entity_type", "")
            final.append(clean)

        # Summary
        from collections import Counter
        tier_counts = Counter(p["tier"] for p in final)
        print(f"\nFinal dataset: {len(final)} probes")
        for t in TIER_NAMES:
            print(f"  {t}: {tier_counts.get(t, 0)}")

        Path(output_path).write_text(json.dumps(final, indent=2, ensure_ascii=False))
        print(f"\nSaved to {output_path}")
        print(f"(Full pool of {len(valid)} valid probes preserved in pipeline_store.json)")
