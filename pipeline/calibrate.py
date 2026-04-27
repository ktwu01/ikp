"""Calibration runner — test each probe against ALL 6 landmarks at once.

Optimized for throughput:
  - 16 probes processed concurrently
  - Within each probe, all 6 landmarks queried in parallel
  - Each landmark: query model, then judge — 2 serial API calls per landmark
  - Total: 16 probes × 6 landmarks × 2 calls = up to 192 in-flight API calls
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .landmarks import LANDMARKS, query_landmark
from .judge import judge
from .store import ProbeStore, TIER_RANGES

logger = logging.getLogger(__name__)

# 16 probes × 6 landmarks = 96 concurrent landmark+judge pairs
CONCURRENT_PROBES = 16


def calibrate_on_landmark(question: str, answer: str, landmark: dict) -> tuple[str, bool, str]:
    """Query one landmark and judge. Returns (landmark_name, correct, response)."""
    response = query_landmark(landmark, question)
    correct = judge(question, answer, response) if response else False
    return landmark["name"], correct, (response or "")[:150]


def calibrate_probe(probe: dict) -> dict:
    """Run one probe through all 6 landmarks in parallel."""
    q = probe["question"]
    a = probe["answer"]

    # Query landmarks in parallel, skipping already calibrated
    results = {}
    to_query = []
    for lm in LANDMARKS:
        key = f"cal_{lm['name']}"
        if key in probe:
            results[lm["name"]] = (probe[key], probe.get(f"resp_{lm['name']}", ""))
        else:
            to_query.append(lm)

    if to_query:
        with ThreadPoolExecutor(max_workers=len(to_query)) as executor:
            futures = {executor.submit(calibrate_on_landmark, q, a, lm): lm for lm in to_query}
            for future in as_completed(futures):
                lm_name, correct, response = future.result()
                results[lm_name] = (correct, response)

    # Store calibration results
    correctness = []
    for lm in LANDMARKS:
        correct, response = results.get(lm["name"], (False, ""))
        probe[f"cal_{lm['name']}"] = correct
        probe[f"resp_{lm['name']}"] = response
        correctness.append(correct)

    # Check monotonicity
    seen_true = False
    monotonic = True
    for c in correctness:
        if c:
            seen_true = True
        elif seen_true:
            monotonic = False
            break

    if not monotonic:
        probe["status"] = "dropped"
        probe["drop_reason"] = "non-monotonic"
        probe["tier"] = None
        return probe

    # Assign tier
    tier = "T7"
    for correct, lm in zip(correctness, LANDMARKS):
        if correct:
            tier = lm["tier"]
            break
    probe["tier"] = tier

    # Range check
    allowed = TIER_RANGES.get(probe.get("source_type", "llm"), set())
    if tier not in allowed:
        probe["status"] = "dropped"
        probe["drop_reason"] = f"out_of_range (source={probe['source_type']}, tier={tier}, allowed={allowed})"
        probe["tier"] = None
        return probe

    probe["status"] = "valid"
    return probe


def run(store: ProbeStore, batch_size: int = 0):
    """Calibrate all pending probes with high parallelism."""
    pending = [p for p in store.probes.values() if p["status"] == "pending"]
    if not pending:
        print("No pending probes to calibrate.")
        store.print_status()
        return

    if batch_size > 0:
        pending = pending[:batch_size]

    print(f"Calibrating {len(pending)} probes against {len(LANDMARKS)} landmarks ({CONCURRENT_PROBES} probes concurrent)...", flush=True)
    print(f"{'#':>5s} {'Source':>12s} {'Tier':>5s} {'Status':>10s} {'Vector':>8s}  Question", flush=True)
    print("-" * 90, flush=True)

    valid = 0
    dropped = 0
    done = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_PROBES) as executor:
        futures = {executor.submit(calibrate_probe, p): p for p in pending}
        for future in as_completed(futures):
            probe = futures[future]
            try:
                future.result()
            except Exception as e:
                probe["status"] = "dropped"
                probe["drop_reason"] = f"error: {e}"

            vector = "".join("T" if probe.get(f"cal_{lm['name']}", False) else "F" for lm in LANDMARKS)
            tier = probe.get("tier") or "—"
            status = probe["status"]
            src = probe.get("source_type", "?")

            if status == "valid":
                valid += 1
            elif status == "dropped":
                dropped += 1

            done += 1
            print(f"{done:5d} {src:>12s} {tier:>5s} {status:>10s} {vector:>8s}  {probe['question'][:45]}", flush=True)

            if done % 200 == 0:
                store.save()
                print(f"  --- checkpoint: {valid} valid, {dropped} dropped out of {done} ---", flush=True)

    store.save()
    print(f"\nDone: {valid} valid, {dropped} dropped out of {len(pending)}", flush=True)
    store.print_status()
