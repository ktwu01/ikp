#!/usr/bin/env python3
"""IKP Pipeline CLI.

Usage:
  python -m pipeline.run status              Show current store status
  python -m pipeline.run ingest              Ingest probes from existing data files
  python -m pipeline.run calibrate           Calibrate all pending probes (all 6 landmarks per probe)
  python -m pipeline.run calibrate 50        Calibrate a batch of 50 pending probes
  python -m pipeline.run export              Export valid probes as final dataset
  python -m pipeline.run gaps                Show what needs to be filled
  python -m pipeline.run gen-wikidata        Generate new diverse Wikidata probes and ingest
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress noisy httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Ensure pipeline is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.store import ProbeStore
from pipeline.landmarks import LANDMARKS

STORE_PATH = Path(__file__).parent.parent / "data" / "pipeline_store.json"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    store = ProbeStore(STORE_PATH)

    if cmd == "status":
        store.print_status()

    elif cmd == "ingest":
        from pipeline.ingest import ingest_all
        ingest_all(store)
        store.print_status()

    elif cmd == "calibrate":
        from pipeline.calibrate import run
        batch_size = int(args[1]) if len(args) > 1 else 0
        run(store, batch_size=batch_size)

    elif cmd == "gen-wikidata":
        from pipeline.generate_wikidata import generate_all
        probes = generate_all(n_per_type=200)
        added = store.add_batch(probes)
        store.save()
        print(f"Added {added} new Wikidata probes")
        store.print_status()

    elif cmd == "export":
        output = Path(__file__).parent.parent / "data" / "probes" / "final_probe_set_v7.json"
        store.export_valid(output)

    elif cmd == "gaps":
        s = store.status()
        if not s["gaps"]:
            print("All targets met!")
        else:
            print("Gaps to fill:")
            for (src, tier), info in sorted(s["gaps"].items()):
                print(f"  {src:12s} {tier}: have {info['have']}, need {info['target']}, gap = {info['gap']}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
