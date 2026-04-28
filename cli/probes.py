"""Probe loading helpers (defaults to the latest available probe set)."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROBE_DIR = PROJECT_ROOT / "data" / "probes"


def load_probes() -> list:
    for name in ("final_probe_set_v9.json", "final_probe_set_v8.json"):
        p = PROBE_DIR / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"No probe set found in {PROBE_DIR}")


def find_probe_by_id(probe_id: str) -> dict | None:
    pid = probe_id.strip().upper()
    for p in load_probes():
        if p["id"].upper() == pid:
            return p
    return None


def find_probes_for_researcher(name: str) -> list:
    needle = name.strip().lower()
    return [
        p for p in load_probes()
        if p.get("researcher_name") and needle in p["researcher_name"].lower()
    ]
