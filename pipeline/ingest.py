"""Ingest probes from existing data files into the store."""

import json
from pathlib import Path
from .store import ProbeStore

PROJECT_ROOT = Path(__file__).parent.parent


def ingest_llm_probes(store: ProbeStore):
    """Ingest LLM-generated probes from calibration batches."""
    files = [
        "llm_probes_calibrated.json",
        "llm_probes_batch2_calibrated.json",
        "llm_probes_batch3_calibrated.json",
        "llm_probes_for_calibration.json",
        "llm_probes_manual.json",
    ]
    total = 0
    for fname in files:
        fpath = PROJECT_ROOT / "data" / "probes" / fname
        if not fpath.exists():
            continue
        probes = json.load(open(fpath))
        added = 0
        for p in probes:
            if not isinstance(p, dict):
                continue
            q = p.get("question", "").strip()
            a = p.get("answer", "").strip()
            if q and a:
                pid = store.add(q, a, "llm",
                    domain=p.get("domain", p.get("category", "general")))
                if pid:
                    added += 1
        total += added
        print(f"  {fname}: +{added} new")
    print(f"  LLM total new: {total}")


def ingest_researcher_probes(store: ProbeStore):
    """Ingest researcher field probes."""
    files = ["researcher_field_probes_v3.json", "researcher_field_probes_v4.json"]
    total = 0
    for fname in files:
        fpath = PROJECT_ROOT / "data" / "probes" / fname
        if not fpath.exists():
            continue
        probes = json.load(open(fpath))
        added = 0
        for p in probes:
            q = (p.get("question_direct") or p.get("question", "")).strip()
            a = p.get("answer", "").strip()
            if q and a:
                pid = store.add(q, a, "researcher",
                    researcher_name=p.get("researcher_name", ""),
                    citation_count=p.get("citation_count", 0),
                    question_fill_blank=p.get("question_fill_blank", ""),
                    domain="computer_science")
                if pid:
                    added += 1
        total += added
        print(f"  {fname}: +{added} new")
    print(f"  Researcher total new: {total}")


def ingest_wikidata_probes(store: ProbeStore):
    """Ingest Wikidata probes."""
    files = ["wikidata_diverse_probes.json", "wikidata_probes.json"]
    total = 0
    for fname in files:
        fpath = PROJECT_ROOT / "data" / "probes" / fname
        if not fpath.exists():
            continue
        probes = json.load(open(fpath))
        added = 0
        for p in probes:
            q = (p.get("question_direct") or p.get("question", "")).strip()
            a = p.get("answer", "").strip()
            if q and a:
                pid = store.add(q, a, "wikidata",
                    wikidata_id=p.get("wikidata_id", p.get("entity_id", "")),
                    sitelink_count=p.get("sitelinks", p.get("sitelink_count", 0)),
                    entity_type=p.get("entity_type", ""),
                    domain=p.get("domain", p.get("entity_type", "general")))
                if pid:
                    added += 1
        total += added
        print(f"  {fname}: +{added} new")
    print(f"  Wikidata total new: {total}")


def ingest_all(store: ProbeStore):
    """Ingest all existing probe sources."""
    print("Ingesting probes...")
    ingest_llm_probes(store)
    ingest_researcher_probes(store)
    ingest_wikidata_probes(store)
    store.save()
    print(f"\nStore now has {len(store.probes)} probes")
