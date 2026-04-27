#!/usr/bin/env python3
"""Generate diverse Wikidata probes using simple SPARQL queries.

No sitelink filtering — tiers are assigned by the landmark calibration pipeline.
Just pull diverse, unambiguous, time-stable facts from Wikidata.

Fact types:
  1. founding_year    — universities, museums, organizations
  2. birth_year       — historical figures (by occupation)
  3. capital          — countries → capital city
  4. currency         — countries → currency
  5. language         — countries → official language
  6. chemical_symbol  — elements → symbol
  7. atomic_number    — elements → atomic number
  8. director         — films → director
  9. composer         — operas/musical works → composer
  10. painter         — paintings → creator
  11. architect       — buildings → architect
  12. river_mouth     — rivers → flows into
  13. headquarters    — companies → HQ city
"""

import json
import time
import logging
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "IKP-Research/1.0 (research@example.org)"}


def run_sparql(query: str) -> list[dict]:
    with httpx.Client(timeout=90) as http:
        for attempt in range(3):
            try:
                r = http.get(SPARQL_URL, params={"query": query, "format": "json"}, headers=HEADERS)
                if r.status_code == 200:
                    return r.json().get("results", {}).get("bindings", [])
                if r.status_code in (429, 502, 503):
                    time.sleep(10 * (attempt + 1))
                    continue
                logger.warning(f"SPARQL {r.status_code}")
                return []
            except Exception as e:
                logger.warning(f"SPARQL error: {e}")
                time.sleep(5)
    return []


# (fact_type, question_template, sparql, answer_field_name)
QUERIES = [
    ("founding_year", "In what year was {label} founded?", """
SELECT ?item ?itemLabel (YEAR(?d) AS ?answer) WHERE {
  ?item wdt:P31 wd:Q3918 . ?item wdt:P571 ?d .
  FILTER(YEAR(?d) > 1000 && YEAR(?d) < 2020)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("founding_year_museum", "In what year was {label} founded?", """
SELECT ?item ?itemLabel (YEAR(?d) AS ?answer) WHERE {
  ?item wdt:P31/wdt:P279* wd:Q33506 . ?item wdt:P571 ?d .
  FILTER(YEAR(?d) > 1500 && YEAR(?d) < 2020)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("founding_year_org", "In what year was {label} founded?", """
SELECT ?item ?itemLabel (YEAR(?d) AS ?answer) WHERE {
  ?item wdt:P31 wd:Q43229 . ?item wdt:P571 ?d .
  FILTER(YEAR(?d) > 1800 && YEAR(?d) < 2020)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("capital", "What is the capital of {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q6256 . ?item wdt:P36 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} LIMIT __N__"""),

    ("currency", "What is the currency of {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q6256 . ?item wdt:P38 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} LIMIT __N__"""),

    ("official_language", "What is the official language of {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q6256 . ?item wdt:P37 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} LIMIT __N__"""),

    ("chemical_symbol", "What is the chemical symbol for {label}?", """
SELECT ?item ?itemLabel ?answer WHERE {
  ?item wdt:P31 wd:Q11344 . ?item wdt:P246 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} LIMIT __N__"""),

    ("atomic_number", "What is the atomic number of {label}?", """
SELECT ?item ?itemLabel ?answer WHERE {
  ?item wdt:P31 wd:Q11344 . ?item wdt:P1086 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} LIMIT __N__"""),

    ("film_director", "Who directed the film {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q11424 . ?item wdt:P57 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("opera_composer", "Who composed {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q1344 . ?item wdt:P86 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("painting_creator", "Who painted {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q3305213 . ?item wdt:P170 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("building_architect", "Who designed {label}?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31/wdt:P279* wd:Q41176 . ?item wdt:P84 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),

    ("river_mouth", "What body of water does the {label} flow into?", """
SELECT ?item ?itemLabel ?answerLabel WHERE {
  ?item wdt:P31 wd:Q4022 . ?item wdt:P403 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} ORDER BY MD5(STR(?item)) LIMIT __N__"""),
]


def generate_all(n_per_type: int = 200) -> list[dict]:
    all_probes = []

    for fact_type, question_tpl, sparql_tpl in QUERIES:
        query = sparql_tpl.replace("__N__", str(n_per_type))
        logger.info(f"Querying {fact_type}...")
        rows = run_sparql(query)

        probes = []
        seen = set()
        for row in rows:
            label = row.get("itemLabel", {}).get("value", "")
            wikidata_id = row.get("item", {}).get("value", "").split("/")[-1]

            # Get answer — either direct value or labeled entity
            if "answerLabel" in row:
                answer = row["answerLabel"]["value"]
            elif "answer" in row:
                answer = str(row["answer"]["value"])
            else:
                continue

            # Skip if no English label (shows as Q-number)
            if not label or label == wikidata_id:
                continue
            if not answer or answer.startswith("Q"):
                continue

            # Deduplicate within fact type
            key = (label.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)

            question = question_tpl.format(label=label)
            probes.append({
                "question": question,
                "answer": answer,
                "source_type": "wikidata",
                "fact_type": fact_type,
                "wikidata_id": wikidata_id,
                "domain": fact_type,
            })

        logger.info(f"  {fact_type}: {len(probes)} probes")
        all_probes.extend(probes)
        time.sleep(3)

    logger.info(f"\nTotal: {len(all_probes)} probes across {len(QUERIES)} fact types")
    return all_probes


if __name__ == "__main__":
    probes = generate_all(n_per_type=200)
    output = Path(__file__).parent.parent / "data" / "probes" / "wikidata_diverse_v2.json"
    output.write_text(json.dumps(probes, indent=2, ensure_ascii=False))
    logger.info(f"Saved to {output}")

    from collections import Counter
    types = Counter(p["fact_type"] for p in probes)
    for t, n in types.most_common():
        logger.info(f"  {t}: {n}")
