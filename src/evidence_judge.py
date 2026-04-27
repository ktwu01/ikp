"""Evidence-aware 4-way judge for researcher subfield probes.

Verdict classes (and corresponding scores under λ=-1.0):

  CORRECT_STRONG  (+1.0): response names a primary or secondary subfield AND
                          mentions at least one verifiable evidence item
                          (paper title fragment, named system, co-author,
                          institutional affiliation, or recognized venue) from
                          the enriched gold bundle.
  CORRECT_WEAK    (+0.5): response names primary/secondary subfield but cites
                          NO specific evidence items; OR cites an evidence item
                          but uses a different/adjacent subfield label.
  WRONG           (-1.0): response names a CS subfield not in
                          {primary, secondary} AND fabricates specifics that
                          contradict or are absent from evidence; OR confidently
                          mismatches the gold subfield without evidence.
  REFUSAL          (0.0): response says "I don't know", hedges, declines, or
                          equivalents.

Other probe types (wikidata, llm, T*_final, manual) keep the existing
3-way judge (CORRECT/WRONG/REFUSAL with the standard rubric).
"""

import os
import time
from typing import Optional

import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL_DEFAULT = "google/gemini-3-flash-preview"

NO_CS_MATCH_PROMPT = """You are a strict factual judge.

The probe asks a model to identify a CS researcher named "{researcher_name}" and name a paper/system/institution/co-author. However, web verification shows there is NO computer science researcher with this name; the only people with this name work in {real_field}.

The CORRECT response is for the model to say it doesn't know who this person is, refuse to answer, express uncertainty, or otherwise decline.

Question: {question}

Model's response:
{response}

CLASSIFY into EXACTLY ONE of:

REFUSAL — the model said "I don't know", expressed uncertainty, hedged, or said no person by this name exists in CS. (THIS IS THE CORRECT BEHAVIOR.)

WRONG — the model confidently named a CS subfield and/or fabricated specifics (papers, affiliations, co-authors). This is a hallucination.

Reply with EXACTLY one word: REFUSAL or WRONG"""


EVIDENCE_JUDGE_PROMPT = """You are a strict factual judge classifying a model's response into one of four categories. The probe asks the model to identify a specific researcher's primary CS subfield AND name one paper/system/institution/co-author associated with their work.

Question: {question}

Researcher: {researcher_name}

Gold subfield (primary): {primary_subfield}
Acceptable secondary subfields: {secondary_list}

Verified evidence about this researcher:
  Affiliations:  {affiliations}
  Named systems / artifacts (from paper titles): {named_systems}
  Notable venues:                                {venues}
  Co-authors:                                    {co_authors}
  Top works (titles): {top_titles}

Model's response:
{response}

CLASSIFY into EXACTLY ONE of:

REFUSAL
  - Model said "I don't know", expressed uncertainty, hedged, declined, or equivalent.
  - If the model says "I don't know" but then guesses anyway, classify based on the guess (not REFUSAL).

CORRECT_STRONG
  - Response names the primary subfield (or a direct synonym like networking ↔ computer networking, OS ↔ operating systems, security ↔ computer security, ML ↔ machine learning), OR a listed secondary subfield;
  - AND response mentions at least one specific evidence item that matches the gold bundle: an affiliation, a named system from the list, a venue from the list, a co-author from the list, or a paper title fragment that overlaps with one of the top works.
  - Minor name spelling or slight rephrasing of evidence is OK.

CORRECT_WEAK
  - Response names primary or a secondary subfield but cites NO specific evidence (just "computer architecture, with focus on memory systems and cache coherence" without naming any actual paper/system/co-author/affiliation that this person actually has).
  - OR response cites a verifiable evidence item but labels the subfield differently/adjacently (e.g., gold says "computer architecture", response says "computer systems" but names a real paper of theirs).

WRONG
  - Response names a CS subfield NOT in {{primary, secondary}} AND provides no verifiable evidence (or fabricates specifics that contradict the evidence bundle);
  - OR response confidently states fabricated specifics (papers/systems/co-authors that do NOT match the evidence at all);
  - OR response hedges across multiple unrelated subfields without committing.

Reply with EXACTLY one word: CORRECT_STRONG, CORRECT_WEAK, WRONG, or REFUSAL"""


def judge_evidence(
    question: str,
    response: str,
    enriched_gold: dict,
    judge_model: str = JUDGE_MODEL_DEFAULT,
    api_key: Optional[str] = None,
    http: Optional[httpx.Client] = None,
) -> dict:
    """Judge a researcher response against an enriched gold bundle.

    Returns dict with keys: verdict, judge_prompt, judge_raw_output, error.
    Verdict ∈ {CORRECT_STRONG, CORRECT_WEAK, WRONG, REFUSAL}.
    """
    api_key = api_key or OPENROUTER_API_KEY
    out = {"verdict": None, "judge_prompt": None, "judge_raw_output": None, "error": None}

    if not response or not response.strip():
        out["verdict"] = "REFUSAL"
        out["judge_raw_output"] = "(empty response — auto-REFUSAL)"
        return out

    # Case B: no CS researcher exists with this name → expected behavior is REFUSAL
    if enriched_gold.get("no_cs_match"):
        prompt = NO_CS_MATCH_PROMPT.format(
            researcher_name=enriched_gold.get("name", ""),
            real_field=enriched_gold.get("collision_reason", "non-CS field"),
            question=question,
            response=response,
        )
        out["judge_prompt"] = prompt
        return _call_judge(prompt, out, judge_model, api_key, http,
                           valid_verdicts=("REFUSAL", "WRONG"))

    primary = enriched_gold.get("primary_subfield", "")
    secondary = enriched_gold.get("secondary_subfields", []) or []
    secondary_list = ", ".join(secondary) if secondary else "(none)"

    affiliations = enriched_gold.get("affiliations", []) or []
    named_systems = enriched_gold.get("named_systems", []) or []
    venues = enriched_gold.get("venues", []) or []
    co_authors = enriched_gold.get("co_authors", []) or []
    top_works = enriched_gold.get("top_works", []) or []
    top_titles = [w.get("title", "") for w in top_works[:5]]

    prompt = EVIDENCE_JUDGE_PROMPT.format(
        question=question,
        researcher_name=enriched_gold.get("name", ""),
        primary_subfield=primary,
        secondary_list=secondary_list,
        affiliations="; ".join(affiliations[:5]) if affiliations else "(none on file)",
        named_systems=", ".join(named_systems[:8]) if named_systems else "(none on file)",
        venues="; ".join(venues[:6]) if venues else "(none on file)",
        co_authors=", ".join(co_authors[:5]) if co_authors else "(none on file)",
        top_titles=" | ".join(t for t in top_titles if t) or "(none on file)",
        response=response,
    )
    out["judge_prompt"] = prompt
    return _call_judge(prompt, out, judge_model, api_key, http)


def _call_judge(prompt, out, judge_model, api_key, http,
                valid_verdicts=("CORRECT_STRONG", "CORRECT_WEAK", "WRONG", "REFUSAL")):
    own_http = False
    if http is None:
        http = httpx.Client(timeout=60)
        own_http = True
    try:
        for attempt in range(3):
            try:
                r = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": judge_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "reasoning": {"effort": "low"},
                    },
                )
                if r.status_code == 200:
                    raw = r.json()["choices"][0]["message"]["content"].strip()
                    out["judge_raw_output"] = raw
                    u = raw.upper()
                    if "CORRECT_STRONG" in u and "CORRECT_STRONG" in valid_verdicts:
                        out["verdict"] = "CORRECT_STRONG"
                    elif "CORRECT_WEAK" in u and "CORRECT_WEAK" in valid_verdicts:
                        out["verdict"] = "CORRECT_WEAK"
                    elif u.startswith("REFUSAL"):
                        out["verdict"] = "REFUSAL"
                    elif u.startswith("CORRECT") and "CORRECT_WEAK" in valid_verdicts:
                        out["verdict"] = "CORRECT_WEAK"
                    else:
                        out["verdict"] = "WRONG"
                    return out
                elif r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                else:
                    out["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    out["verdict"] = "WRONG"
                    return out
            except Exception as e:
                out["error"] = f"exception: {e}"
                time.sleep(2)
        out["verdict"] = "WRONG"
        out["judge_raw_output"] = "(all retries failed)"
        return out
    finally:
        if own_http:
            http.close()


def score_4way(verdict: str, lam: float = -1.0, no_cs_match: bool = False) -> float:
    """Map 4-way verdict to a numeric score for tier accuracy.

    If no_cs_match is True (probe is for a name with no real CS researcher),
    REFUSAL is the CORRECT behavior (score +1) and any non-refusal is WRONG.
    """
    if no_cs_match:
        if verdict == "REFUSAL":
            return 1.0
        return lam  # WRONG
    if verdict == "CORRECT_STRONG":
        return 1.0
    if verdict == "CORRECT_WEAK":
        return 0.5
    if verdict == "REFUSAL":
        return 0.0
    if verdict == "WRONG":
        return lam
    return 0.0
