"""LLM-as-judge scoring for researcher field probes and flexible matching.

For researcher probes, the model's answer is a research field name.
We need flexible matching because:
1. Same field has multiple names ("computer networking" = "networking" = "networks")
2. Adjacent fields should be accepted (if gold="computer networking" and
   model says "distributed systems", this is close enough — guessing even
   a broad CS subfield without knowing the person is very unlikely)
3. The model may phrase it differently ("ML systems" vs "machine learning systems")

For factual probes (founding year, chemical formula), exact match is fine.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Field adjacency groups — fields in the same group are considered correct matches
ADJACENT_FIELDS = [
    {"computer networking", "networking", "networks", "data center networking",
     "network systems", "internet systems", "network measurement"},
    {"operating systems", "systems software", "OS", "kernel systems"},
    {"distributed systems", "cloud computing", "parallel systems",
     "cluster computing", "computer systems"},
    {"computer architecture", "hardware systems", "processor design",
     "microarchitecture", "VLSI", "chip design", "hardware"},
    {"computer security", "cybersecurity", "information security",
     "network security", "systems security", "security"},
    {"database systems", "data management", "databases", "data systems"},
    {"machine learning", "deep learning", "artificial intelligence", "AI", "ML"},
    {"programming languages", "compilers", "program analysis", "PL",
     "software verification", "formal methods"},
    {"human-computer interaction", "HCI", "user interfaces", "UX"},
    {"data mining", "knowledge discovery", "information retrieval", "web mining"},
    {"theoretical computer science", "algorithms", "complexity theory",
     "computational complexity", "theory of computation"},
    {"storage systems", "file systems", "storage", "persistent storage"},
    {"computer graphics", "visualization", "rendering", "graphics"},
    {"software engineering", "software development", "SE"},
    {"computer vision", "image processing", "visual computing"},
    {"natural language processing", "NLP", "computational linguistics"},
    {"robotics", "autonomous systems", "robot systems"},
]

# Build lookup: field_name -> group_index
_field_to_group = {}
for i, group in enumerate(ADJACENT_FIELDS):
    for field in group:
        _field_to_group[field.lower()] = i


def fields_match(gold: str, predicted: str) -> bool:
    """Check if predicted field matches or is adjacent to gold field.

    Returns True if:
    1. Exact match (after normalization)
    2. Gold and predicted are in the same adjacency group
    3. One contains the other as a substring
    """
    gold_lower = gold.lower().strip()
    pred_lower = predicted.lower().strip()

    # Exact match
    if gold_lower == pred_lower:
        return True

    # Substring match
    if gold_lower in pred_lower or pred_lower in gold_lower:
        return True

    # Adjacency group match
    gold_group = _field_to_group.get(gold_lower)
    pred_group = _field_to_group.get(pred_lower)
    if gold_group is not None and pred_group is not None:
        if gold_group == pred_group:
            return True

    # Check if any word from gold's group appears in predicted
    if gold_group is not None:
        group = ADJACENT_FIELDS[gold_group]
        for field_name in group:
            if field_name.lower() in pred_lower:
                return True

    return False


def score_researcher_probe(response: str, gold_field: str) -> bool:
    """Score a researcher probe response.

    The response should mention the researcher's field.
    Accept if:
    - Response contains the gold field or an adjacent field
    - Response does NOT say "unknown" or "I don't know"
    """
    if not response or not response.strip():
        return False

    response_lower = response.lower()

    # Check for refusal/unknown
    refusal_patterns = ["unknown", "i don't know", "i'm not sure",
                       "i cannot", "i can't", "not familiar",
                       "no information", "not aware"]
    for pattern in refusal_patterns:
        if pattern in response_lower:
            return False

    # Check field match
    return fields_match(gold_field, response)


def score_with_llm_judge(
    response: str,
    gold_answer: str,
    question: str,
    judge_client,
    judge_model: str = "anthropic/claude-sonnet-4",
) -> bool:
    """Use an LLM as judge for complex matching cases.

    This is the fallback when simple matching fails.
    The judge LLM evaluates whether the response is correct
    given the question and gold answer.
    """
    prompt = f"""You are judging whether a model's response correctly answers a factual question.

Question: {question}
Gold answer: {gold_answer}
Model's response: {response}

Rules:
- For research field questions: accept adjacent/related fields as correct.
  E.g., if gold is "computer networking" and response mentions "distributed systems" or "network systems", that's correct.
- For founding year questions: the year must match exactly.
- For chemical formula questions: the formula must match exactly.
- If the model says "I don't know" or refuses, that is INCORRECT.
- If the model gives a wrong answer, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT"""

    try:
        result = judge_client.get_response_text(
            judge_model,
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        return "CORRECT" in result.upper()
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return False
