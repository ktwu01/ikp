"""Evaluation mode: query a probe ID against preset + user models, then
score each response with the paper's judge prompt."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from cli.judge import judge
from cli.presets import PRESET
from cli.probes import find_probe_by_id
from cli.query import query_model

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"
SYMBOLS = {"CORRECT": ("✓", GREEN), "WRONG": ("✗", RED), "REFUSAL": ("?", YELLOW)}


def _parse_extra_models(specs: list[str]) -> list[dict]:
    models = []
    for spec in specs or []:
        parts = [p.strip() for p in spec.split(",")]
        kv = dict(p.split("=", 1) for p in parts if "=" in p)
        model_id = kv.get("id") or parts[0]
        models.append({
            "name": kv.get("name", model_id.split("/")[-1]),
            "id": model_id,
            "type": kv.get("type", "openrouter"),
            "thinking": kv.get("thinking", "false").lower() == "true",
            "tier": kv.get("tier", "USER"),
        })
    return models


def _run_one(model: dict, question: str, gold: str) -> dict:
    try:
        response = query_model(model, question)
    except Exception as e:
        return {"model": model, "response": f"<error: {e}>", "verdict": "WRONG"}
    try:
        verdict = judge(question, gold, response)
    except Exception as e:
        verdict = f"JUDGE_ERR: {e}"
    return {"model": model, "response": response, "verdict": verdict}


def run_eval(args):
    probe = find_probe_by_id(args.probe_id)
    if not probe:
        print(f"Probe '{args.probe_id}' not found.")
        return

    models = list(PRESET) + _parse_extra_models(args.model)
    question, gold = probe["question"], probe["answer"]

    print(f"\n{BOLD}Probe:{RESET} {probe['id']}  ({probe['tier']}, {probe.get('domain','?')})")
    print(f"{BOLD}Q:{RESET} {question}")
    print(f"{BOLD}Gold:{RESET} {gold}\n")

    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(models))) as ex:
        futures = {ex.submit(_run_one, m, question, gold): m for m in models}
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda r: models.index(r["model"]))

    width = max(len(r["model"]["name"]) for r in results) + 2
    for r in results:
        m = r["model"]
        sym, color = SYMBOLS.get(r["verdict"], ("•", DIM))
        resp = (r["response"].strip() or "<no response>").replace("\n", " ")
        if len(resp) > 200:
            resp = resp[:197] + "..."
        print(f"  {color}{sym}{RESET} {m['name']:<{width}} {DIM}[{m['tier']}]{RESET} "
              f"{color}{r['verdict']:<8}{RESET} {resp}")

    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    total = len(results)
    print(f"\n  {correct}/{total} correct\n")
