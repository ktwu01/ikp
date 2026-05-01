"""Evaluation mode: query a probe ID against preset + user models, then
score each response with the paper's judge prompt."""

from cli.judge import judge
from cli.presets import PRESET
from cli.probes import find_probe_by_id
from cli.progress import run_with_progress
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


def _make_work_fn(question: str, gold: str):
    def work(model: dict, set_status):
        set_status("querying")
        response = query_model(model, question)
        set_status("judging")
        try:
            verdict = judge(question, gold, response)
        except Exception as e:
            verdict = f"JUDGE_ERR: {e}"
        return {"response": response, "verdict": verdict}
    return work


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

    name_w = max(len(m["name"]) for m in models)
    tier_w = max(len(f"[{m['tier']}]") for m in models)

    def format_done(model, result, error, elapsed):
        if error:
            verdict = "WRONG"
            resp = f"<error: {error}>"
        else:
            verdict = result["verdict"]
            resp = (result["response"].strip() or "<no response>").replace("\n", " ")
        if len(resp) > 200:
            resp = resp[:197] + "..."
        sym, color = SYMBOLS.get(verdict, ("•", DIM))
        tag = f"[{model['tier']}]"
        timing = f"{DIM}({elapsed:.1f}s){RESET}"
        return (f"{color}{sym}{RESET} {model['name']:<{name_w}} {DIM}{tag:<{tier_w}}{RESET} "
                f"{color}{verdict:<8}{RESET} {resp} {timing}")

    results = run_with_progress(models, _make_work_fn(question, gold), format_done=format_done)
    correct = sum(1 for r in results
                  if not r["error"] and r["result"] and r["result"]["verdict"] == "CORRECT")
    print(f"\n  {correct}/{len(results)} correct\n")
