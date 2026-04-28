"""Research mode: query landmark + SOTA models with a researcher name or
free-form question so readers can explore the paper interactively."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from cli.presets import PRESET
from cli.probes import find_probes_for_researcher
from cli.query import query_model

DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _run_against_models(question: str, models: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(models))) as ex:
        futures = {ex.submit(query_model, m, question): m for m in models}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                resp = fut.result()
            except Exception as e:
                resp = f"<error: {e}>"
            results.append({"model": m, "response": resp})
    results.sort(key=lambda r: PRESET.index(r["model"]) if r["model"] in PRESET else 999)
    return results


def _print_responses(question: str, results: list, gold: str | None = None):
    print(f"\n{BOLD}Q:{RESET} {question}")
    if gold:
        print(f"{DIM}Gold answer: {gold}{RESET}")
    print()
    width = max(len(r["model"]["name"]) for r in results) + 2
    for r in results:
        m = r["model"]
        tag = f"[{m['tier']}]"
        resp = r["response"].strip() or "<no response>"
        print(f"  {m['name']:<{width}} {DIM}{tag:<7}{RESET} {resp}")
    print()


def run_research(args):
    if args.question:
        question = args.question
        gold = None
        _print_responses(question, _run_against_models(question, PRESET), gold)
        return

    name = args.researcher
    probes = find_probes_for_researcher(name)
    if not probes:
        print(f"No researcher probes found matching '{name}'.")
        print("Tip: try a partial name; matching is case-insensitive substring.")
        return

    if len(probes) > 1 and not args.all:
        print(f"\nFound {len(probes)} probes matching '{name}':")
        for i, p in enumerate(probes[:20]):
            print(f"  [{i}] {p['id']}  {p['researcher_name']}  ({p['tier']})")
        if len(probes) > 20:
            print(f"  ... and {len(probes) - 20} more")
        print("\nRun with --all to query every match, or use --probe-id <ID> in eval mode.")
        probes = probes[:1]

    for p in probes:
        question = p["question"]
        results = _run_against_models(question, PRESET)
        _print_responses(question, results, gold=p.get("answer"))
