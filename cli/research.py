"""Research mode: query landmark + SOTA models with a researcher name or
free-form question so readers can explore the paper interactively."""

from cli.presets import PRESET
from cli.probes import find_probes_for_researcher
from cli.progress import run_with_progress
from cli.query import query_model

DIM = "\033[90m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _print_header(question: str, gold: str | None):
    print(f"\n{BOLD}Q:{RESET} {question}")
    if gold:
        print(f"{DIM}Gold answer: {gold}{RESET}")
    print()


def _make_formatter(models: list):
    name_w = max(len(m["name"]) for m in models)
    tier_w = max(len(f"[{m['tier']}]") for m in models)

    def format_done(model, result, error, elapsed):
        tag = f"[{model['tier']}]"
        if error:
            body = f"<error: {error}>"
        else:
            body = (result or "").strip() or "<no response>"
        body = body.replace("\n", " ")
        timing = f"{DIM}({elapsed:.1f}s){RESET}"
        return f"{GREEN}✓{RESET} {model['name']:<{name_w}} {DIM}{tag:<{tier_w}}{RESET} {body} {timing}"

    return format_done


def _run_against_models(question: str, models: list):
    run_with_progress(
        models,
        lambda m, set_status: query_model(m, question),
        format_done=_make_formatter(models),
    )


def run_research(args):
    if args.question:
        _print_header(args.question, gold=None)
        _run_against_models(args.question, PRESET)
        print()
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
        _print_header(p["question"], gold=p.get("answer"))
        _run_against_models(p["question"], PRESET)
        print()
