"""IKP CLI entrypoint. Two subcommands:

  research  — explore the benchmark by researcher name or free-form question
  eval      — re-run a specific probe ID against landmarks + SOTA + user models
"""

import argparse
import sys

from cli.evaluate import run_eval
from cli.research import run_research


def main():
    parser = argparse.ArgumentParser(
        prog="ikp",
        description="IKP interactive CLI — explore researcher knowledge across LLMs.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("research", help="Look up a researcher or ask a free-form question.")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--researcher", help="Researcher name (substring match against the probe set).")
    g.add_argument("--question", help="Free-form factual question.")
    r.add_argument("--all", action="store_true",
                   help="If multiple researcher probes match, query every one (default: first match only).")

    e = sub.add_parser("eval", help="Run a probe ID against preset + user models with LLM-judge scoring.")
    e.add_argument("probe_id", help="Probe ID, e.g. IKP_T7_1234.")
    e.add_argument("--model", action="append", default=[],
                   help="Extra model spec, repeatable. Format: 'openrouter_id' or "
                        "'id=foo/bar,name=foo,thinking=true,type=openrouter,tier=USER'.")

    args = parser.parse_args()
    if args.mode == "research":
        run_research(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
