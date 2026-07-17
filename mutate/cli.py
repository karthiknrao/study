"""Command-line runner: ``python -m mutate.cli --prompt "..." --op flip_char``.

For SYNTACTIC operators the mutated prompt is produced offline and printed directly.
For SEMANTIC (requires_llm) operators the filled meta-prompt is printed -- it is the
instruction an LLM would execute to produce the final variant. (Wire a backend by calling
``apply_operator`` programmatically with ``llm=<callable>``.)
"""
from __future__ import annotations

import argparse
import json
import sys

from .apply import apply_operator, list_operators


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mutate", description="Apply a mutation operator to a prompt.")
    p.add_argument("--prompt", "-p", required=False, help="The prompt to mutate.")
    p.add_argument("--op", "-o", default=None, help="Operator name. If omitted, one is sampled.")
    p.add_argument("--params", default=None,
                   help='Operator params as JSON, e.g. \'{"mode":"word_order"}\'.')
    p.add_argument("--list", action="store_true", help="List all operators and exit.")
    p.add_argument("--family", default=None, help="Filter --list by 'semantic' or 'syntactic'.")
    p.add_argument("--params-menu", default=None,
                   help="Print the option menu for an operator's enum-like params, then exit.")
    args = p.parse_args(argv)

    if args.list:
        print(f"{'operator':<26} {'family':<10} description")
        print("-" * 88)
        for name, family, desc, _src in list_operators(args.family):
            print(f"{name:<26} {family:<10} {desc}")
        return 0

    if args.params_menu:
        from .options import get_param_options
        try:
            menu = get_param_options(args.params_menu)
        except KeyError as exc:
            p.error(str(exc))
        print(f"# {args.params_menu} param options")
        for param, opts in menu.items():
            shown = opts if len(opts) <= 12 else opts[:12] + [f"... (+{len(opts) - 12})"]
            print(f"{param}: {shown}")
        return 0

    if not args.prompt:
        p.error("--prompt/-p is required unless --list is used")

    params = json.loads(args.params) if args.params else None
    if args.op is None:
        from .apply import sample_operator
        args.op = sample_operator()
        print(f"# sampled operator: {args.op}", file=sys.stderr)

    result = apply_operator(args.op, args.prompt, params=params)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
