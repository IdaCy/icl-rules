"""CLI: python -m icl_articulation.datagen.generators <rule_id>

Dispatches the rule to its registered generator module and runs the full
gated emit pipeline (Gate A schema, Gate B groundtruth, Gate C battery, Gate D
confound). On success it writes data/<rule_id>/items.jsonl +
data/<rule_id>/confound_report.json and prints the EmitSummary as JSON. Any gate
failure raises (loud) and exits non-zero with nothing written.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m icl_articulation.datagen.generators",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "rule_id",
        nargs="?",
        help="rule_id to generate (omit with --list to see registered rules)",
    )
    p.add_argument("--seed", type=int, default=registry.DEFAULT_SEED, help="generation seed")
    p.add_argument(
        "--no-write",
        action="store_true",
        help="run all gates but do not write items.jsonl (dry run)",
    )
    p.add_argument(
        "--no-pos",
        action="store_true",
        help="skip the 6 nltk first-word-POS battery predicates",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="output root (defaults to repo data/); items go under <root>/<rule_id>/",
    )
    p.add_argument(
        "--output-rule-id",
        default=None,
        help="output dataset directory and stored item rule_id; use for variants such as <rule>_deconfounded",
    )
    p.add_argument("--list", action="store_true", help="list registered rule_ids and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(registry.registered_rules()))
        return 0
    if not args.rule_id:
        print("error: rule_id is required (or use --list)", file=sys.stderr)
        return 2
    summary = registry.run(
        args.rule_id,
        seed=args.seed,
        write=not args.no_write,
        run_pos=not args.no_pos,
        data_dir=args.data_dir,
        output_rule_id=args.output_rule_id,
        stored_rule_id=args.output_rule_id,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    return 0 if summary.all_gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
