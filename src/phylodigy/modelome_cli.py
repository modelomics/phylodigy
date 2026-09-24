"""Command-line registration for modelome workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .io import read_json_object, write_json_data


def add_modelome_commands(commands: argparse._SubParsersAction) -> None:
    """Register modelome planning and tree-building commands."""
    plan = commands.add_parser(
        "modelome-plan",
        help="plan the graph-profile coverage needed for a modelome",
    )
    plan.add_argument("entries", metavar="ENTRIES")
    plan.add_argument("-o", "--output", metavar="PATH")
    plan.set_defaults(handler=_modelome_plan)

    tree = commands.add_parser(
        "modelome-tree",
        help="build a graph-character tree from a modelome entry bundle",
    )
    tree.add_argument("entries", metavar="ENTRIES")
    tree.add_argument("--profiles-dir", metavar="PATH")
    tree.add_argument(
        "--bindings",
        metavar="JSON_FILE",
        help="JSON object mapping modelome entry IDs to artifact IDs",
    )
    tree.add_argument("--max-taxa", type=int, default=100)
    tree.add_argument("-o", "--output", metavar="PATH")
    tree.add_argument("--newick", metavar="PATH")
    tree.set_defaults(handler=_modelome_tree)


def _modelome_plan(args: argparse.Namespace) -> int:
    # Keep the optional modelome pipeline out of the import path for other commands.
    from .modelome import plan_modelome_tree

    result = plan_modelome_tree(args.entries)
    write_json_data(result, args.output)
    return 0


def _modelome_tree(args: argparse.Namespace) -> int:
    from .modelome import build_modelome_tree

    bindings = read_json_object(args.bindings) if args.bindings else None
    result = build_modelome_tree(
        args.entries,
        profiles_dir=args.profiles_dir,
        bindings=bindings,
        max_taxa=args.max_taxa,
    )

    if args.newick:
        if args.output in (None, "-") and args.newick == "-":
            raise ValueError("cannot write JSON and Newick to standard output together")
        if args.output not in (None, "-") and args.newick != "-":
            if Path(args.output).resolve() == Path(args.newick).resolve():
                raise ValueError("JSON output and Newick output must use different paths")
        if result.get("tree") is None:
            raise ValueError("cannot write Newick: modelome result has no inferred tree")
        from .newick import lineage_to_newick

        newick = lineage_to_newick(result) + "\n"
        if args.newick == "-":
            sys.stdout.write(newick)
        else:
            Path(args.newick).write_text(newick, encoding="utf-8")

    write_json_data(result, args.output)
    return 0


__all__ = ["add_modelome_commands"]
