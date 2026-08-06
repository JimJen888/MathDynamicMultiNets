"""
Command line entry point.

    python -m dynamicmultinet.cli tools
        Print the controller's instruction set -- every operation the machine
        can perform, with its parameters. Read this before writing a goal.

    python -m dynamicmultinet.cli catalogue
        Print the generators and oracles a controller may choose from.

    python -m dynamicmultinet.cli run --goal "..." [--llm]
        Start a machine and hand it a goal. With --llm, Claude decides what to
        do next; without it, the run is a no-op scaffold (there is no default
        plan for an arbitrary goal) and you should use the examples instead.

    python -m dynamicmultinet.cli inspect --library DIR
        Load a saved library and print it, priced.
"""

from __future__ import annotations

import argparse

from .controller import DEFAULT_MODEL, LLMController
from .generators import catalogue as generator_catalogue
from .machine import RenMachine
from .oracles import catalogue as oracle_catalogue
from .tools import build_tools


def cmd_tools(_: argparse.Namespace) -> None:
    for tool in build_tools(RenMachine(with_prior=False)).values():
        params = ", ".join(tool.input_schema["properties"]) or "(none)"
        print(f"\n{tool.name}({params})\n  {tool.description}")


def cmd_catalogue(_: argparse.Namespace) -> None:
    print("GENERATORS\n" + generator_catalogue())
    print("\nORACLES\n" + oracle_catalogue())


def cmd_run(args: argparse.Namespace) -> None:
    machine = RenMachine(goal=args.goal, device=args.device)
    if not args.llm:
        print("Nothing to run: an arbitrary goal needs a controller that can "
              "decide what to do.\nRe-run with --llm (needs ANTHROPIC_API_KEY or "
              "an `ant auth login` profile),\nor run one of the scripted "
              "examples in examples/.")
        print("\nmachine state:\n" + machine.state())
        return
    run = LLMController(machine, model=args.model, max_steps=args.max_steps).run(args.goal)
    print("\n" + run.summary())
    print("\n" + machine.library.table())
    if machine.benchmark:
        print("\n" + machine.report().summary())
    if args.save:
        print("saved:", machine.save(args.save))


def cmd_inspect(args: argparse.Namespace) -> None:
    machine = RenMachine(with_prior=False, device=args.device)
    machine.load(args.library)
    print(machine.library.table())
    print(f"\ntotal description length: {machine.library.total_bits():.0f} bits")


def main() -> None:
    ap = argparse.ArgumentParser(prog="dynamicmultinet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="cuda when a GPU is present; pass cpu to pin it")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="print the controller's instruction set").set_defaults(
        func=cmd_tools)
    sub.add_parser("catalogue", help="print generators and oracles").set_defaults(
        func=cmd_catalogue)

    run = sub.add_parser("run", help="hand a goal to a machine")
    run.add_argument("--goal", required=True)
    run.add_argument("--llm", action="store_true")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--max-steps", type=int, default=40)
    run.add_argument("--save", default="")
    run.set_defaults(func=cmd_run)

    ins = sub.add_parser("inspect", help="load and price a saved library")
    ins.add_argument("--library", required=True)
    ins.set_defaults(func=cmd_inspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
