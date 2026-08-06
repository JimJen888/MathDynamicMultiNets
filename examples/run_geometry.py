"""
Experiment 2: proving that the interior angles of a triangle sum to 180 degrees.

The workflow of Figure 3, as a rule chain. The machine starts from a DRAWING --
a triangle and an auxiliary line at the wrong place and the wrong angle -- and
has to reach a symbolic conclusion. Nothing about that is a symbol-manipulation
problem until the very last step:

    sketch --construct_aux_line (learned, specific -> specific)-->  ... repeat
    sketch --read_angle_facts   (learned, specific -> abstract)-->  A1=B1, A2=B2,
                                                                   A1+A3+A2=180
           --substitute_equalities (prior, abstract -> abstract)--> B1+A3+B2=180

Two learned rules, and they are different KINDS of rule, which is the point:

  * `construct_aux_line` looks at the drawing and edits it. Its output is
    another drawing, so it can be applied again -- the construction loop is
    proof search over a rule that stays inside the specific domain. The
    decision is perception (is the line through the apex? is it parallel?); the
    update is arithmetic on scene parameters.
  * `read_angle_facts` looks at the finished construction and says what it
    licenses. It only licenses the alternate-angle equalities when the line
    really does pass through the apex AND is really parallel to the opposite
    edge -- so it has to be able to see both, and a net that guesses "the facts
    hold" on every input fails verification.

The final substitution is exact and symbolic, and belongs in the abstract
domain. That division -- see, then compute -- is the architecture.

    python examples/run_geometry.py           # GPU if there is one
    python examples/run_geometry.py --quick   # ~30 s, weaker rules
    python examples/run_geometry.py --llm     # Claude drives instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinet import RenMachine, ScriptedController          # noqa: E402
from dynamicmultinet.controller import LLMController                # noqa: E402

GOAL = (
    "Starting from a drawing of a triangle with an auxiliary line in the wrong "
    "position, learn the rules needed to reach the symbolic conclusion "
    "B1+A3+B2=180 -- the interior angles of a triangle sum to a flat angle."
)
CONCLUSION = "B1+A3+B2=180"


def plan(n_train: int, epochs: int) -> list[tuple[str, dict]]:
    return [
        ("inspect_machine", {}),
        ("show_catalogue", {}),

        # --- rule 1: the construction step (specific -> specific) -----------
        ("generate_data", {"generator": "triangle_scenes", "n": n_train, "seed": 3,
                           "name": "geo_train", "params": {"solved_fraction": 0.25}}),
        ("label_data", {"dataset": "geo_train", "oracle": "next_construction_step"}),
        ("declare_rule", {"name": "construct_aux_line", "domain_in": "specific",
                          "out_domain": "specific", "kind": "scene_action",
                          "from_oracle": "next_construction_step",
                          "description": "move the auxiliary line through the apex, "
                                         "then rotate it parallel to the opposite edge"}),
        ("train_rule", {"rule": "construct_aux_line", "dataset": "geo_train",
                        "epochs": epochs}),
        ("generate_data", {"generator": "triangle_scenes", "n": 200, "seed": 61,
                           "name": "geo_fresh", "params": {"solved_fraction": 0.25}}),
        ("label_data", {"dataset": "geo_fresh", "oracle": "next_construction_step"}),
        ("verify_rule", {"rule": "construct_aux_line", "dataset": "geo_fresh",
                         "oracle": "next_construction_step", "threshold": 0.90}),
        ("grow_ensemble", {"rule": "construct_aux_line", "dataset": "geo_train",
                           "epochs": max(6, epochs // 2)}),

        # --- rule 2: reading the finished construction ----------------------
        ("generate_data", {"generator": "triangle_scenes", "n": n_train, "seed": 11,
                           "name": "facts_train", "params": {"solved_fraction": 0.4}}),
        ("label_data", {"dataset": "facts_train", "oracle": "alternate_angle_facts"}),
        ("declare_rule", {"name": "read_angle_facts", "domain_in": "specific",
                          "out_domain": "abstract",
                          "from_oracle": "alternate_angle_facts",
                          "description": "state the angle equalities the drawing "
                                         "licenses, if it licenses any"}),
        ("train_rule", {"rule": "read_angle_facts", "dataset": "facts_train",
                        "epochs": epochs}),
        ("generate_data", {"generator": "triangle_scenes", "n": 200, "seed": 77,
                           "name": "facts_fresh", "params": {"solved_fraction": 0.4}}),
        ("label_data", {"dataset": "facts_fresh", "oracle": "alternate_angle_facts"}),
        ("verify_rule", {"rule": "read_angle_facts", "dataset": "facts_fresh",
                         "oracle": "alternate_angle_facts", "threshold": 0.90}),

        # --- the proof ------------------------------------------------------
        # A drawing that is NOT already in the proof configuration: the line
        # starts off the apex and at the wrong angle, so the chain has to run
        # the construction loop before there is anything to read.
        ("generate_data", {"generator": "triangle_scenes", "n": 8, "seed": 123,
                           "name": "geo_demo", "params": {"solved_fraction": 0.0}}),
        ("put_example_on_tape", {"dataset": "geo_demo", "index": 0}),
        ("prove_from_tape", {"domain": "specific", "target": CONCLUSION,
                             "max_depth": 14}),
        ("add_task_from_example", {"name": "triangle_180", "dataset": "geo_demo",
                                   "index": 0, "target": CONCLUSION, "max_depth": 14}),
        ("library_report", {}),
        ("finish", {"summary": "construction and angle-reading rules learned and "
                               "verified; the 180-degree conclusion reached from a "
                               "drawing"}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--device", default=None,
                    help="cuda when a GPU is present; pass cpu to pin it")
    ap.add_argument("--dump", default="", help="write the construction sketches here")
    args = ap.parse_args()

    n_train, epochs = (300, 10) if args.quick else (1200, 30)
    machine = RenMachine(goal=GOAL, device=args.device)

    if args.llm:
        run = LLMController(machine, max_steps=50).run(GOAL)
    else:
        run = ScriptedController(machine).run(plan(n_train, epochs))

    print("\n" + "=" * 78)
    print(run.summary())

    proof = machine.prove_from_tape("specific", CONCLUSION, max_depth=14)
    print("\n--- the proof ---")
    print(proof.as_text())
    if proof.found:
        print(f"\n  crosses domains: {proof.crosses_domains}   "
              f"confidence {proof.confidence:.4f}")
        machine.keep_proof(proof, "triangle_angle_sum",
                           "interior angles of a triangle sum to a flat angle")
        print("  kept as the rule 'triangle_angle_sum' -- the machine will not "
              "search for it again")

    if args.dump:
        # Every intermediate drawing, so the construction can be looked at.
        from dynamicmultinet.render import save_png

        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for i, (_, cell) in enumerate(machine.specific):
            if cell.image is not None:
                save_png(cell.image, str(out / f"step{i:02d}_{cell.text[:40]}.png"))
                written += 1
        print(f"\nwrote {written} sketches to {out}")

    print("\n--- library ---")
    print(machine.library.table())
    print("\n--- objective ---")
    print(machine.report().summary())


if __name__ == "__main__":
    main()
