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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamicmultinets import RenMachine, ScriptedController          # noqa: E402
from dynamicmultinets.controller import LLMController                # noqa: E402
from dynamicmultinets.render import save_gallery                     # noqa: E402

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

        # --- fold the construction loop into one rule -----------------------
        # Proof search follows one action per drawing, so the construction is a
        # single path and one wrong perception ends it -- "search space
        # exhausted" after a dozen nodes, on rules that both verify above 0.99.
        # This runs the constructor to its own fixed point and keeps whichever
        # drawing `read_angle_facts` most calls the proof configuration, so a
        # wrong step costs a candidate instead of the proof. The judge has to
        # be the OTHER rule: a constructor scoring its own work is not evidence.
        ("iterate_rule", {"step": "construct_aux_line",
                          "judge": "read_angle_facts",
                          "judge_target": "A1=B1,A2=B2,A1+A3+A2=180",
                          "new_name": "construct_until_parallel",
                          "max_iters": 12}),

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
    ap.add_argument("--dump", default="renders/geometry",
                    help="directory the construction sketches are written to, so "
                         "the drawing the proof ran on can be checked by eye")
    ap.add_argument("--no-dump", action="store_true", help="do not write any images")
    args = ap.parse_args()

    # 1200x30 leaves both rules around 0.95, and 0.95 compounded over a
    # seven-link construction is a 0.68 proof -- the chain is only ever as good
    # as its weakest perception step raised to the power of how often the
    # construction loop fires. Trained to convergence they verify at ~1.00 and
    # ~0.99, the construction stops wasting a rotation on the tolerance
    # boundary, and the same proof comes out above 0.95. It costs ~10 minutes
    # on a GPU; --quick is still there for a 30-second smoke test.
    n_train, epochs = (300, 10) if args.quick else (4000, 60)
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

    if args.dump and not args.no_dump:
        out = Path(args.dump)
        if not out.is_absolute():
            out = ROOT / out

        # Every drawing on the tape, then the proof replayed on the one it
        # started from. The search keeps only the TEXT of each step, and text
        # is the one thing this experiment is not about: the construction is
        # only convincing if the auxiliary line can be seen walking onto the
        # apex and turning parallel, one image per step.
        written = save_gallery(
            ((cell.text or "cell", cell.image) for _, cell in machine.specific),
            str(out), prefix="tape", reset=True)

        def replay():
            cell = machine.head("specific")
            yield f"start_{cell.text or 'cell'}", cell.image
            for st in proof.steps:
                cell = machine.library.get(st.rule).apply(cell)
                if cell is None:                 # only if a rule stopped applying
                    break
                yield f"{st.rule}_{st.after}", cell.image

        if proof.found:
            written += save_gallery(replay(), str(out), prefix="step")

        # Held-out scenes captioned with what each rule answered and what the
        # oracle wanted -- the verification numbers, made checkable by eye.
        # `answer_text` because construct_aux_line answers with a drawing, and
        # what it DECIDED lives in that cell's meta, not in its caption.
        from dynamicmultinets.verify import answer_text, normalize

        for ds_name, rule_name in (("geo_fresh", "construct_aux_line"),
                                   ("facts_fresh", "read_angle_facts")):
            if ds_name not in machine.datasets or rule_name not in machine.library:
                continue
            rule = machine.library.get(rule_name)

            def cases(ds=machine.datasets[ds_name], rule=rule):
                for ex in ds.examples[:8]:
                    want = answer_text(ex.out) if ex.labeled else "?"
                    got = answer_text(rule.apply(ex.inp)) or "(no answer)"
                    mark = ("unlabeled" if not ex.labeled
                            else "ok" if normalize(got) == normalize(want)
                            else "WRONG")
                    yield f"{mark}_want-{want}_got-{got}", ex.inp.image

            written += save_gallery(cases(), str(out), prefix=f"{rule_name}_")

        print(f"\nwrote {written} images to {out} (captions in {out / 'index.txt'})")

    print("\n--- library ---")
    print(machine.library.table())
    print("\n--- objective ---")
    print(machine.report().summary())


if __name__ == "__main__":
    main()
