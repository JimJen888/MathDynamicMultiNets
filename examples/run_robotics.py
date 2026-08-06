"""
Appendix A: the Vision-Sketch-Reasoning-Action loop, as a mapping rule.

The robotic architecture of Figure A1 is the same machine pointed at the
outside world. A camera writes to the specific tape; a vision-to-sketch mapping
replaces the scene with a simplified drawing -- cylinders and spheres for the
robot, small spheres for obstacles, an arrow from the end effector towards the
goal -- and a learned rule maps that drawing to an escape direction in the
abstract domain, which is then executed.

The rule this example forms IS DetourNet, structurally:

    RuleNet(num_classes=7, num_slots=1)   ==   DetourNet with a smaller action set

Same two views, same shared encoder, same semantic-palette front end, same
fa - fb fusion, same single softmax over candidate motions. The labelling rule
is detourNet.label_best_detour verbatim in priority order: go straight if the
straight path is clear (an unnecessary detour is a wasted step), otherwise take
the reachable candidate that unblocks the goal and loses least ground, and emit
no label at all when nothing works -- a state with no valid detour teaches the
net nothing except noise.

What the Ren-machine framing adds on top of a plain classifier is the rest of
the loop: the drawing is a tape cell the machine can rewrite and look at again,
the decision is a symbol it can reason about, and the whole thing is a rule
that gets verified and priced like any other.

    python examples/run_robotics.py           # GPU if there is one
    python examples/run_robotics.py --quick   # ~30 s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamicmultinet import RenMachine, ScriptedController          # noqa: E402
from dynamicmultinet.controller import LLMController                # noqa: E402
from dynamicmultinet.oracles import ESCAPE_DIRECTIONS               # noqa: E402
from dynamicmultinet.render import save_gallery                     # noqa: E402

GOAL = (
    "Learn to choose an escape direction by looking at a sketch of the robot, "
    "the obstacles and the goal; verify it against the collision geometry, and "
    "specialise it on the cases it gets wrong."
)


def plan(n_train: int, epochs: int) -> list[tuple[str, dict]]:
    return [
        ("inspect_machine", {}),
        ("show_catalogue", {}),
        ("generate_data", {"generator": "robot_scenes", "n": n_train, "seed": 5,
                           "name": "scenes_train",
                           "params": {"n_obstacles": 2, "blocked_fraction": 0.5}}),
        ("label_data", {"dataset": "scenes_train", "oracle": "best_escape_direction"}),
        ("declare_rule", {"name": "sketch_to_direction", "domain_in": "specific",
                          "out_domain": "abstract",
                          "from_oracle": "best_escape_direction",
                          "description": "escape direction from a sketch: straight to "
                                         "the goal when the path is clear, otherwise "
                                         "the detour that loses least ground"}),
        ("train_rule", {"rule": "sketch_to_direction", "dataset": "scenes_train",
                        "epochs": epochs}),

        # Fresh scenes from a different seed: the collision geometry is the
        # ground truth here, so this is a 'measured' check, not a derived one.
        ("generate_data", {"generator": "robot_scenes", "n": 300, "seed": 404,
                           "name": "scenes_fresh",
                           "params": {"n_obstacles": 2, "blocked_fraction": 0.5}}),
        ("label_data", {"dataset": "scenes_fresh", "oracle": "best_escape_direction"}),
        ("verify_rule", {"rule": "sketch_to_direction", "dataset": "scenes_fresh",
                         "oracle": "best_escape_direction", "threshold": 0.85}),

        # Section 3 again: the base rule is wrong on a minority of scenes; train
        # a specialist on exactly those and let it override -- but only if it
        # actually helps on examples neither of them was fitted to.
        ("grow_ensemble", {"rule": "sketch_to_direction", "dataset": "scenes_train",
                           "epochs": epochs}),

        ("put_example_on_tape", {"dataset": "scenes_fresh", "index": 0}),
        ("apply_rule", {"rule": "sketch_to_direction"}),
        ("finish", {"summary": "sketch-to-direction rule learned, verified against "
                               "the collision geometry, and specialised"}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--device", default=None,
                    help="cuda when a GPU is present; pass cpu to pin it")
    ap.add_argument("--dump", default="renders/robotics",
                    help="directory the sketches are written to, so the rule's "
                         "answers can be checked against what it was looking at")
    ap.add_argument("--no-dump", action="store_true", help="do not write any images")
    args = ap.parse_args()

    n_train, epochs = (300, 10) if args.quick else (1500, 30)
    machine = RenMachine(goal=GOAL, device=args.device)

    if args.llm:
        run = LLMController(machine, max_steps=40).run(GOAL)
    else:
        run = ScriptedController(machine).run(plan(n_train, epochs))

    print("\n" + "=" * 78)
    print(run.summary())

    # The planner's view: candidates ranked best-first, which is what
    # DetourNet.rank gives the trial loop. The collision check vets whatever
    # this proposes, so "the right answer is in the top 3" is the operational
    # criterion -- top-1 alone understates a rule whose runner-up was fine.
    from dynamicmultinet.verify import topk_report

    rule = machine.library.get("sketch_to_direction")
    fresh = machine.datasets["scenes_fresh"]
    print("\n--- ranked candidates on five fresh scenes ---")
    for ex in fresh.examples[:5]:
        want = ex.out.text if ex.labeled else "?"
        ranked = rule.rank_options(ex.inp, 3)
        shown = ", ".join(f"{n}({p:.2f})" for n, p in ranked)
        mark = ("ok  " if ranked[0][0] == want
                else "top3" if want in [n for n, _ in ranked] else "MISS")
        print(f"  truth {want:>7}  {mark}  ranked: {shown}")

    stats = topk_report(rule, fresh, k=3)
    print(f"\n  over {stats['n']} fresh scenes: top-1 {stats['top1']:.3f}, "
          f"top-3 {stats['top3']:.3f}")
    print("  per class (correct/total):")
    for cls, (ok, tot) in stats["per_class"].items():
        print(f"    {cls:>7}: {ok:>3}/{tot:<4}")
    print("  'direct' is the class that must not be missed -- answering a detour\n"
          "  on a clear path wastes a step, but answering 'direct' on a blocked\n"
          "  one drives the arm into the obstacle.")

    for name in ("sketch_to_direction", "sketch_to_direction_ens"):  # _ens may not exist
        if name in machine.library:
            r = machine.library.get(name)
            print(f"\n{name}: {r.stats.summary()}, "
                  f"{'trusted' if r.trusted else 'NOT trusted'}")
            if r.stats.counterexamples:
                print("  first counterexamples:")
                for c in r.stats.counterexamples[:3]:
                    print("   ", c)

    print(f"\naction set: {', '.join(ESCAPE_DIRECTIONS)}")
    print("the same architecture with 19 classes and the full 0.05/0.40/0.75 x "
          "6-axis action set is DetourNet")

    if args.dump and not args.no_dump:
        # The sketches the rule was actually looking at, captioned with what it
        # answered and what the collision geometry says. Reading those two off
        # the picture is the only way to tell a wrong answer from a scene whose
        # obstacles make the "right" one debatable.
        out = Path(args.dump)
        if not out.is_absolute():
            out = ROOT / out

        def scenes():
            for ex in fresh.examples[:8]:
                want = ex.out.text if ex.labeled else "?"
                ranked = rule.rank_options(ex.inp, 3)
                mark = ("unlabeled" if not ex.labeled
                        else "ok" if ranked[0][0] == want else "WRONG")
                shown = " ".join(f"{n}:{p:.2f}" for n, p in ranked)
                yield f"{mark}_truth-{want}_pred-{ranked[0][0]}  [{shown}]", ex.inp.image

        n = save_gallery(scenes(), str(out), prefix="scene", reset=True)
        n += save_gallery(((c.text or "cell", c.image) for _, c in machine.specific),
                          str(out), prefix="tape")
        print(f"\nwrote {n} images to {out} (captions in {out / 'index.txt'})")


if __name__ == "__main__":
    main()
