"""
Experiment 1: autonomously learning the distributive rule, and using it.

This is the workflow of Figure 2. The machine

  1. generates experiments in the SPECIFIC domain -- products drawn on the
     screen -- and checks each rewrite numerically before keeping it, which is
     the paper's "it finds that actually the equation 12x30 = 10x30 + 2x30
     holds true";
  2. trains a mapping rule on them (specific -> specific);
  3. verifies it on data it never trained on, twice: against an oracle, and
     against a chain of rules it already trusts (agreement between two
     independent routes is stronger evidence than either alone);
  4. grows a specialist over the cases the base rule still gets wrong -- the
     section-3 construction for pushing a rule towards 100%;
  5. learns to READ the screen (specific -> abstract), which is the step that
     lets a chain come back out of the specific domain;
  6. proves things with mixed chains and prices the library.

The ending is not rigged. The learned rewrite turns out to duplicate an
algebraic identity the machine already had, and the conciseness objective says
so: two extra domain crossings cost more than they buy. The reader, by
contrast, does something no symbolic rule can do, and survives. That asymmetry
is the interesting result, and it is what the objective is for.

    python examples/run_multiplication.py            # two nets, 60 epochs each
    python examples/run_multiplication.py --quick    # ~30 s, weaker rules
    python examples/run_multiplication.py --llm      # Claude drives instead
    python examples/run_multiplication.py --device cpu   # pin the device
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
    "Learn the distributive rule of multiplication from experiments you generate "
    "in the specific domain, verify it, learn to read expressions off the screen, "
    "and end with the most concise library that still solves the benchmark tasks."
)


def plan(n_train: int, epochs: int) -> list[tuple[str, dict]]:
    """The reference sequence, expressed in exactly the tool calls an LLM
    controller would have to make."""
    return [
        ("inspect_machine", {}),
        ("show_catalogue", {}),

        # What "useful" means. Two of these can only be solved by crossing out
        # of the specific domain, which is why a reader has to exist at all.
        ("add_task", {"name": "rewrite", "start": "12*30",
                      "target": "10*30+2*30"}),
        ("add_task", {"name": "value", "start": "12*34", "target": "408"}),
        ("add_task", {"name": "read_screen", "start": "47*83", "target": "47*83",
                      "domain": "specific", "observed": True}),
        ("add_task", {"name": "screen_to_value", "start": "12*30", "target": "360",
                      "domain": "specific", "observed": True}),

        # --- deciding WHAT to form ------------------------------------------
        # BEFORE anything is learned, which is the only point at which the
        # question is real. The machine here has prior symbolic rules and no
        # perception, so a product DRAWN on the screen is unreachable: that is
        # what the proposals have to explain. Run this after the reader exists
        # and every case below comes back solved by
        # `read_expression -> decimal_split -> distribute_symbolic`, the tool
        # proposes nothing, and the step is theatre.
        #
        # The scripted plan does not consume what comes back -- a fixed list of
        # tool calls cannot branch on a result. It is the LLM controller
        # (`--llm`) that reads the proposals and decides. Here they are printed
        # so the two can be compared: what the machine would have chosen,
        # against what this plan goes on to do.
        # The solved side is WORKED INSTANCES of the same regrouping, not table
        # lookups. `9*7 => 63` shares no structure with what is being asked; a
        # product actually split at its place-value boundary does, and each
        # part bottoms out in arithmetic and the 9x9 table, so the machine
        # really can do them. They are posed on the ABSTRACT tape because that
        # is where it can: pose them as drawings and they are unsolved too,
        # leaving an analogy whose solved side is empty and which therefore
        # compares nothing.
        # The solved side is DERIVED, not listed: the machine runs its own
        # trusted chain over generated products, so every worked case arrives
        # with a real derivation and is an instance of exactly the regrouping
        # in question. `solved_expand` carries each part one level further, so
        # the analogy shows the whole tree down to the times table rather than
        # its first row. Two hand-written anchors stay, because a reader wants
        # to see the shape without running anything.
        ("propose_rules", {"unsolved": ["12*30 => 10*30+2*30",
                                        "47*83 => 40*83+7*83"],
                           "solved": ["6*9 => 54", "2*5 => 10"],
                           "solved_via": ["decimal_split", "distribute_symbolic"],
                           "solved_expand": ["decimal_split_right",
                                             "distribute_symbolic_right"],
                           "n_solved": 12,
                           "domain": "specific", "observed": True,
                           "solved_domain": "abstract"}),

        # --- reading the screen: specific -> abstract -----------------------
        # First, because everything else depends on it. A rule that works on
        # pictures cannot be cross-checked against algebra until the machine
        # can get the picture back into symbols.
        ("generate_data", {"generator": "rendered_expressions", "n": n_train,
                           "seed": 7, "name": "read_train",
                           "params": {"max_terms": 1, "digits": 2}}),
        ("label_data", {"dataset": "read_train", "oracle": "read_back"}),
        ("declare_rule", {"name": "read_expression", "domain_in": "specific",
                          "out_domain": "abstract", "num_slots": 6,
                          "description": "read a drawn expression back into symbols"}),
        ("train_rule", {"rule": "read_expression", "dataset": "read_train",
                        "epochs": epochs}),
        ("generate_data", {"generator": "rendered_expressions", "n": 150, "seed": 55,
                           "name": "read_fresh", "params": {"max_terms": 1, "digits": 2}}),
        ("label_data", {"dataset": "read_fresh", "oracle": "read_back"}),
        ("verify_rule", {"rule": "read_expression", "dataset": "read_fresh",
                         "oracle": "read_back", "threshold": 0.91}),

        # --- the distributive rule, learned in the specific domain ----------
        ("generate_data", {"generator": "mul_pairs", "n": n_train, "seed": 1,
                           "name": "mul_train",
                           "params": {"a_digits": 2, "b_digits": 2, "domain": "specific"}}),
        ("label_data", {"dataset": "mul_train", "oracle": "distributive_rewrite"}),
        ("declare_rule", {"name": "distributive_learned", "domain_in": "specific",
                          "out_domain": "specific", "num_slots": 16,
                          "description": "rewrite a*b as a sum of place-value products, "
                                         "learned from drawings"}),
        ("train_rule", {"rule": "distributive_learned", "dataset": "mul_train",
                        "epochs": epochs}),

        # Fresh data, and a tail of 3-digit left factors the rule never saw:
        # the paper's claim is generalisation to unseen and larger integers.
        ("generate_data", {"generator": "mul_pairs", "n": max(120, n_train // 8),
                           "seed": 99, "name": "mul_fresh",
                           "params": {"a_digits": 2, "b_digits": 2, "domain": "specific"}}),
        ("label_data", {"dataset": "mul_fresh", "oracle": "distributive_rewrite"}),
        ("verify_rule", {"rule": "distributive_learned", "dataset": "mul_fresh",
                         "oracle": "distributive_rewrite", "threshold": 0.91}),
        # The independent route: read the picture, do the algebra, draw it
        # again. Agreeing with THAT is evidence about the world rather than
        # about the oracle that trained it -- and it is only available because
        # the reader was verified first.
        ("verify_against_rules", {"rule": "distributive_learned",
                                  "reference_chain": ["read_expression",
                                                      "decimal_split",
                                                      "distribute_symbolic",
                                                      "render"],
                                  "dataset": "mul_fresh", "threshold": 0.91}),
        # Section 3: specialise on what is left over. The ensemble is only
        # created when it beats the base on held-out data -- often it does not,
        # and the run says so. When it is created it lands under
        # 'distributive_learned_ens' and has to be verified like any new rule.
        ("grow_ensemble", {"rule": "distributive_learned", "dataset": "mul_train",
                           "epochs": max(6, epochs // 2)}),

        # --- reasoning as the transfer of mapping rules ---------------------
        ("prove", {"start": "12*30", "target": "10*30+2*30", "max_depth": 5}),
        ("prove", {"start": "12*30", "target": "360", "domain": "specific",
                   "observed": True, "max_depth": 5}),

        # --- what did all that cost? ----------------------------------------
        ("library_report", {}),
        ("simplify_library", {"probe": "mul_fresh", "apply_changes": False}),
        ("finish", {"summary": "distributive rule learned and verified; reader "
                               "learned; library priced and a simplification proposed"}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny run, for smoke testing")
    ap.add_argument("--llm", action="store_true", help="let Claude drive instead")
    ap.add_argument("--device", default=None,
                    help="cuda when a GPU is present; pass cpu to pin it")
    ap.add_argument("--save", default="", help="directory to save the library to")
    ap.add_argument("--dump", default="renders/multiplication",
                    help="directory the drawn expressions are written to, so what "
                         "the nets read and drew can be checked by eye")
    ap.add_argument("--no-dump", action="store_true", help="do not write any images")
    args = ap.parse_args()

    # 3500, not 2000. At 2000 the distributive rule's verified accuracy was a
    # lottery: five training seeds on identical data gave 0.909, 0.957, 0.978,
    # 0.987, 1.000, so whether it cleared its threshold -- and therefore whether
    # any proof could use it -- depended on the seed rather than on the rule.
    # Measured over the same five seeds: 3500 gives 0.939, 0.983, 1.000, 1.000,
    # 1.000 (~90 s per net) and 6000 gives 1.000 five times out of five (~135 s).
    # 3500 is the compromise, and it is why `plan` verifies at 0.91: that floor
    # of 0.939 clears 0.91 on every seed but would have missed 0.95 on one.
    # See also the determinism note in train.py, which removes the run-to-run
    # drift that made the number move even at a fixed seed.
    n_train, epochs = (400, 15) if args.quick else (3500, 60)
    machine = RenMachine(goal=GOAL, device=args.device)

    if args.llm:
        run = LLMController(machine, max_steps=60).run(GOAL)
    else:
        run = ScriptedController(machine).run(plan(n_train, epochs))

    print("\n" + "=" * 78)
    print(run.summary())
    print("\n--- final library ---")
    print(machine.library.table())
    report = machine.report()
    print("\n--- objective ---")
    print(report.summary())

    print("\n--- what the objective concluded ---")
    for name in ("distributive_learned", "distributive_learned_ens", "read_expression"):
        if name in machine.library:
            used = report.used_by.get(name, [])
            print(f"  {name}: {'used by ' + ', '.join(used) if used else 'UNUSED'}"
                  f"  ({machine.library.get(name).cost_bits():.0f} bits, "
                  f"{'trusted' if machine.library.get(name).trusted else 'untrusted'})")
    if report.used_by.get("read_expression"):
        print("  A learned rule that merely rediscovers an identity the machine\n"
              "  already holds does not pay for itself -- two extra domain crossings\n"
              "  cost more than the rewrite saves. A rule that reads the screen has\n"
              "  no symbolic substitute, so it stays. That is the conciseness\n"
              "  objective working, not a failure of the learned rule.")
    else:
        print("  Nothing learned here cleared its verification threshold, so no\n"
              "  proof may use it and everything reads as unused. That is the\n"
              "  expected outcome of --quick: 400 examples and 15 epochs is not\n"
              "  enough to learn to read. Run without --quick for the real result.")

    if args.dump and not args.no_dump:
        # Both learned rules here are about pixels: one reads a drawing, the
        # other rewrites one. Accuracy numbers say how often they were right;
        # these images say what they were looking at when they were not.
        from dynamicmultinets.verify import answer_text, normalize

        out = Path(args.dump)
        if not out.is_absolute():
            out = ROOT / out
        written, first = 0, True

        for ds_name, rule_name in (("read_fresh", "read_expression"),
                                   ("mul_fresh", "distributive_learned"),
                                   ("mul_fresh", "distributive_learned_ens")):
            if ds_name not in machine.datasets or rule_name not in machine.library:
                continue
            rule = machine.library.get(rule_name)

            def cases(ds=machine.datasets[ds_name], rule=rule):
                for ex in ds.examples[:8]:
                    want = answer_text(ex.out) if ex.labeled else "?"
                    got = rule.apply(ex.inp)
                    text = answer_text(got) or "(no answer)"
                    mark = ("unlabeled" if not ex.labeled
                            else "ok" if normalize(text) == normalize(want)
                            else "WRONG")
                    yield f"{mark}_in-{ex.inp.text}_want-{want}_got-{text}", ex.inp.image
                    # A specific -> specific rule answers in pixels; save the
                    # drawing it produced, not just the string behind it.
                    if got is not None and got.image is not None:
                        yield f"{mark}_drawn-{text}", got.image

            written += save_gallery(cases(), str(out), prefix=f"{rule_name}_",
                                    reset=first)
            first = False

        # The proof tape: every expression the chains wrote, as drawn.
        written += save_gallery(((c.text or "cell", c.image) for _, c in machine.specific),
                                str(out), prefix="tape", reset=first)
        print(f"\nwrote {written} images to {out} (captions in {out / 'index.txt'})")

    if args.save:
        print("\nsaved:", machine.save(args.save))


if __name__ == "__main__":
    main()
