"""
Experiment 4: how far the machine gets on the Riemann hypothesis, and where it
stops.

RH is not a cell. It quantifies over the zeros of an analytic function, and
nothing in this package can write that down, let alone search for it. What CAN
be written down is an equivalent: Robin (1984) proved that RH holds if and only
if

    sigma(n) < e^gamma * n * ln ln n     for every n > 5040,

and Lagarias (2002) proved the same for sigma(n) <= H_n + exp(H_n) ln(H_n) over
every n >= 1. Both re-express a statement about zeros as a predicate over the
integers that is EXACTLY DECIDABLE one n at a time -- and an integer, drawn on
the screen, is exactly the kind of thing this machine reasons about.

So the experiment is the machine's own workflow pointed at RH:

  1. it proposes what to form, BEFORE anything is learned, from cases it cannot
     do (a drawn integer, verdict unknown) beside cases it can (the same
     integers as symbols);
  2. it learns to READ integers off the screen -- the one step here that has to
     be learned, because reading pixels is the one thing no symbolic rule does;
  3. it composes that reader with the exact divisor-sum rules into a single
     specific -> abstract rule that takes a DRAWING to a verdict;
  4. it verifies that composite against a DEFINITIONAL oracle, on integers it
     never trained on -- which is the paper's strict limit of an empirical rule:
     an oracle that enumerates divisors is not sampling, it is deciding;
  5. it checks the Robin route against the Lagarias route, two independent
     criteria that must agree above 5040;
  6. it proves individual instances by search, as mixed chains that cross
     domains;
  7. and then it states, as a transfer claim it CANNOT discharge, the step from
     "verified on the integers I was shown" to "holds for every n > 5040".

Step 7 is the result. Everything above it works and is verified; the gap
between a family of exact instances and a universally quantified statement is
not a missing rule or a training budget, and the machine reporting it as an
untested transfer rather than folding it into a chain with a confidence is the
honest behaviour this architecture is built to have.

    python examples/run_riemann.py            # ~4 min, one net
    python examples/run_riemann.py --quick    # ~30 s, weaker reader
    python examples/run_riemann.py --llm      # Claude drives instead
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
    "Decide Robin's criterion for integers drawn on the screen. Learn to read "
    "them, compose the reader with the exact divisor-sum rules you already "
    "have, verify the composite against a definitional oracle on integers you "
    "never trained on, and report honestly what the step from those instances "
    "to all n>5040 would require."
)

# The reader is declared with 6 slots but trained on integers of at most 5
# digits. That is deliberate and it is the same shape as the multiplication
# experiment's 3-digit tail: the rule must be ABLE to express an answer it was
# never shown, so that the transfer to 6-digit integers is a real test of
# generalisation rather than a vocabulary error.
TRAIN_DIGITS, TAIL_DIGITS = 5, 6


def plan(n_train: int, epochs: int) -> list[tuple[str, dict]]:
    """The reference sequence, as the tool calls an LLM controller would make."""
    return [
        ("inspect_machine", {}),
        ("show_catalogue", {}),

        # What "useful" means here. The first is decidable by the rules the
        # machine already has; the second is the same question posed as a
        # DRAWING, which nothing can touch until a reader exists.
        ("add_task", {"name": "robin_symbolic", "start": "10080",
                      "target": "robin_holds"}),
        ("add_task", {"name": "robin_from_screen", "start": "10080",
                      "target": "robin_holds", "domain": "specific",
                      "observed": True}),

        # --- deciding WHAT to form ------------------------------------------
        # Before anything is learned, which is the only point at which the
        # question is real. The unsolved side is integers drawn on the screen;
        # the solved side is integers as symbols, which the divisor-sum chain
        # already settles. The gap between them is perception, and that is what
        # the proposals should be about -- not about number theory, which is
        # already in the library and exact.
        ("propose_rules", {"unsolved": ["10080 => robin_holds",
                                        "720720 => robin_holds"],
                           "solved": ["5041 => robin_holds", "5040 => robin_fails"],
                           "domain": "specific", "observed": True,
                           "solved_domain": "abstract"}),

        # --- learning to read integers --------------------------------------
        # `read_back` is reused rather than replaced: it is the existing
        # constructed oracle for reader supervision, and a reader is the one
        # rule type it is legitimate for (you cannot learn to read without
        # being told what the text said).
        ("generate_data", {"generator": "rendered_integers", "n": n_train,
                           "seed": 7, "name": "read_train",
                           "params": {"min_digits": 1, "max_digits": TRAIN_DIGITS}}),
        ("label_data", {"dataset": "read_train", "oracle": "read_back"}),
        ("declare_rule", {"name": "read_integer", "domain_in": "specific",
                          "out_domain": "abstract", "num_slots": TAIL_DIGITS,
                          "description": "read a drawn integer back into symbols"}),
        ("train_rule", {"rule": "read_integer", "dataset": "read_train",
                        "epochs": epochs}),
        ("generate_data", {"generator": "rendered_integers", "n": 300, "seed": 55,
                           "name": "read_fresh",
                           "params": {"min_digits": 1, "max_digits": TRAIN_DIGITS}}),
        ("label_data", {"dataset": "read_fresh", "oracle": "read_back"}),
        ("verify_rule", {"rule": "read_integer", "dataset": "read_fresh",
                         "threshold": 0.95}),

        # --- the composite: a drawing to a verdict ---------------------------
        # One learned link and three exact ones. The confidence of the whole is
        # the product, so the reader's accuracy is the ONLY thing that can make
        # this composite wrong -- which is the division of labour the paper
        # argues for: nets perceive, known rules compute.
        ("compose_rules", {"rules": ["read_integer", "divisor_sum",
                                     "robin_ratio", "robin_decide"],
                           "new_name": "robin_from_drawing",
                           "description": "decide Robin's criterion for an integer "
                                          "drawn on the screen"}),
        ("compose_rules", {"rules": ["read_integer", "divisor_sum",
                                     "lagarias_decide"],
                           "new_name": "lagarias_from_drawing",
                           "description": "decide Lagarias's criterion for an "
                                          "integer drawn on the screen"}),

        # --- verification against a DEFINITIONAL oracle -----------------------
        # The step that matters, and the one this experiment got wrong first.
        #
        # The obvious test set -- random integers above 5040, labelled by
        # `robin_verdict` -- is VACUOUS. Robin's inequality holds at every n in
        # that range, so 300 cases carry 300 identical labels and a rule that
        # answers "robin_holds" without reading anything scores 1.000. That is
        # not a hypothetical: the first run of this file put a reader with 0.133
        # character accuracy through exactly that set and it came back perfect.
        # It is kept below, AFTER the real test and labelled as the control it
        # is, because a verification number whose base rate is 1.000 is a number
        # about the dataset and not about the rule.
        #
        # `robin_cases` is the honest test: Robin's 26 exceptional integers,
        # which fail, balanced against integers above 5040, which pass. Now the
        # verdict depends on WHICH integer is on the tape, so the composite
        # cannot clear the threshold without the reader actually reading.
        ("generate_data", {"generator": "robin_cases", "n": 60, "seed": 3,
                           "name": "robin_balanced",
                           "params": {"fail_fraction": 0.5,
                                      "max_digits": TRAIN_DIGITS}}),
        ("label_data", {"dataset": "robin_balanced", "oracle": "robin_verdict"}),
        ("verify_rule", {"rule": "robin_from_drawing", "dataset": "robin_balanced",
                         "oracle": "robin_verdict", "threshold": 0.95}),

        # The control. Same rule, same oracle, a family whose labels do not
        # vary; whatever it reports is the base rate.
        ("generate_data", {"generator": "rendered_integers", "n": 300, "seed": 101,
                           "name": "robin_fresh",
                           "params": {"min_digits": 4, "max_digits": TRAIN_DIGITS,
                                      "above": 5040}}),
        ("label_data", {"dataset": "robin_fresh", "oracle": "robin_verdict"}),
        ("verify_rule", {"rule": "robin_from_drawing", "dataset": "robin_fresh",
                         "oracle": "robin_verdict", "threshold": 0.95}),

        # Lagarias's criterion has NO exceptional set: assuming RH it holds at
        # every n>=1, so no balanced test set for it exists to be built. That is
        # worth stating rather than working around -- the criterion is a fine
        # independent route to a verdict and a useless discriminative target,
        # and this call is here to show the difference, not to pass.
        ("verify_rule", {"rule": "lagarias_from_drawing", "dataset": "robin_fresh",
                         "oracle": "lagarias_verdict", "threshold": 0.95}),

        # --- the transfer: integers longer than any it was trained on --------
        # The paper's claim for the distributive rule was generalisation to
        # unseen and larger integers. Here it is the same claim and a harder
        # one, because a 6-digit drawing is a WIDER IMAGE than the reader ever
        # saw, not merely a larger number.
        ("generate_data", {"generator": "rendered_integers", "n": 200, "seed": 202,
                           "name": "robin_tail",
                           "params": {"min_digits": TAIL_DIGITS,
                                      "max_digits": TAIL_DIGITS}}),
        ("label_data", {"dataset": "robin_tail", "oracle": "robin_verdict"}),
        ("verify_rule", {"rule": "robin_from_drawing", "dataset": "robin_tail",
                         "oracle": "robin_verdict", "threshold": 0.95}),

        # The verdict on that tail is vacuous too -- every 6-digit integer
        # passes Robin. What is NOT vacuous is whether the reader can read a
        # drawing wider than any it trained on, so the tail is scored a second
        # time against `read_back`, where the label is the integer itself and
        # varies by construction. This is the honest generalisation number.
        ("label_data", {"dataset": "robin_tail", "oracle": "read_back"}),
        ("verify_rule", {"rule": "read_integer", "dataset": "robin_tail",
                         "oracle": "read_back", "threshold": 0.95}),

        # --- proving instances ------------------------------------------------
        # `observed=True` matters: it stops `transcribe_unsafe` answering by
        # copying the caption, so the proof has to actually read the drawing.
        ("prove", {"start": "10080", "target": "robin_holds", "max_depth": 5}),
        ("prove", {"start": "10080", "target": "robin_holds", "domain": "specific",
                   "observed": True, "max_depth": 6}),
        ("prove", {"start": "5040", "target": "robin_fails", "domain": "specific",
                   "observed": True, "max_depth": 6}),

        # --- what would it take to get the rest? ------------------------------
        # NOT posed as a cell reading "all_n_above_5040". That was the first
        # attempt and it produced a fake proof: the string was rendered, the
        # reader turned the drawing into some digits, the exact chain computed a
        # divisor sum for those digits, and the machine reported the universal
        # claim SOLVED via render -> robin_from_drawing. Nothing in the chain can
        # tell that its input was never an integer, so a nonsense cell comes back
        # with a verdict and a confidence. `report_open_transfer` in main() shows
        # that happening on purpose instead, which is the more useful place for
        # it, and the transfer claim stays where it cannot be faked: in prose,
        # unproved.
        ("propose_rules", {"unsolved": ["27720 => robin_holds"],
                           "solved": ["10080 => robin_holds",
                                      "720720 => robin_holds"],
                           "domain": "specific", "observed": True,
                           "solved_domain": "abstract"}),

        ("library_report", {}),
        ("finish", {"summary": "reader learned and verified; the Robin and Lagarias "
                               "composites verified against definitional oracles on "
                               "unseen integers; the step to all n>5040 left "
                               "explicitly untested"}),
    ]



def report_base_rates(machine) -> None:
    """Each rule's accuracy next to the score of answering with the majority label.

    Both numbers, always, because on this problem they are easy to confuse and
    the gap between them is what decides whether a verification meant anything.
    Above 5040 the Robin verdict never varies, so the base rate is 1.000 and so
    is the accuracy of a rule that reads nothing.

    Datasets are re-labelled here for their OWN oracle before scoring. That is
    not defensive coding: `verify_rule` relabels in place, so the Lagarias
    control leaves `robin_fresh` carrying Lagarias labels, and reading the
    stored labels afterwards reports the wrong experiment.
    """
    from collections import Counter

    from dynamicmultinets import oracles
    from dynamicmultinets.verify import answer_text, normalize

    print("\n--- what each verification could have shown ---")
    print(f"  {'dataset':16}{'rule':22}{'n':>5}{'acc':>8}{'base':>8}  reading")
    for ds_name, rule_name, oracle_name in (
            ("robin_balanced", "robin_from_drawing", "robin_verdict"),
            ("robin_fresh", "robin_from_drawing", "robin_verdict"),
            ("robin_tail", "robin_from_drawing", "robin_verdict"),
            ("robin_fresh", "lagarias_from_drawing", "lagarias_verdict"),
            ("read_fresh", "read_integer", "read_back"),
            ("robin_tail", "read_integer", "read_back")):
        if ds_name not in machine.datasets or rule_name not in machine.library:
            continue
        ds = machine.datasets[ds_name]
        oracles.label(ds, oracle_name)
        rule = machine.library.get(rule_name)
        labels, correct = [], 0
        for ex in ds.examples:
            if not ex.labeled:
                continue
            want = answer_text(ex.out)
            labels.append(want)
            got = rule.apply(ex.inp)
            if got is not None and normalize(answer_text(got)) == normalize(want):
                correct += 1
        if not labels:
            continue
        counts = Counter(labels)
        base = max(counts.values()) / len(labels)
        acc = correct / len(labels)
        reading = ("VACUOUS: one label, a rule that reads nothing also scores 1.000"
                   if base >= 0.999 else
                   f"informative: beats base rate by {acc - base:+.3f}")
        print(f"  {ds_name:16}{rule_name:22}{len(labels):5}{acc:8.3f}{base:8.3f}  {reading}")


def report_open_transfer(machine) -> None:
    """The two things the verified composite still cannot do.

    Both are demonstrated rather than asserted, because both are easy to state
    in a way that sounds like a caveat and is actually the whole result.
    """
    from dynamicmultinets.tapes import Content

    print("\n--- the two gaps, demonstrated ---")

    if "robin_from_drawing" in machine.library:
        rule = machine.library.get("robin_from_drawing")
        # 1. The chain has no idea what it is looking at. Hand it a drawing that
        # is not an integer and it does not decline -- it reads SOMETHING,
        # computes a divisor sum for it, and returns a verdict with the same
        # confidence it reports for a real one.
        for text in ("all_n_above_5040", "zeta(s)=0"):
            got = rule.apply(Content.specific_text(text))
            answer = got.text if got is not None else "(declined)"
            print(f"  non-integer cell {text!r:20} -> {answer}"
                  f"   [confidence {rule.confidence():.3f}]")
        print("     A cell that is not an integer still comes back with a verdict.\n"
              "     Nothing downstream of the reader can tell; this is why the\n"
              "     universal claim must not be posed as a cell for the prover to\n"
              "     reach, which would 'prove' it the same way.")

    # 2. The quantifier. Stated as arithmetic on what was actually checked.
    checked = sum(len(machine.datasets[n].examples)
                  for n in ("robin_balanced", "robin_fresh", "robin_tail")
                  if n in machine.datasets)
    print(f"\n  integers checked by this run: {checked}")
    print("  integers Robin's criterion quantifies over: infinitely many")
    print("     Independent computation has verified RH-equivalent statements far\n"
          "     beyond anything here -- the first 10^13 zeros, exhaustive checks of\n"
          "     Robin's inequality over enormous ranges -- and that is still not a\n"
          "     proof and never becomes one. The machine is on the correct side of\n"
          "     that line and cannot cross it by being run longer.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny run, for smoke testing")
    ap.add_argument("--llm", action="store_true", help="let Claude drive instead")
    ap.add_argument("--device", default=None,
                    help="cuda when a GPU is present; pass cpu to pin it")
    ap.add_argument("--dump", default="renders/riemann",
                    help="directory the drawn integers are written to, so what the "
                         "reader saw when it was wrong can be checked by eye")
    ap.add_argument("--no-dump", action="store_true", help="do not write any images")
    ap.add_argument("--form", choices=["text", "image"], default="text",
                    help="how propose_rules shows the drawn cells: text writes "
                         "each layout out as characters (the default), image "
                         "sends the rendered pixels. Only affects --llm runs")
    args = ap.parse_args()

    n_train, epochs = (400, 15) if args.quick else (4000, 60)
    machine = RenMachine(goal=GOAL, device=args.device, propose_form=args.form)

    if args.llm:
        run = LLMController(machine, max_steps=60).run(GOAL)
    else:
        run = ScriptedController(machine).run(plan(n_train, epochs))

    print("\n" + "=" * 78)
    print(run.summary())
    print("\n--- final library ---")
    print(machine.library.table())

    print("\n--- what was actually established ---")
    for name in ("read_integer", "robin_from_drawing", "lagarias_from_drawing"):
        if name in machine.library:
            r = machine.library.get(name)
            print(f"  {name}: {'trusted' if r.trusted else 'UNTRUSTED'}, "
                  f"confidence {r.confidence():.4f}, {r.stats.summary()}")

    report_base_rates(machine)
    report_open_transfer(machine)

    print("""
--- and what was not ---
  Every verdict above is about an integer that was ON THE TAPE. Robin's
  criterion is the statement that the inequality holds for EVERY n>5040, and no
  number of exact instances reaches it: the machine has checked a finite family
  against a definition, which is evidence of exactly the kind that already
  exists for RH in abundance and has never been a proof of it.

  The final propose_rules call is where that shows up in the machine's own
  terms. A shared-pattern proposal is tested by running the pattern on the
  unknown family; the unknown family here is infinite, so the test does not
  terminate and the proposal cannot be discharged. That is not a limitation of
  the search budget. It is the difference between a rule verified on a family
  and a theorem, and it is the whole distance still to be covered.""")

    if args.dump and not args.no_dump:
        from dynamicmultinets.verify import answer_text, normalize

        out = Path(args.dump)
        if not out.is_absolute():
            out = ROOT / out
        written, first = 0, True
        for ds_name, rule_name in (("read_fresh", "read_integer"),
                                   ("robin_fresh", "robin_from_drawing"),
                                   ("robin_tail", "robin_from_drawing")):
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

            written += save_gallery(cases(), str(out), prefix=f"{ds_name}_{rule_name}",
                                    reset=first)
            first = False
        if written:
            print(f"\nwrote {written} images to {out}")


if __name__ == "__main__":
    main()
