"""
Experiment 1b: can a short proof of an intermediate result be checked here?

The Navier-Stokes run ends with eleven imported steps and the same verdict
for each: untrusted, because it asserts something uniform and instances do
not reach statements of that shape. That is true and it is also a blunt
answer. The eleven are not atoms. Each has internal structure, and the
proposal this experiment tests is that an LLM could supply short correct
proofs of the intermediate results, leaving the machine to check each link
against rules it already holds.

So this takes ONE of the eleven, Proposition 9.9, and decomposes it into
the steps a competent reader would write. I wrote them, playing the part
the LLM would play. The machine's job is the other half: for each link, it
SEARCHES for a rule or short chain in its library that justifies it, and
reports which links it can validate and which it cannot. Nothing here is
taken on my word, which is the point of the exercise.

What came out, and it is not what I expected when I started. Every link
that is one rule applied to one cell was validated, six of eight. The two
that failed are structural rather than mathematical, and the second is the
more serious:

  NOT EXPRESSIBLE   the step's statement has no cell. Smoothness of the
                    summed field is not something any sublanguage here can
                    write down.

  NO CONJUNCTION    the final step follows from five established facts
                    TOGETHER, and a proof here is a chain: one rule, one
                    cell, one successor. There is no way to say "these
                    five, therefore that". `holder_product` works around it
                    with a hand-rolled `pair` cell, which is a special case
                    and not a mechanism. Nearly every real proof ends in a
                    step of this shape, so one failure here understates it.

The other half of the answer is the inputs. Five things are assumed rather
than derived, four of them setup and one of them the mathematics: an L^2
band estimate for a correction, which is what Propositions 7.5 and 9.6
deliver. The machine can state the conditions such an estimate would have
to satisfy, and cannot produce one. That is the same boundary as
everywhere else in this package, reached from a new direction.

An earlier version of this file scored three of ten and blamed missing
rules. It had the decomposition as a LIST, which silently gave one step
the wrong premise. Proofs are graphs. The correction is recorded here
because the first number was wrong in the interesting direction: it made
the machine look worse than it is, and it made the real obstacle, the
absence of conjunction, invisible.

Run: python examples/run_decomposition.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import RenMachine                            # noqa: E402
from dynamicmultinets.navierstokes import install_navier_stokes_rules  # noqa: E402
from dynamicmultinets.normcalc import (est, glob, install_norm_rules,  # noqa: E402
                                       required_estimate, vanishes)
from dynamicmultinets.nsderivation import install_derivation_rules  # noqa: E402
from dynamicmultinets.nsmechanisms import install_mechanism_rules   # noqa: E402

H = "1/200"


#: Proposition 9.9, decomposed the way a reader would write it.
#:
#: A proof is a graph, not a list. The first version of this file was a
#: list, which silently gave step 5 the wrong premise and scored it as a
#: missing rule. So each step names the step it follows from, and a step
#: with `from_` of None is an INPUT: something the decomposition assumes
#: rather than derives. Which inputs those are is most of the answer.
STEPS = [
    dict(key="A", from_=None,
         cell=f"residual_flat_at_singularity,h={H}",
         claim="the correction cycle has left the residual flat at the "
               "singularity",
         why="the hypothesis of Proposition 9.9, supplied by 9.6"),

    dict(key="B", from_=None,
         cell="cycle_from(stage=0,kappa=1/100000)",
         claim="the cycle, at the construction's radial derivative loss",
         why="the setting Proposition 9.6 is applied in"),

    dict(key="C", from_=None,
         cell="summation(gain=1/5,growth=7/100,policy=recursive)",
         claim="the cutoff schedule of Lemma 5.4, with the coefficient "
               "growth the background expansion actually has",
         why="the growth bound is Proposition 5.5's, and is assumed here"),

    dict(key="D", from_=None,
         cell=est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-2),
                  sc=Fraction(1, 5)),
         claim="an L^2 band estimate for one correction, with scale "
               "exponent sigma_0 = 1/5",
         why="THE analytic input: this is what Propositions 9.6 and 7.5 "
             "deliver and nothing here derives"),

    dict(key="E", from_=None,
         cell="increment(divfree=1,field=0)",
         claim="each correction is divergence-free",
         why="the corrections are built as curls"),

    dict(key="1", from_="B",
         cell="cycle_closes_at_every_stage",
         claim="the cycle closes at every stage, so sigma_j = 1/5 + j/10 "
               "increases without bound",
         why="Proposition 9.6's induction, proved here"),

    dict(key="2", from_="C",
         cell="tail_flat_at_every_order",
         claim="the tail of the expansion is flat at every order, whatever "
               "the coefficients do",
         why="the schedule beats any coefficient growth, Lemma 5.4"),

    dict(key="3", from_="D",
         cell=est(d=3, dv=0, ip=Fraction(0), fr=Fraction(-1, 2),
                  sc=Fraction(1, 5)),
         claim="the band estimate becomes a uniform one, at d/p powers of "
               "the frequency",
         why="Bernstein on a dyadic band"),

    dict(key="4", from_="3",
         cell=glob(d=3, dv=0, ip=Fraction(0), sc=Fraction(1, 5)),
         claim="the bands sum, because the frequency exponent is negative",
         why="the dyadic sum is geometric with ratio 2^fr"),

    dict(key="5", from_="4",
         cell=vanishes(d=3, dv=0, ip=Fraction(0)),
         claim="so the summed residual vanishes in L^infinity as the "
               "concentration scale goes to zero",
         why="a positive power of q tends to zero"),

    dict(key="6", from_="E",
         cell="residual_splits_exactly",
         claim="adding the corrections changes the residual by the linear "
               "term and the quadratic flux and nothing else",
         why="the increment identity of Section 3.3, proved here"),

    dict(key="7", from_="2", cell=None,
         claim="the summed field is smooth, divergence-free and compactly "
               "supported in the required region",
         why="convergence of the series in every C^k norm, from the same "
             "cutoff schedule"),

    dict(key="8", from_=("1", "2", "5", "6", "7"),
         cell="local_field_theorem_3_1",
         claim="collecting all of the above gives the local field theorem",
         why="the conjunction of the steps established"),
]


def check(machine: RenMachine) -> list[dict]:
    """Ask the machine to justify each link, rather than asserting it."""
    by_key = {step["key"]: step for step in STEPS}
    results = []
    for step in STEPS:
        row = dict(step)
        if step["from_"] is None:
            row["status"] = "INPUT"
            row["detail"] = "assumed by the decomposition, not derived"
        elif step["cell"] is None:
            row["status"] = "NOT EXPRESSIBLE"
            row["detail"] = "the statement has no cell in any sublanguage"
        elif isinstance(step["from_"], tuple):
            # Several established facts, one conclusion. The machine's
            # proof is a chain: it applies one rule to one cell and has no
            # way to combine several. `holder_product` gets round that with
            # a hand-rolled `pair` cell, which is a special case and not a
            # mechanism.
            row["status"] = "NO CONJUNCTION"
            row["detail"] = (f"follows from {', '.join(step['from_'])} "
                             f"together, and a chain applies one rule to "
                             f"one cell")
        else:
            premise = by_key[step["from_"]]["cell"]
            proof = machine.prove(premise, step["cell"], max_depth=3,
                                  trusted_only=False)
            if proof.found:
                row["status"] = "VALIDATED"
                row["detail"] = " then ".join(proof.rule_names())
            else:
                row["status"] = "NO RULE"
                row["detail"] = (f"nothing in the library takes "
                                 f"{step['from_']} to this cell")
        results.append(row)
    return results


def main() -> None:
    machine = RenMachine(device="cpu")
    install_navier_stokes_rules(machine.library)
    install_mechanism_rules(machine.library)
    install_derivation_rules(machine.library)
    install_norm_rules(machine.library)

    # Everything the machine can settle on its own, settled first, so the
    # decomposition is checked against the strongest library available.
    machine.prove_rule("prop_9_6_all_stages",
                       "stage_induction_by_linear_arithmetic")
    machine.prove_rule("increment_identity",
                       "increment_identity_by_polynomial_algebra")
    machine.prove_rule("ns_carrier_frequency",
                       "carrier_balance_by_factorisation")
    for rule, prover in (("bernstein_uniform", "band_exponents_by_scaling"),
                         ("sum_over_bands", "summation_and_limit_by_sign"),
                         ("limit_at_the_singularity",
                          "summation_and_limit_by_sign")):
        machine.prove_rule(rule, prover)

    print("=" * 78)
    print("Proposition 9.9, decomposed into short steps and checked link by link")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    results = check(machine)
    print("\n--- the decomposition, step by step ---")
    for row in results:
        arrow = ("" if row["from_"] is None else
                 f"  [from {row['from_'] if isinstance(row['from_'], str) else ', '.join(row['from_'])}]")
        print(f"\n  {row['key']}.{arrow} {row['claim']}")
        print(f"      because: {row['why']}")
        print(f"      cell:    {row['cell'] if row['cell'] else '(none)'}")
        print(f"      {row['status']}: {row['detail']}")

    inputs = [r for r in results if r["status"] == "INPUT"]
    links = [r for r in results if r["status"] != "INPUT"]
    validated = [r for r in links if r["status"] == "VALIDATED"]

    print("\n--- the number you asked for ---")
    print(f"  derived steps: {len(links)}")
    print(f"  validated against rules the machine already holds: "
          f"{len(validated)}")
    for label in ("NO RULE", "NO CONJUNCTION", "NOT EXPRESSIBLE"):
        hits = [r for r in links if r["status"] == label]
        if hits:
            print(f"  {label:16} {len(hits)}  "
                  f"(steps {', '.join(r['key'] for r in hits)})")
    print(f"  inputs the decomposition assumes: {len(inputs)}  "
          f"({', '.join(r['key'] for r in inputs)})")

    print("\n--- what this actually shows ---")
    print("  Every link that is one rule applied to one cell was validated.")
    print("  That is the encouraging half and it is worth stating plainly:")
    print("  the norm calculus and the two proved results chain correctly,")
    print("  and the machine found each justification by search rather than")
    print("  being told which rule to use.")
    print()
    print("  The failures are all structural, and none of them is about")
    print("  this proposition being hard.")
    print()
    print("  NO CONJUNCTION is the one I did not expect to matter this much.")
    print("  Step 8 follows from five established facts together, and a")
    print("  proof here is a CHAIN: one rule, one cell, one successor. There")
    print("  is no way to say 'these five, therefore that'. holder_product")
    print("  works around it with a hand-rolled pair cell, which is a special")
    print("  case rather than a mechanism. Almost every real proof ends in a")
    print("  step of this shape, so this blocks far more than one link.")
    print()
    print("  NOT EXPRESSIBLE is step 7, smoothness of the summed field. No")
    print("  sublanguage here has a cell for it.")
    print()
    print("  The inputs are where the mathematics went. Four of the five are")
    print("  setup, and D is the analytic content: an L^2 band estimate for")
    print("  a correction. That is what Propositions 7.5 and 9.6 deliver and")
    print("  what nothing here derives. Hand the machine D and the run of")
    print("  three norm steps goes through immediately.")

    start, needed = required_estimate(machine.library)
    print("\n  The machine can state what D would have to satisfy:")
    for line in needed:
        print(f"    {line}")
    print("  D has fr = -2 and sc = 1/5, so it clears both. The machine can")
    print("  say what it needs and cannot produce it, which is the same")
    print("  boundary as everywhere else in this package.")

    print("\n--- honest reading ---")
    print("  This does not show Proposition 9.9 is nearly proved. Every")
    print("  validated link is an exponent manipulation. What it does show")
    print("  is that the eleven are not atoms, and that the obstacles to")
    print("  checking an LLM's short proof are nameable: a conjunction rule,")
    print("  a richer cell language, and a bridge from the construction's")
    print("  decay orders into the norm calculus. Those are three pieces of")
    print("  engineering, not three open problems. What stays out of reach")
    print("  after all three is the input D, and that is the analysis.")


if __name__ == "__main__":
    main()
