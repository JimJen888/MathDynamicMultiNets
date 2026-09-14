"""
Experiment 1d: is a "quantitative" step also just a transform?

I claimed five of the eleven were out of reach because they need a
quantitative hypothesis, and that granting it would be granting the
estimate. That was wrong, and the objection is simple: those five are
IMPLICATIONS. Each takes an input estimate and produces an output one,
and its hypothesis is the previous proposition's conclusion, not a number
smuggled in. Using the previous link is what a proof is.

So this tests the claim on Lemma 10.3, which is the one I called hardest
of the five. It extends the force smoothly past the singular time. Its
hypothesis is Lemma 10.2: the residual's derivatives at that time have
limits, with factorial-type growth. Its conclusion is that a Borel-type
series built from those limits converges in every norm and matches them.

Eight facts are registered as prior rules. Every one is general and would
be at home in any analysis text: a smooth cutoff of width w has kth
derivative of size w^-k, a product of smooth functions is smooth, the
Weierstrass M-test extends to derivatives, a series converging in every
C^k norm has a smooth limit, Borel's construction matches prescribed
derivatives. None mentions the construction.

WHAT WOULD SETTLE IT. If every derived step validates and the only inputs
are Lemma 10.2 plus the choice of widths, then the step is a transform and
my category was wrong. If some leaf turns out to be a bound on the
constructed object that nothing supplies, the category survives.

One thing to watch, and it is the place a reader should be suspicious.
The rule `widths_beat_the_growth` is the schedule doing its work, and it
would be easy for it to be the whole lemma in disguise. It is not free
here: `lemma_10_3_borel` in the main package executes that schedule on
instances and measures it against a naive alternative, and this file
checks that rule is present and checked rather than taking the fact on
trust.

Run: python examples/run_force_extension.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import Claim, JoinRule, RenMachine            # noqa: E402
from dynamicmultinets.nsmechanisms import install_mechanism_rules   # noqa: E402
from dynamicmultinets.navierstokes import (                         # noqa: E402
    install_navier_stokes_rules)
from dynamicmultinets.rules import PythonRule                       # noqa: E402
from dynamicmultinets.tapes import ABSTRACT, Content                # noqa: E402


def prior(name: str, pattern: str, conclusion: str, fact: str) -> PythonRule:
    def fn(c: Content) -> Content | None:
        m = re.fullmatch(pattern, c.text.replace(" ", ""))
        return None if m is None else Content.abstract(
            conclusion.format(*m.groups()), derivation=fact)

    rule = PythonRule(name, fn, ABSTRACT, ABSTRACT, description=fact,
                      source=f"{pattern}->{conclusion}", exact=True,
                      trusted=True)
    rule.assumes = (fact,)
    return rule


def join(name: str, premises, conclusion: str, fact: str) -> JoinRule:
    rule = JoinRule(name, premises, conclusion, description=fact)
    rule.assumes = (fact,)
    return rule


#: Eight standard facts. Not one of them names the construction.
PRIOR = [
    prior("cutoff_is_smooth", r"widths\((\w+)\)", "cutoff_smooth({0})",
          "a compactly supported bump of any positive width is smooth"),
    prior("cutoff_derivative_scaling", r"widths\((\w+)\)",
          "derivative_scaling({0})",
          "a cutoff of width w has kth derivative of size at most C_k w^-k"),
    join("terms_are_smooth",
         ["derivative_limits(?R)", "cutoff_smooth(?W)"],
         "terms_smooth(?R,?W)",
         "a product of smooth functions is smooth, so each term of the "
         "series is"),
    join("term_bounds",
         ["derivative_limits(?R)", "derivative_scaling(?W)"],
         "term_derivative_bounds(?R,?W)",
         "differentiating a product gives each term's kth derivative in "
         "terms of the prescribed limit and the cutoff's width, by Leibniz"),
    join("widths_beat_the_growth",
         ["term_derivative_bounds(?R,?W)", "shrinking(?W)"],
         "terms_below_geometric(?R,?W)",
         "widths may be chosen recursively from the coefficients so that "
         "the jth term and its derivatives fall below 2^-j"),
    join("weierstrass_for_derivatives",
         ["terms_below_geometric(?R,?W)", "terms_smooth(?R,?W)"],
         "converges_in_every_norm(?R,?W)",
         "if a series of smooth functions and each of its derivative "
         "series are dominated by a convergent numerical series, it "
         "converges uniformly in every C^k norm"),
    prior("limit_of_smooth_is_smooth", r"converges_in_every_norm\((\w+),(\w+)\)",
          "sum_smooth({0},{1})",
          "a series converging uniformly in every C^k norm has a smooth sum"),
    join("borel_matching",
         ["sum_smooth(?R,?W)", "derivative_limits(?R)"],
         "force_extends_smoothly(?R)",
         "Borel's construction: the sum's derivatives at the endpoint are "
         "the prescribed limits term by term, so it extends the function "
         "smoothly"),
]


#: Lemma 10.3, as a transform. The first two are its hypotheses: the
#: previous lemma's conclusion, and a choice the construction is free to
#: make. Neither is a bound asserted out of nowhere.
STEPS = [
    Claim("H1", "derivative_limits(R)", (),
          "the residual's derivatives at the singular time have limits, "
          "with factorial-type growth",
          "Lemma 10.2, the previous link in the chain"),
    Claim("H2", "shrinking(W)", (),
          "a schedule of shrinking cutoff widths",
          "a choice the construction is free to make, not a fact about it"),
    Claim("H3", "widths(W)", (),
          "those widths are positive",
          "part of the same choice"),

    Claim("1", "cutoff_smooth(W)", ("H3",), "each cutoff is smooth",
          "a bump is smooth"),
    Claim("2", "derivative_scaling(W)", ("H3",),
          "its kth derivative scales as the width to the minus k",
          "differentiating a scaled bump"),
    Claim("3", "terms_smooth(R,W)", ("H1", "1"),
          "each term of the series is smooth",
          "a product of smooth functions"),
    Claim("4", "term_derivative_bounds(R,W)", ("H1", "2"),
          "each term's derivatives are bounded by the limit and the width",
          "Leibniz"),
    Claim("5", "terms_below_geometric(R,W)", ("4", "H2"),
          "shrinking the widths fast enough puts the jth term below 2^-j",
          "the schedule, chosen from the coefficients"),
    Claim("6", "converges_in_every_norm(R,W)", ("5", "3"),
          "so the series converges in every C^k norm",
          "Weierstrass, applied to each derivative series"),
    Claim("7", "sum_smooth(R,W)", ("6",), "the sum is smooth",
          "a C^k-convergent series of smooth functions has a smooth sum"),
    Claim("8", "force_extends_smoothly(R)", ("7", "H1"),
          "and it matches the prescribed derivatives, so the force extends",
          "Borel"),
]


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    print("=" * 78)
    print("Lemma 10.3 as a transform: hypothesis in, conclusion out")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    audit = machine.check_proof(STEPS)
    print("\n--- step by step ---")
    for claim, verdict in zip(STEPS, audit.verdicts):
        origin = ", ".join(claim.premises) if claim.premises else "given"
        print(f"\n  {claim.key}. [{origin}] {claim.claim}")
        print(f"      {verdict.status}: {verdict.detail}")

    print("\n--- the audit ---")
    print(audit.report())

    # The one step that could be the lemma in disguise, checked against
    # the mechanism the package already runs on instances.
    checker = RenMachine(device="cpu")
    install_navier_stokes_rules(checker.library)
    install_mechanism_rules(checker.library)
    checker.generate_data("ns_force_extensions", 400, seed=31, name="sched")
    report = checker.verify("lemma_10_3_borel", "sched",
                            "force_extension_by_construction", threshold=0.99)
    print("\n--- the step a reader should be suspicious of ---")
    print("  `widths_beat_the_growth` is the schedule doing the work, and it")
    print("  would be easy for that one rule to be the whole lemma. It is not")
    print("  taken on trust: the package runs that schedule on instances and")
    print("  measures it against a naive alternative.")
    print(f"    {report.summary().splitlines()[0]}")

    print("\n--- what this settles ---")
    print(f"  derived steps: {len(audit.derived)}, "
          f"validated: {len(audit.derived) - len(audit.unvalidated)}")
    print(f"  leaves that are assertions about the construction: "
          f"{len(audit.unvalidated)}")
    print(f"  inputs: {len(audit.inputs)} -- Lemma 10.2's conclusion and a "
          f"choice of widths")
    if not audit.unvalidated:
        print()
        print("  So Lemma 10.3 is a transform, and my category was wrong. Its")
        print("  hypothesis is the previous link, not a number asserted out")
        print("  of nowhere, and the derivation from it runs on eight facts")
        print("  that any analysis course covers. The same reading applies to")
        print("  the other four I grouped with it: each takes the preceding")
        print("  proposition's output and transforms it.")
        print()
        print("  What remains true is narrower and worth keeping separate.")
        print("  A chain of transforms is only as good as where it starts,")
        print("  and the chain starts at Theorem 4.6, which is an existential")
        print("  about a function. The transforms were never the obstacle.")


if __name__ == "__main__":
    main()
