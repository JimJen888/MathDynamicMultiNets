"""
Experiment 1m: Proposition 9.5, the first front-half step, on the apparatus.

Two turns ago I stopped here, because this proof names no public theorem
and reasons entirely in the construction's own classes. The objection was
half right and half wrong. Half right: it cannot be decomposed into
Cauchy-Schwarz and Gronwall, because it does not use them. Half wrong:
the calculus it does use is Proposition 6.6, a stated result with a proof,
now registered in `apparatus.py`.

With that in place this proof decomposes like any other, and three of its
steps are not citations but DERIVATIONS the machine performs:

  the nonlinear wave residual   two primary amplitudes at order 1/2
                                multiply to order 1, and the radial
                                derivative costs kappa_s, giving 1 - kappa_s
                                -- which is the exponent the paper states

  the tangential means          the cross interaction sits at 3/2 - kappa_s
                                and the axial terms at 2 - kappa_s, so the
                                sum sits at the weaker, 1.49999, which is
                                why the paper can quote 1.49

  the stage-zero orders         every achieved exponent is compared against
                                what a stage-zero state requires, exactly

The rest are the paper's own citations: the mean balances (8.3), the
temporal update (8.20), the five-equation correction (8.25), and
Proposition 7.5 for the covariance that cancels the order-zero stress.

WHAT THIS DOES NOT DO. It checks the bookkeeping, not the estimates behind
each cited exponent. When the paper says the axial fluxes have exponent at
least 2 - kappa_s, that is an estimate this file takes as given. What the
machine now verifies is that the exponents COMBINE as claimed and that the
result clears the threshold, which is the part the proof spends its words
on and the part most likely to hide an arithmetic slip.

Run: python examples/run_initialize.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import Claim, RenMachine                      # noqa: E402
from dynamicmultinets.apparatus import (KAPPA_S, clears,             # noqa: E402
                                        install_apparatus, mean, pair, wave)

#: What Definition 9.4 requires of a state at stage zero.
B0 = Fraction(7, 10)
C0 = Fraction(6, 5)

#: The exponents the paper asserts, each as a citation rather than a
#: derivation. These are the estimates; the arithmetic on them is not.
CITED = [
    ("primary amplitudes at order 1/2", wave(Fraction(1, 2), 1),
     "the construction's primary transverse amplitudes"),
    ("the mean balances (8.3)", mean(1 - KAPPA_S),
     "g_r, p_m, E_theta and E_z lie in M at order 1 - kappa_s before any "
     "mean update"),
    ("the covariance cancellation", mean(Fraction(3, 2) - KAPPA_S),
     "Proposition 7.5's covariance cancels the order-zero stress exactly, "
     "leaving the cross interaction between the curl correction and a "
     "primary amplitude at 3/2 - kappa_s"),
    ("the axial terms", mean(2 - KAPPA_S),
     "axial fluxes, the axial pressure term and the higher-order auxiliary "
     "stress have exponent at least 2 - kappa_s"),
    ("the temporal update (8.20)", mean(2 - 2 * KAPPA_S),
     "after the update every residual change has exponent at least "
     "H_0 + 1 - 2 kappa_s"),
    ("the five-equation correction (8.25)", mean(Fraction(9, 5) - KAPPA_S),
     "every term not cancelled by the system gains more than 0.8 above "
     "H_0, so the remaining defects exceed 1.8 - kappa_s"),
]


#: The same proof as a claim list, so it can join the chain in
#: `run_complete.py`. The inline checks below derive the three exponents;
#: these are the same steps with their premises named.
STEPS = [
    Claim("H1", wave(Fraction(1, 2), 1), (),
          "the primary transverse amplitudes, at order 1/2",
          "CITED estimate: the construction's primary amplitudes"),
    Claim("H2", mean(Fraction(3, 2) - KAPPA_S), (),
          "the cross interaction, after the covariance cancellation",
          "CITED estimate: Proposition 7.5's covariance"),
    Claim("H3", mean(2 - KAPPA_S), (),
          "the axial fluxes, pressure term and higher-order stress",
          "CITED estimate: the construction's axial bounds"),
    Claim("H4", mean(Fraction(9, 5) - KAPPA_S), (),
          "the defects after the five-equation correction",
          "CITED estimate: (8.25)"),

    Claim("1", wave(Fraction(1), 2), ("H1", "H1"),
          "two primary amplitudes multiply to order one",
          "the product law of Proposition 6.6"),
    Claim("2", wave(1 - KAPPA_S, 2), ("1",),
          "and the radial derivative costs kappa_s, which is the "
          "nonlinear wave residual the paper states",
          "the operator table (6.32)"),
    Claim("3", mean(Fraction(3, 2) - KAPPA_S), ("H2", "H3"),
          "the tangential means sit at the weaker of the two orders",
          "a sum across orders sits at the smallest"),
    Claim("4", "state_at_stage_zero", ("2", "3", "H4"),
          "every achieved order clears what stage zero requires, so this "
          "is a state at stage zero with B0 = 0.7 and C0* = 1.2",
          "Definition 9.4"),
]


def make_stage_zero_rule():
    """The final comparison, as a rule rather than a printed check."""
    from dynamicmultinets import JoinRule

    rule = JoinRule(
        "clears_stage_zero",
        [wave(1 - KAPPA_S, 2), mean(Fraction(3, 2) - KAPPA_S),
         mean(Fraction(9, 5) - KAPPA_S)],
        "state_at_stage_zero",
        description="the wave residual clears B0 = 7/10 and both mean "
                    "orders clear C0* = 6/5, so Definition 9.4 is met")
    rule.assumes = ("Definition 9.4: a state at stage zero requires the "
                    "wave residual at B0 = 7/10 and the mean orders at "
                    "C0* = 6/5, and each achieved order clears its "
                    "threshold exactly",)
    return rule


PRIOR = [make_stage_zero_rule()]


def main() -> None:
    machine = RenMachine(device="cpu")
    install_apparatus(machine.library)

    print("=" * 78)
    print("Proposition 9.5 on the order calculus")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    print("\n--- what is cited: the estimates ---")
    for label, cell, why in CITED:
        machine.assume(cell, why, source="the paper", standing="estimate")
        print(f"  {label:34} {cell}")

    print("\n--- what is derived: the bookkeeping ---")
    product = machine.library.get("class_product")
    weakest = machine.library.get("weakest_term")
    radial = machine.library.get("radial_derivative")

    from dynamicmultinets.tapes import Content

    amplitude = wave(Fraction(1, 2), 1)
    squared = product.apply(Content.abstract(pair(amplitude, amplitude)))
    residual = radial.apply(squared)
    print(f"  nonlinear wave residual: {amplitude} squared is {squared.text},")
    print(f"    then the radial derivative gives {residual.text}")
    print(f"    the paper states 1 - kappa_s = {1 - KAPPA_S}: "
          f"{'matches' if residual.text == wave(1 - KAPPA_S, 2) else 'DIFFERS'}")

    means = weakest.apply(Content.abstract(
        pair(mean(Fraction(3, 2) - KAPPA_S), mean(2 - KAPPA_S))))
    print(f"\n  tangential means: the weaker of 3/2 - kappa_s and 2 - kappa_s")
    print(f"    is {means.text}, so the paper's quoted 1.49 is clear by "
          f"{'yes' if clears(means.text, Fraction(149, 100)) else 'NO'}")

    print(f"\n  stage-zero requirements, B0 = {B0} and C0* = {C0}:")
    checks = [
        ("the wave residual", residual.text, B0),
        ("the tangential means", mean(Fraction(149, 100)), C0),
        ("the residual changes", mean(2 - 2 * KAPPA_S), C0),
        ("the remaining defects", mean(Fraction(9, 5) - KAPPA_S), C0),
    ]
    ok = True
    for label, cell, need in checks:
        good = clears(cell, need)
        ok = ok and good
        print(f"    {label:24} {cell:22} clears {need}: "
              f"{'yes' if good else 'NO'}")

    print("\n--- the result ---")
    print(f"  every achieved order clears what stage zero requires: "
          f"{'yes' if ok else 'no'}")
    print(f"  derived rather than cited: 3 of the proof's steps")
    print(f"  cited estimates: {len(CITED)}")

    print("\n--- reading it honestly ---")
    print("  The bookkeeping checks out and the estimates behind it do not")
    print("  come from here. When the paper says the axial terms sit at")
    print("  2 - kappa_s, this file takes that as given; what it verifies is")
    print("  that those exponents combine as claimed and clear the")
    print("  threshold. That is the part the proof spends its words on and")
    print("  the part where an arithmetic slip would hide.")
    print()
    print("  Which is a weaker result than the Section 10 decompositions and")
    print("  worth saying so plainly. There, the cited facts were theorems")
    print("  anyone can check. Here they are the construction's own")
    print("  estimates, and six of them carry this step. The apparatus made")
    print("  the proof expressible; it did not make the estimates cheap.")


if __name__ == "__main__":
    main()
