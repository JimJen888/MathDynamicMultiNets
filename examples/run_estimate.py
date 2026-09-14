"""
Experiment 1p: do the granted estimates decompose too?

The chain reaching Theorem 1.1 rests on 25 of the paper's own estimates,
granted rather than proved. The question is whether those are leaves or
just another level: are they themselves established by derivations whose
steps apply known rules?

This tests it on Lemma 7.4, the pulse amplitude estimate, which is one of
the 25 and carries Proposition 7.5 and through it the whole wave
construction. Its proof, in the paper:

  control the ratio of the decaying and growing frame coordinates, then
  convert to the radial and tangential coordinates

which unpacks as

  1  the ratio r = z-/z+ satisfies a Riccati equation with r(0) = 0
  2  at r = +-K/S the sign of r' points inward, so the region is
     invariant and |r| stays bounded
  3  with the ratio bounded, the logarithmic derivative of z+/P is
     O(1/S), so integrating it bounds log(z+/P)
  4  hence z+ never reaches zero and sits between two multiples of the
     envelope
  5  the frame matrix converts z+ and r into the radial and tangential
     components, which is algebra
  6  differentiating the coordinate equation gives every fixed derivative
  7  the pressure formula supplies the extra half power

Five named tools: an invariant-region argument for a scalar differential
inequality, the fundamental theorem of calculus, positivity of a
continuous function that cannot cross zero, linear algebra in the frame,
and differentiating an ODE for higher derivatives.

WHAT THIS SETTLES AND WHAT IT DOES NOT. If the leaves here are named, the
25 are not a floor either, and the recursion keeps descending through
standard mathematics. What it does not settle is where the recursion
stops: each level so far has been named rules plus fewer citations, and
the honest report is the count at each level rather than a claim about the
limit.

Run: python examples/run_estimate.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import Claim, JoinRule, RenMachine            # noqa: E402
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


PRIOR = [
    prior("riccati_form", r"frame_system\((\w+)\)", "ratio_equation({0})",
          "dividing the two frame coordinates turns a linear system into a "
          "scalar Riccati equation for their ratio, with zero initial value"),
    join("invariant_region",
         ["ratio_equation(?T)", "inward_sign(?T)"],
         "ratio_bounded(?T)",
         "if the derivative of a scalar solution points inward at both "
         "edges of an interval containing its initial value, the solution "
         "never leaves it"),
    join("log_derivative_bounded",
         ["ratio_bounded(?T)", "envelope_definition(?T)"],
         "log_ratio_small(?T)",
         "with the ratio bounded, the logarithmic derivative of the growing "
         "coordinate against the envelope is O(1/S)"),
    prior("integrate_the_derivative", r"log_ratio_small\((\w+)\)",
          "log_ratio_bounded({0})",
          "the fundamental theorem of calculus turns a bound on a "
          "derivative over an interval of bounded length into a bound on "
          "the function"),
    prior("cannot_cross_zero", r"log_ratio_bounded\((\w+)\)",
          "growing_coordinate_positive({0})",
          "a continuous function whose logarithm stays finite never "
          "reaches zero, so the coordinate keeps the sign of its initial "
          "value and sits between two multiples of the envelope"),
    join("frame_conversion",
         ["growing_coordinate_positive(?T)", "ratio_bounded(?T)"],
         "components_estimated(?T)",
         "multiplying by the frame matrix expresses the radial and "
         "tangential components in the growing coordinate and the ratio, "
         "which is linear algebra"),
    prior("differentiate_the_ode", r"components_estimated\((\w+)\)",
          "derivatives_estimated({0})",
          "differentiating the coordinate equation and induction on the "
          "order bound every fixed derivative of the solution"),
    join("pulse_estimate",
         ["components_estimated(?T)", "derivatives_estimated(?T)",
          "pressure_formula(?T)"],
         "pulse_amplitude_estimate(?T)",
         "collecting the two-sided envelope bound, the ratio of the "
         "components, the derivative bounds and the extra half power in "
         "the pressure"),
]


STEPS = [
    Claim("H1", "frame_system(T)", (),
          "the homogeneous amplitude system in frame coordinates",
          "the construction's pulse equation, with its stated initial data"),
    Claim("H2", "inward_sign(T)", (),
          "at the edges of the small interval the ratio's derivative "
          "points inward",
          "a sign computation on the Riccati coefficients for large S"),
    Claim("H3", "envelope_definition(T)", (),
          "the envelope, and the growth rate it is defined by",
          "the construction's envelope"),
    Claim("H4", "pressure_formula(T)", (),
          "the homogeneous pressure formula",
          "equation (7.13) at zero forcing"),

    Claim("1", "ratio_equation(T)", ("H1",),
          "the ratio of the two coordinates satisfies a Riccati equation",
          "dividing the system"),
    Claim("2", "ratio_bounded(T)", ("1", "H2"),
          "and stays in a small interval around zero",
          "the region is invariant"),
    Claim("3", "log_ratio_small(T)", ("2", "H3"),
          "so the logarithmic derivative against the envelope is small",
          "substituting the bounded ratio"),
    Claim("4", "log_ratio_bounded(T)", ("3",),
          "and integrating it bounds the logarithm itself",
          "the fundamental theorem of calculus"),
    Claim("5", "growing_coordinate_positive(T)", ("4",),
          "hence the growing coordinate never vanishes and is comparable "
          "to the envelope", "a finite logarithm cannot reach zero"),
    Claim("6", "components_estimated(T)", ("5", "2"),
          "the frame matrix converts this into the two components",
          "linear algebra"),
    Claim("7", "derivatives_estimated(T)", ("6",),
          "and differentiating the equation bounds every fixed derivative",
          "induction on the order"),
    Claim("8", "pulse_amplitude_estimate(T)", ("6", "7", "H4"),
          "which is Lemma 7.4", "collecting"),
]


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    print("=" * 78)
    print("Lemma 7.4: one of the 25 granted estimates, decomposed")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    audit = machine.check_proof(STEPS)
    print("\n--- step by step ---")
    for claim, verdict in zip(STEPS, audit.verdicts):
        origin = ", ".join(claim.premises) if claim.premises else "given"
        print(f"  {claim.key}. [{origin}] {claim.claim[:58]}")
        print(f"      {verdict.status}: {verdict.detail[:60]}")

    print("\n--- the audit ---")
    print(audit.report())

    print("\n--- the count ---")
    print(f"  derived steps: {len(audit.derived)}, "
          f"validated: {len(audit.derived) - len(audit.unvalidated)}")
    print(f"  named rules used: {len(PRIOR)}")
    print(f"  inputs: {len(audit.inputs)} -- the pulse equation, a sign")
    print(f"    computation, the envelope, and the pressure formula")

    print("\n--- reading it ---")
    if not audit.unvalidated:
        print("  So the 25 are not a floor either. One of them decomposes")
        print("  into an invariant-region argument, the fundamental theorem")
        print("  of calculus, a continuity argument, linear algebra and")
        print("  induction on the derivative order. Every one of those is")
        print("  named mathematics.")
        print()
        print("  This has now held at six levels: the imported steps, their")
        print("  proofs, the computations inside those proofs, the order")
        print("  calculus, the construction's own steps, and now one of the")
        print("  estimates they rest on. I have stopped expecting a level")
        print("  where it fails.")
        print()
        print("  What stays true is that each level trades one citation for")
        print("  several, and the recursion ends only at the definitions.")
        print("  The honest report is the count at each level, not a claim")
        print("  about where it stops.")


if __name__ == "__main__":
    main()
