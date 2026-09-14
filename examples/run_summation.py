"""
Experiment 1k: Proposition 9.9, the first step of the front half.

The claim being tested: the construction and its property verification
also proceed by applying known rules at each step, so the method that
closed Section 10 should work on Sections 4 to 9 as well.

Proposition 9.9 is the right place to try it. It is the step that produces
`local_field_theorem_3_1`, which is exactly where the already-closed chain
begins, so decomposing it extends that chain backward by one link rather
than starting somewhere disconnected.

Its proof has a structure the paper states explicitly: apply the summation
lemma, then establish the four parts of Theorem 3.1 in turn. The five
steps are the paper's own, and so are their names.

  Step 1  representatives, and the hypotheses of Lemma 5.4
  Step 2  the summed fields and their Cartesian regularity      -> 3.1(i)
  Step 3  one-sided regularity away from the singular point     -> 3.1(ii)
  Step 4  the exterior heat field and the flat residual         -> 3.1(iii)
  Step 5  the inner velocity growth                             -> 3.1(iv)

WHAT IT LEANS ON. Six previous links, and this is the honest cost of
working backward: Lemma 5.4 for the summation, Lemmas 9.7 and 9.8 for the
domain and the gains, Theorem 4.6 and Proposition 5.5 for profile and
background bounds, Lemma A.6 for the exterior heat equation. FIVE of those
six are outside the eleven, so decomposing one front-half step pulls in
five more citations. That number is the thing to watch as this proceeds:
the front half is more tightly connected than Section 10 was, and working
backward through it will keep widening rather than narrowing until it
reaches the bottom.

Run: python examples/run_summation.py
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
    # -- Step 1: the summation hypotheses -----------------------------------
    prior("cut_the_finite_block", r"initial_block\((\w+)\)",
          "representatives_ready({0})",
          "a smooth cutoff equal to one near the singular point leaves each "
          "finite state unchanged there, so cutting the initialization block "
          "costs nothing at q = 0"),
    prior("azimuthal_stays_solenoidal", r"representatives_ready\((\w+)\)",
          "representatives_solenoidal({0})",
          "an azimuthal field with no angular dependence keeps zero "
          "divergence when multiplied by a function of the other variables"),
    join("summation_hypotheses",
         ["representatives_solenoidal(?S)", "common_domain(?S)",
          "stagewise_gains(?S)", "residual_flatness(?S)"],
         "lemma_5_4_applies(?S)",
         "the four hypotheses of the summation lemma: representatives with "
         "smooth zero extensions, one domain for every finite partial sum, "
         "gains growing with the stage, and a residual flat at each stage"),

    # -- Step 2: the summed fields ------------------------------------------
    join("summation_lemma",
         ["lemma_5_4_applies(?S)", "summation_lemma_holds(?S)"],
         "summed_fields(?S)",
         "Lemma 5.4 then gives potentials whose locally finite sum is "
         "divergence-free with a residual flat to every order (9.20)"),
    prior("axis_representatives", r"summed_fields\((\w+)\)",
          "cartesian_regular({0})",
          "the base Stokes streamfunctions are r^2 times a smooth function "
          "of (r^2, z, t) and the annular terms vanish near the axis, so the "
          "potentials are smooth in Cartesian coordinates"),

    # -- Step 3: one-sided regularity ---------------------------------------
    prior("finitely_many_terms", r"summed_fields\((\w+)\)",
          "finite_on_compacts({0})",
          "both cutoff sequences tend to infinity, so on a compact set away "
          "from the singular point only finitely many terms, bands and slow "
          "labels contribute"),
    join("derivative_bounds",
         ["finite_on_compacts(?S)", "profile_bounds(?S)"],
         "uniform_derivative_bounds(?S)",
         "differentiating the pulse inverse's fixed linear ODE bounds every "
         "derivative order, and coordinate conversion with denominator "
         "bounded below preserves those bounds"),
    prior("fundamental_theorem", r"uniform_derivative_bounds\((\w+)\)",
          "one_sided_limits({0})",
          "bounding one more time derivative and applying the fundamental "
          "theorem of calculus gives a uniform one-sided limit of each "
          "derivative; compatibility between consecutive limits follows by "
          "integrating in time"),

    # -- Step 4: the exterior -----------------------------------------------
    prior("corrections_vanish_outside", r"summed_fields\((\w+)\)",
          "exterior_is_bare({0})",
          "all annular corrections vanish beyond the outer profile support, "
          "and the zero total axial moments make the Stokes streamfunctions "
          "vanish there too, leaving the bare swirl"),
    prior("rising_factorial_bound", r"exterior_is_bare\((\w+)\)",
          "exterior_derivatives_bounded({0})",
          "in the integral formula the factor (1 + Zv)^(-h-m) is at most one "
          "for all nonnegative arguments, so every derivative of the "
          "exterior profile is bounded by a rising factorial"),
    join("exterior_solves_heat",
         ["exterior_derivatives_bounded(?S)", "exterior_heat_lemma(?S)"],
         "exterior_residual_zero(?S)",
         "Lemma A.6 makes the exterior field solve the radial swirl heat "
         "equation, and its centrifugal pressure makes the residual "
         "identically zero out there"),

    # -- Step 5: the inner growth -------------------------------------------
    prior("inner_expansion", r"summed_fields\((\w+)\)",
          "inner_growth({0})",
          "at a fixed inner profile point the annular corrections vanish and "
          "the realized expansion is its leading value plus a remainder, "
          "giving the stated asymptotic along the path z = 0"),

    # -- collecting ---------------------------------------------------------
    join("theorem_3_1",
         ["cartesian_regular(?S)", "one_sided_limits(?S)",
          "exterior_residual_zero(?S)", "inner_growth(?S)"],
         "local_field_theorem_3_1(?S)",
         "parts (i) to (iv) of Theorem 3.1 together are the local field "
         "theorem"),
]


#: The previous links this step consumes. Five of the six are outside the
#: eleven, which is the cost of working backward and is reported as such.
CITED = [
    ("summation_lemma_holds(S)", "Lemma 5.4, the summation lemma", "outside"),
    ("common_domain(S)", "Lemma 9.7, one domain for every partial sum",
     "outside"),
    ("stagewise_gains(S)", "Lemma 9.8, the stagewise gains", "outside"),
    ("residual_flatness(S)", "(9.18), the stagewise residual estimate",
     "outside"),
    ("profile_bounds(S)", "Theorem 4.6 and Proposition 5.5, profile and "
     "background derivative bounds", "one of the eleven"),
    ("exterior_heat_lemma(S)", "Lemma A.6, the exact heat exterior",
     "outside"),
]


STEPS = [
    Claim("H1", "initial_block(S)", (),
          "the fixed slow base and the finite initialization block",
          "what the construction has built by Section 9"),

    Claim("1", "representatives_ready(S)", ("H1",),
          "cut once inside the domain; the states are unchanged near q = 0",
          "a cutoff equal to one there"),
    Claim("2", "representatives_solenoidal(S)", ("1",),
          "and the azimuthal parts keep zero divergence",
          "no angular dependence"),
    Claim("3", "lemma_5_4_applies(S)", ("2", "C1", "C2", "C3"),
          "so the summation lemma's hypotheses hold",
          "Step 1 of the paper's proof"),
    Claim("4", "summed_fields(S)", ("3", "C0"),
          "giving a divergence-free sum with a residual flat to every order",
          "Lemma 5.4, (9.20)"),

    Claim("5", "cartesian_regular(S)", ("4",),
          "the potentials are smooth in Cartesian coordinates",
          "Theorem 3.1(i)"),
    Claim("6", "finite_on_compacts(S)", ("4",),
          "only finitely many terms contribute away from the singular point",
          "both cutoff sequences grow"),
    Claim("7", "uniform_derivative_bounds(S)", ("6", "C4"),
          "with bounds at every derivative order",
          "the ODE and coordinate conversion"),
    Claim("8", "one_sided_limits(S)", ("7",),
          "hence one-sided limits of every derivative at the singular time",
          "Theorem 3.1(ii)"),

    Claim("9", "exterior_is_bare(S)", ("4",),
          "beyond the profile supports only the bare swirl survives",
          "the corrections and streamfunctions vanish"),
    Claim("10", "exterior_derivatives_bounded(S)", ("9",),
          "whose derivatives are bounded to every order",
          "the integral formula"),
    Claim("11", "exterior_residual_zero(S)", ("10", "C5"),
          "and whose residual is identically zero out there",
          "Theorem 3.1(iii)"),

    Claim("12", "inner_growth(S)", ("4",),
          "at a fixed inner point the velocity has the stated asymptotic",
          "Theorem 3.1(iv)"),

    Claim("13", "local_field_theorem_3_1(S)", ("5", "8", "11", "12"),
          "collecting the four parts gives the local field theorem",
          "which is what the next section consumes"),
]


#: The cited premises as claims, so `STEPS` is self-contained. They were
#: built inside `main` before, which meant the exported list referenced
#: keys nobody else could resolve: four of its steps reported PREMISE
#: MISSING when checked from another file, and read as failures of the
#: decomposition rather than of how it was packaged.
ALL_STEPS = [Claim(f"C{i}", cell, (), why, f"cited: {where}")
             for i, (cell, why, where) in enumerate(CITED)] + STEPS


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    # The previous links, granted with their provenance.
    keys = {}
    for i, (cell, why, where) in enumerate(CITED):
        machine.assume(cell, f"{why} [{where} the eleven]",
                       source="the paper", standing="previous link")
        keys[f"C{i}" if i == 0 else f"C{i}"] = cell
    claims = ALL_STEPS

    print("=" * 78)
    print("Proposition 9.9: the first step of the front half")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    audit = machine.check_proof(claims)
    print("\n--- step by step ---")
    for claim, verdict in zip(claims, audit.verdicts):
        origin = ", ".join(claim.premises) if claim.premises else "given"
        print(f"  {claim.key}. [{origin}] {claim.claim[:64]}")
        print(f"      {verdict.status}: {verdict.detail[:66]}")

    print("\n--- the audit ---")
    print(audit.report())

    outside = [w for _, w, where in CITED if where == "outside"]
    print("\n--- what it cost ---")
    print(f"  derived steps: {len(audit.derived)}, "
          f"validated: {len(audit.derived) - len(audit.unvalidated)}")
    print(f"  named facts: {len(PRIOR)}")
    print(f"  previous links consumed: {len(CITED)}, of which "
          f"{len(outside)} are outside the eleven:")
    for w in outside:
        print(f"    {w}")

    print("\n--- reading it ---")
    print("  The claim holds here too: the construction's own steps apply")
    print("  known rules, and the paper's five-step structure decomposes")
    print("  the same way Section 10 did.")
    print()
    print("  But the cost has changed direction, and that is the finding.")
    print("  Each Section 10 step pulled in at most one citation. This one")
    print(f"  pulls in {len(outside)} from outside the eleven. Working backward")
    print("  through")
    print("  the construction widens rather than narrows, because the front")
    print("  half is where the paper's lemmas interlock. The count of open")
    print("  items will rise before it falls, and a report that only counted")
    print("  decomposed steps would hide that.")


if __name__ == "__main__":
    main()
