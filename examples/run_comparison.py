"""
Experiment 1e: Lemma 10.5, the one I said had no representation at all.

I classified this step as out of reach because it "turns on a pressure
flux controlled by Riesz transforms on the whole space, and nothing here
models that". That was a claim about the machine made without reading the
proof. Having now read it, the objection was wrong in the same way the
others were: Lemma 10.5 is an implication, from hypotheses supplied by
Lemma 10.3 and equation (1.1), through standard harmonic analysis, to
v = u.

This decomposition follows the paper's own proof, section by section, and
the section names are its: pressure gradient, pressure flux, difference
energy. Fourteen facts are registered as prior rules and every one is a
named theorem of analysis -- Riesz boundedness, Sobolev, Young in both its
forms, Hoelder, Gronwall, the Liouville argument for a distribution
supported at the origin. None mentions the construction.

WHERE TO BE SUSPICIOUS. Two steps do real work and could hide the lemma.
The commutator kernel bound is a computation the paper carries out, and
here it is a registered fact rather than something derived, so it is on
the assumption list under its own name. And "every power of A_R is below
two, so Young absorbs them" is the hinge of the whole energy estimate; it
is also registered rather than derived, and a reader should check the
exponents in the paper rather than take it from this file.

WHAT WOULD SETTLE IT. Every derived step validating, with the only inputs
being the hypotheses the lemma is stated with.

Run: python examples/run_comparison.py
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


#: Fourteen facts. Every one is a named theorem or a stated computation
#: from the paper, and not one names the construction.
PRIOR = [
    join("subtract_the_equations",
         ["solves_1_1(?v)", "solves_1_1(?u)", "same_force(?v,?u)"],
         "difference_equation(?v,?u)",
         "subtracting two solutions with the same force cancels it, leaving "
         "(10.15): d_t w + (v.grad)w + (w.grad)u = lap w - grad pi, div w = 0"),

    # -- pressure gradient --------------------------------------------------
    prior("flux_is_integrable", r"difference_equation\((\w+),(\w+)\)",
          "flux_in_L1({0},{1})",
          "with w in L^2 and u bounded, g = v (x) v - u (x) u is in L^1 "
          "uniformly in time"),
    prior("fourier_of_L1_is_bounded", r"flux_in_L1\((\w+),(\w+)\)",
          "riesz_pressure_in_negative_sobolev({0},{1})",
          "the Fourier transform of an L^1 function is bounded, so the "
          "double Riesz transform of g lies in H^-s for every s > 3/2"),
    prior("multiplier_identity", r"flux_in_L1\((\w+),(\w+)\)",
          "riesz_pressure_solves_poisson({0},{1})",
          "the multiplier -xi_i xi_j / |xi|^2 gives lap pi* = - sum d_i d_j "
          "g_ij, the value at the origin being irrelevant"),
    prior("divergence_of_the_difference", r"difference_equation\((\w+),(\w+)\)",
          "true_pressure_solves_poisson({0},{1})",
          "taking the divergence of the difference equation gives the same "
          "Poisson equation for the true pressure"),
    join("harmonic_and_tempered_is_zero",
         ["riesz_pressure_solves_poisson(?v,?u)",
          "true_pressure_solves_poisson(?v,?u)",
          "riesz_pressure_in_negative_sobolev(?v,?u)"],
         "pressure_gradients_agree(?v,?u)",
         "the difference of the two pressure gradients is harmonic and lies "
         "in H^-3; its Fourier transform is a weighted L^2 function "
         "supported at the origin, hence zero. This fixes the pressure "
         "gradient with no growth condition on P"),

    # -- pressure flux ------------------------------------------------------
    prior("sobolev_on_the_cutoff", r"difference_equation\((\w+),(\w+)\)",
          "sobolev_bound({0},{1})",
          "the Sobolev inequality and the product rule give (10.17): "
          "B_R <= C (A_R + R^-1 ||w||_2)"),
    prior("riesz_is_bounded_on_Lp", r"flux_in_L1\((\w+),(\w+)\)",
          "riesz_part_bounded({0},{1})",
          "Riesz transforms are bounded on L^p for 1 < p < infinity, which "
          "bounds the first sum of (10.18) in L^3/2"),
    prior("commutator_kernel", r"flux_in_L1\((\w+),(\w+)\)",
          "commutator_part_bounded({0},{1})",
          "the commutator kernel is bounded by C |x-y|^-3 min(|x-y|/R, 1), "
          "whose L^4/3 norm is at most C R^-1; Young's convolution "
          "inequality then bounds the second sum of (10.18) by C_T R^-3/4"),
    join("pressure_flux_bound",
         ["pressure_gradients_agree(?v,?u)", "riesz_part_bounded(?v,?u)",
          "commutator_part_bounded(?v,?u)", "sobolev_bound(?v,?u)"],
         "pressure_flux_bounded(?v,?u)",
         "pairing the two parts of (10.18) against C R^-1 |w| on the "
         "annulus, with interpolation of the L^3 and L^4 norms of w, gives "
         "(10.19)"),

    # -- difference energy --------------------------------------------------
    join("energy_pairing_with_cutoff",
         ["difference_equation(?v,?u)", "sobolev_bound(?v,?u)"],
         "cutoff_energy_identity(?v,?u)",
         "pairing the difference equation with chi_R w gives an identity "
         "for E_R' with a stretching term and three boundary fluxes"),
    prior("transport_flux", r"cutoff_energy_identity\((\w+),(\w+)\)",
          "transport_flux_bounded({0},{1})",
          "on the support of grad chi_R the two fields agree, and Hoelder "
          "bounds the transport flux by C_T R^-1 B_R^3/2"),
    join("young_absorbs_the_fluxes",
         ["cutoff_energy_identity(?v,?u)", "transport_flux_bounded(?v,?u)",
          "pressure_flux_bounded(?v,?u)"],
         "differential_inequality(?v,?u)",
         "every power of A_R appearing in the flux bounds is at most 3/2, "
         "below 2, so Young's inequality absorbs them into the dissipation "
         "and leaves E_R' <= 2 ||grad u||_inf E_R + C_T / R"),
    join("gronwall_from_rest",
         ["differential_inequality(?v,?u)", "zero_initial_difference(?v,?u)"],
         "energy_vanishes_as_R_grows(?v,?u)",
         "Gronwall with E_R(0) = 0 and ||grad u||_inf bounded on [0,T] gives "
         "E_R(t) <= C_T' / R"),
    prior("exhaust_the_space", r"energy_vanishes_as_R_grows\((\w+),(\w+)\)",
          "fields_agree({0},{1})",
          "chi_R is one on each fixed ball for all large R, so letting R go "
          "to infinity gives w = 0 throughout the interval"),
]


#: Lemma 10.5, following the paper's proof. The first four are its
#: hypotheses: what it is stated to assume.
STEPS = [
    Claim("H1", "solves_1_1(v)", (),
          "v is a smooth solution of (1.1) at viscosity one on [0,T]",
          "hypothesis of the lemma"),
    Claim("H2", "solves_1_1(u)", (),
          "u is the localized velocity of Proposition 10.1, which solves "
          "the same equation",
          "hypothesis, supplied by Proposition 10.1 and (10.5)"),
    Claim("H3", "same_force(v,u)", (),
          "both carry the force of Lemma 10.3",
          "hypothesis of the lemma, supplied by Lemma 10.3"),
    Claim("H4", "zero_initial_difference(v,u)", (),
          "both start from zero initial velocity, so their difference does",
          "hypothesis of the lemma"),

    Claim("1", "difference_equation(v,u)", ("H1", "H2", "H3"),
          "the common force cancels and the difference solves (10.15)",
          "subtracting the two equations"),
    Claim("2", "flux_in_L1(v,u)", ("1",),
          "the momentum flux difference is integrable, uniformly in time",
          "w is square integrable and u is bounded"),

    Claim("3", "riesz_pressure_in_negative_sobolev(v,u)", ("2",),
          "the Riesz pressure lies in a negative Sobolev space",
          "the Fourier transform of an L^1 function is bounded"),
    Claim("4", "riesz_pressure_solves_poisson(v,u)", ("2",),
          "and it solves the Poisson equation for the flux",
          "the multiplier identity"),
    Claim("5", "true_pressure_solves_poisson(v,u)", ("1",),
          "so does the true pressure difference",
          "divergence of (10.15)"),
    Claim("6", "pressure_gradients_agree(v,u)", ("4", "5", "3"),
          "the two pressure gradients agree, with no growth condition on P",
          "their difference is harmonic and tempered, hence zero"),

    Claim("7", "sobolev_bound(v,u)", ("1",),
          "the sixth-power norm on the cutoff is controlled by the "
          "dissipation", "Sobolev and the product rule, (10.17)"),
    Claim("8", "riesz_part_bounded(v,u)", ("2",),
          "the Riesz part of the localized pressure is bounded",
          "L^p boundedness of Riesz transforms"),
    Claim("9", "commutator_part_bounded(v,u)", ("2",),
          "so is the commutator part, with a gain in R",
          "the kernel bound and Young's convolution inequality"),
    Claim("10", "pressure_flux_bounded(v,u)", ("6", "8", "9", "7"),
          "hence the pressure flux across the expanding sphere is bounded",
          "(10.19)"),

    Claim("11", "cutoff_energy_identity(v,u)", ("1", "7"),
          "pairing the difference equation with the cutoff gives the energy "
          "identity", "all integrations have compact support"),
    Claim("12", "transport_flux_bounded(v,u)", ("11",),
          "the transport flux is bounded",
          "the fields agree on the annulus, then Hoelder"),
    Claim("13", "differential_inequality(v,u)", ("11", "12", "10"),
          "every flux power is below two, so Young absorbs them",
          "leaving a Gronwall-ready inequality"),
    Claim("14", "energy_vanishes_as_R_grows(v,u)", ("13", "H4"),
          "Gronwall from zero initial energy gives a bound decaying in R",
          "the stretching coefficient is bounded on [0,T]"),
    Claim("15", "fields_agree(v,u)", ("14",),
          "so the difference vanishes and v = u on the interval",
          "letting the cutoff exhaust the space"),
]


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    print("=" * 78)
    print("Lemma 10.5, following the paper's own proof")
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

    print("\n--- what this settles ---")
    print(f"  derived steps: {len(audit.derived)}, "
          f"validated: {len(audit.derived) - len(audit.unvalidated)}")
    print(f"  leaves that are assertions about the construction: "
          f"{len(audit.unvalidated)}")
    print(f"  inputs: {len(audit.inputs)} -- the lemma's stated hypotheses")
    if not audit.unvalidated:
        print()
        print("  I called this one out of reach because it 'turns on a")
        print("  pressure flux controlled by Riesz transforms on the whole")
        print("  space, and nothing here models that'. That was a claim")
        print("  about the machine made without reading the proof. The")
        print("  Riesz transforms appear in a registered fact, the way")
        print("  Cauchy-Schwarz did in Lemma 10.4, and the argument around")
        print("  them is an implication like any other.")
    print()
    print("  Two steps carry real weight and are registered rather than")
    print("  derived: the commutator kernel bound, and the claim that every")
    print("  power of A_R is below two so Young absorbs. Both are on the")
    print("  assumption list under their own names, and a reader should")
    print("  check those two in the paper rather than take them from here.")


if __name__ == "__main__":
    main()
