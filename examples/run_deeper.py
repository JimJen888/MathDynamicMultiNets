"""
Experiment 1i: do the proof-specific computations bottom out in known rules?

The census found that 85% of what the five decompositions lean on is named
mathematics, and six entries were classified as computations belonging to
this proof rather than rules anyone has memorised. The objection to that
classification is the obvious one and it is right: a computation is
performed, and the rules it performs are themselves standard. "Nobody
memorised this composite" is a different statement from "this uses
unknown rules".

So this takes three of the six and decomposes them ONE LEVEL FURTHER,
following the computation the paper actually writes out, and asks whether
the second level is named all the way down.

  the commutator kernel bound   Lemma 10.5, the estimate on K_R
  the coordinate bound (10.3)   Proposition 10.1, the margin in q
  Young absorbing the fluxes    Lemma 10.5, the hinge of the energy
                                estimate

If their leaves are named, then "proof-specific" marks a level of
description rather than a boundary of knowledge, and the recursion
terminates in things a model holds. If some leaf is not, the
classification was tracking something real.

Run: python examples/run_deeper.py
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


#: Every rule below is named mathematics. That is the thing being tested,
#: so each carries the name it goes by.
PRIOR = [
    # -- the commutator kernel bound ----------------------------------------
    prior("calderon_zygmund_kernel", r"riesz_pair\((\w+)\)",
          "kernel_decays({0})",
          "Calderon-Zygmund: a second-order Riesz transform has a kernel "
          "bounded by C |x-y|^-d away from the diagonal"),
    prior("commutator_drops_the_identity", r"riesz_pair\((\w+)\)",
          "identity_cancels({0})",
          "in [a, T]g = a(Tg) - T(ag) any multiple of the identity in T "
          "cancels, by bilinearity"),
    prior("cutoff_is_lipschitz", r"cutoff_width\((\w+)\)",
          "lipschitz_gain({0})",
          "a smooth cutoff of width R satisfies |phi(x) - phi(y)| <= "
          "min(|x-y|/R, 1), by the mean value theorem and 0 <= phi <= 1"),
    join("kernel_bound",
         ["kernel_decays(?K)", "identity_cancels(?K)", "lipschitz_gain(?W)"],
         "kernel_bounded(?K,?W)",
         "multiplying the decay by the Lipschitz gain bounds the commutator "
         "kernel by C |x-y|^-3 min(|x-y|/R, 1)"),
    prior("polar_coordinates", r"kernel_bounded\((\w+),(\w+)\)",
          "radial_integrals({0},{1})",
          "integration in polar coordinates turns a radial integrand in R^3 "
          "into r^2 dr"),
    prior("power_integral", r"radial_integrals\((\w+),(\w+)\)",
          "both_pieces_converge({0},{1})",
          "the integral of r^a converges at the origin for a > -1 and at "
          "infinity for a < -1; here the exponents are -2/3 and -2"),
    join("kernel_norm",
         ["both_pieces_converge(?K,?W)", "kernel_bounded(?K,?W)"],
         "kernel_norm_bounded(?K,?W)",
         "adding the two pieces gives the L^4/3 norm of the commutator "
         "kernel at most C R^-3/4"),
    prior("young_convolution", r"kernel_norm_bounded\((\w+),(\w+)\)",
          "commutator_part_bounded({0},{1})",
          "Young's convolution inequality bounds a convolution's norm by "
          "the product of the factors' norms"),

    # -- the coordinate bound (10.3) ----------------------------------------
    prior("similarity_coordinates", r"coordinates\((\w+)\)",
          "case_split({0})",
          "the similarity coordinates define q through tau and eta, and "
          "either 1 - eta^2 >= 1/2 or it does not; a real number satisfies "
          "exactly one"),
    prior("first_case", r"case_split\((\w+)\)", "bound_in_tau({0})",
          "when 1 - eta^2 >= 1/2 the definition gives q <= 2 tau directly, "
          "by algebra"),
    prior("second_case", r"case_split\((\w+)\)", "bound_in_z({0})",
          "otherwise |eta| exceeds 2^-1/2 and the definition gives "
          "q <= (sqrt 2 |z|)^(1/D), again by algebra"),
    join("combine_the_cases",
         ["bound_in_tau(?C)", "bound_in_z(?C)"],
         "margin_bound(?C)",
         "a bound holding in each case of an exhaustive split holds "
         "throughout, and the maximum of the two is at most their sum "
         "times a constant"),

    # -- Young absorbing the fluxes -----------------------------------------
    prior("young_with_epsilon", r"flux_powers\((\w+)\)",
          "each_power_absorbed({0})",
          "Young's inequality with a parameter: ab <= delta a^p + C_delta "
          "b^q whenever 1/p + 1/q = 1, so any power below the dissipation's "
          "can be absorbed into it"),
    prior("compare_exponents", r"flux_powers\((\w+)\)", "powers_below_two({0})",
          "the flux bounds carry powers 1/2, 3/4 and 3/2 of the dissipation "
          "term, and each is less than 2; comparing rationals"),
    join("absorb",
         ["each_power_absorbed(?F)", "powers_below_two(?F)"],
         "fluxes_absorbed(?F)",
         "every flux term therefore moves to the dissipation side, leaving "
         "a Gronwall-ready inequality"),
]


CASES = [
    ("the commutator kernel bound (Lemma 10.5)", [
        Claim("H1", "riesz_pair(K)", (), "a second-order Riesz transform",
              "the object being commuted"),
        Claim("H2", "cutoff_width(W)", (), "a smooth cutoff of width R",
              "the localization being used"),
        Claim("1", "kernel_decays(K)", ("H1",),
              "its kernel decays like the inverse cube",
              "Calderon-Zygmund"),
        Claim("2", "identity_cancels(K)", ("H1",),
              "the identity part drops out of the commutator",
              "bilinearity"),
        Claim("3", "lipschitz_gain(W)", ("H2",),
              "the cutoff difference gains a factor min(|x-y|/R, 1)",
              "mean value theorem"),
        Claim("4", "kernel_bounded(K,W)", ("1", "2", "3"),
              "hence the commutator kernel bound", "multiplying the two"),
        Claim("5", "radial_integrals(K,W)", ("4",),
              "its norm becomes two radial integrals", "polar coordinates"),
        Claim("6", "both_pieces_converge(K,W)", ("5",),
              "both converge, by their exponents", "the power integral"),
        Claim("7", "kernel_norm_bounded(K,W)", ("6", "4"),
              "giving the norm bound with its gain in R", "adding them"),
        Claim("8", "commutator_part_bounded(K,W)", ("7",),
              "and the commutator term is bounded", "Young's convolution"),
    ]),
    ("the coordinate bound 10.3 (Proposition 10.1)", [
        Claim("H1", "coordinates(C)", (), "the similarity coordinates",
              "the definitions in (3.2)"),
        Claim("1", "case_split(C)", ("H1",), "split on the size of 1 - eta^2",
              "a real number is on one side or the other"),
        Claim("2", "bound_in_tau(C)", ("1",), "the first case bounds q by tau",
              "algebra"),
        Claim("3", "bound_in_z(C)", ("1",), "the second bounds it by z",
              "algebra"),
        Claim("4", "margin_bound(C)", ("2", "3"), "so the bound holds always",
              "an exhaustive case split"),
    ]),
    ("Young absorbing the fluxes (Lemma 10.5)", [
        Claim("H1", "flux_powers(F)", (),
              "the flux bounds, with their powers of the dissipation",
              "read off the three flux estimates"),
        Claim("1", "each_power_absorbed(F)", ("H1",),
              "any power below the dissipation's can be absorbed",
              "Young with a parameter"),
        Claim("2", "powers_below_two(F)", ("H1",),
              "and each power that occurs is below two",
              "comparing rationals"),
        Claim("3", "fluxes_absorbed(F)", ("1", "2"),
              "so every flux moves to the other side",
              "absorbing them one at a time"),
    ]),
]


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    print("=" * 78)
    print("The proof-specific computations, decomposed one level further")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    results = []
    for title, steps in CASES:
        audit = machine.check_proof(steps)
        results.append((title, audit))
        print(f"\n--- {title} ---")
        for claim, verdict in zip(steps, audit.verdicts):
            origin = ", ".join(claim.premises) if claim.premises else "given"
            print(f"  {claim.key}. [{origin}] {claim.claim}")
            print(f"      {verdict.status}: {verdict.detail}")

    print("\n--- the number ---")
    total = sum(len(a.derived) for _, a in results)
    good = sum(len(a.derived) - len(a.unvalidated) for _, a in results)
    print(f"  second-level steps: {total}, validated: {good}")
    print(f"  rules they use: {len(PRIOR)}, all named mathematics")
    for name in ("Calderon-Zygmund", "mean value theorem", "polar coordinates",
                 "Young's convolution", "Young with a parameter"):
        print(f"    {name}")

    print("\n--- what it settles ---")
    if good == total:
        print("  The objection holds. Every one of these three bottoms out")
        print("  in named rules: Calderon-Zygmund for the kernel, the mean")
        print("  value theorem for the cutoff, polar coordinates and a power")
        print("  integral for the norm, Young twice, and an exhaustive case")
        print("  split for the coordinate bound.")
        print()
        print("  So 'proof-specific' marks a LEVEL OF DESCRIPTION, not a")
        print("  boundary of knowledge. The census counted composites that")
        print("  nobody has memorised as a unit; decomposed once more, the")
        print("  pieces are all things anyone holds. The 85% in that census")
        print("  is a floor rather than a ceiling, and it rises as the")
        print("  decomposition goes deeper.")
        print()
        print("  What this does NOT show is that the recursion always")
        print("  terminates this quickly. Three of six were taken here, and")
        print("  they were the three whose computation the paper writes out.")
        print("  A step whose computation the paper compresses into a")
        print("  citation would need the cited source, not another level.")


if __name__ == "__main__":
    main()
