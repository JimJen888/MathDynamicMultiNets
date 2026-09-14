"""
Experiment 1g: Proposition 10.1 and Theorem 1.1, following the paper.

Two more of the eleven, both taken from the paper's own proofs rather than
reconstructed. They are the two ends of Section 10: the localization that
turns the local fields into compactly supported ones, and the theorem that
collects everything.

Proposition 10.1 is almost entirely structural. Choose cutoffs with a
margin from the outer boundary, apply them to the potential rather than to
the field, and the divergence-free property survives because the
divergence of a curl is zero. The one quantitative ingredient is the
coordinate bound (10.3), which is what lets a fixed spatial cutoff sit
inside the local domain.

Theorem 1.1 is a conjunction, and it is the one step whose shape I had
right: it collects Lemma 10.3, Lemma 10.4, Lemma 10.5 and the growth
asymptotic, and argues by contradiction. What I had wrong was calling it
out of reach for that reason. A conjunction is a step like any other once
the machine has `JoinRule`, and the parts it collects are the previous
links.

WHERE TO BE SUSPICIOUS. Two registered facts do real work. The coordinate
bound (10.3) is a computation from the similarity coordinates, carried
here as a named assumption. And the growth asymptotic (10.21) comes from
Theorem 3.1(ii), which is a genuine import from outside Section 10, listed
as such rather than folded into the count.

Run: python examples/run_theorem.py
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
    # -- Proposition 10.1, the localization ---------------------------------
    prior("coordinate_bound", r"local_fields\((\w+)\)", "margin_exists({0})",
          "the similarity coordinates give q <= C0 (tau + |z|^(1/D)), so "
          "thresholds in tau and z can be chosen keeping the cutoff support "
          "a fixed distance inside the outer boundary (10.3)"),
    prior("cutoffs_exist", r"margin_exists\((\w+)\)", "cutoff_chosen({0})",
          "a smooth axisymmetric spatial cutoff and a smooth temporal cutoff "
          "exist with the required supports; their product is smooth"),
    join("cut_the_potential",
         ["local_fields(?L)", "cutoff_chosen(?L)"],
         "cut_fields(?L)",
         "apply the cutoff to the vector potential and the azimuthal field "
         "rather than to the velocity: u = curl(cA) + cB e_theta, p = c p_loc "
         "(10.4)"),
    prior("curl_is_solenoidal", r"cut_fields\((\w+)\)", "cut_divfree({0})",
          "the divergence of a curl vanishes, and an azimuthal field with no "
          "angular dependence contributes nothing, so the cut field is still "
          "divergence-free"),
    prior("axis_regularity", r"cut_fields\((\w+)\)", "cut_smooth({0})",
          "the Cartesian representatives of the base potentials are smooth "
          "at the axis, and the cutoffs give a smooth zero extension "
          "elsewhere"),
    prior("support_is_compact", r"cut_fields\((\w+)\)", "cut_compact({0})",
          "the support is contained in the spatial cutoff's support, and the "
          "temporal cutoff makes the fields vanish for small times"),
    join("localized_fields_exist",
         ["cut_divfree(?L)", "cut_smooth(?L)", "cut_compact(?L)"],
         "localized_fields(?L)",
         "collecting: smooth, divergence-free, compactly supported, and "
         "agreeing with the local fields near the origin at late times"),

    # -- Theorem 1.1, the conjunction ---------------------------------------
    prior("solves_on_the_interval", r"localized_fields\((\w+)\)",
          "exact_on_interval({0})",
          "the force is DEFINED by (10.5) as the residual, so the field "
          "solves the equation exactly on [0,1)"),
    prior("regularity_class", r"localized_fields\((\w+)\)",
          "in_H3_on_compacts({0})",
          "smoothness and fixed compact support give membership in "
          "C([0,T]; H^3) for every T < 1"),
    join("velocity_diverges",
         ["exact_on_interval(?L)", "inner_growth_asymptotic(?L)"],
         "unbounded_along_a_path(?L)",
         "along the path (10.20) the cutoffs equal one and the inner "
         "asymptotic (3.6) gives u_theta = tau^-A (e0 + O(tau^2h)), which "
         "diverges as the points approach the origin (10.21)"),
    join("no_classical_continuation",
         ["unbounded_along_a_path(?L)", "in_H3_on_compacts(?L)"],
         "lifespan_is_the_interval(?L)",
         "the embedding H^3 into L^infinity excludes a classical "
         "continuation through time one, so the maximal classical existence "
         "interval is [0,1)"),
    join("contradiction_with_a_competitor",
         ["lifespan_is_the_interval(?L)", "competitor_equals_it(?L)",
          "bounded_energy(?L)", "smooth_compact_force(?L)"],
         "theorem_1_1_forced_blowup(?L)",
         "a global smooth solution of bounded energy would equal the "
         "constructed field throughout [0,1) by the comparison lemma, and "
         "the divergence above contradicts its boundedness on a compact "
         "neighbourhood; the force is nonzero since otherwise the energy "
         "identity would force the zero solution"),
]


STEPS_10_1 = [
    Claim("H1", "local_fields(L)", (),
          "the local fields of Proposition 9.9 on the local domain",
          "the previous link in the chain"),

    Claim("1", "margin_exists(L)", ("H1",),
          "thresholds exist keeping a fixed cutoff inside the domain",
          "the coordinate bound (10.3)"),
    Claim("2", "cutoff_chosen(L)", ("1",),
          "so smooth spatial and temporal cutoffs may be chosen",
          "standard existence of bump functions"),
    Claim("3", "cut_fields(L)", ("H1", "2"),
          "cut the potential and the azimuthal field, not the velocity",
          "(10.4)"),
    Claim("4", "cut_divfree(L)", ("3",),
          "the cut field is still divergence-free",
          "the divergence of a curl vanishes"),
    Claim("5", "cut_smooth(L)", ("3",), "and smooth, including at the axis",
          "Cartesian representatives plus zero extension"),
    Claim("6", "cut_compact(L)", ("3",),
          "and compactly supported, vanishing for small times",
          "the two cutoffs"),
    Claim("7", "localized_fields(L)", ("4", "5", "6"),
          "which is Proposition 10.1", "collecting the three"),
]


STEPS_1_1 = [
    Claim("H1", "localized_fields(L)", (),
          "the localized fields of Proposition 10.1", "previous link"),
    Claim("H2", "inner_growth_asymptotic(L)", (),
          "the inner swirl asymptotic (3.6) of Theorem 3.1(ii)",
          "IMPORT from outside Section 10: Theorem 3.1"),
    Claim("H3", "competitor_equals_it(L)", (),
          "a global smooth solution of bounded energy equals the "
          "constructed field on every shorter interval",
          "previous link: Lemma 10.5"),
    Claim("H4", "bounded_energy(L)", (),
          "the constructed field has bounded kinetic energy",
          "previous link: Lemma 10.4"),
    Claim("H5", "smooth_compact_force(L)", (),
          "the force is smooth with compact support",
          "previous link: Lemma 10.3"),

    Claim("1", "exact_on_interval(L)", ("H1",),
          "the field solves the equation exactly on the interval",
          "the force is defined as the residual, (10.5)"),
    Claim("2", "in_H3_on_compacts(L)", ("H1",),
          "and lies in the classical regularity class",
          "smoothness and compact support"),
    Claim("3", "unbounded_along_a_path(L)", ("1", "H2"),
          "the swirl diverges along points approaching the origin",
          "(10.21)"),
    Claim("4", "lifespan_is_the_interval(L)", ("3", "2"),
          "so no classical continuation through time one exists",
          "the Sobolev embedding"),
    Claim("5", "theorem_1_1_forced_blowup(L)", ("4", "H3", "H4", "H5"),
          "and a global smooth solution of bounded energy is impossible",
          "the comparison lemma plus the divergence"),
]


def show(machine, title: str, steps) -> None:
    audit = machine.check_proof(steps)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for claim, verdict in zip(steps, audit.verdicts):
        origin = ", ".join(claim.premises) if claim.premises else "given"
        print(f"  {claim.key}. [{origin}] {claim.claim}")
        print(f"      {verdict.status}: {verdict.detail}")
    print()
    print(audit.report())
    return audit


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])
    a = show(machine, "Proposition 10.1: localization", STEPS_10_1)
    b = show(machine, "Theorem 1.1: the conjunction", STEPS_1_1)

    print("\n" + "=" * 78)
    print("--- what these two settle ---")
    for name, audit in (("Proposition 10.1", a), ("Theorem 1.1", b)):
        print(f"  {name:18} {len(audit.derived) - len(audit.unvalidated)}"
              f"/{len(audit.derived)} steps, {len(audit.inputs)} inputs, "
              f"{len(audit.assumptions)} facts")
    print()
    print("  Theorem 1.1 is the one whose SHAPE I had right and whose")
    print("  reachability I had wrong. I called it out of reach because it")
    print("  is the conjunction of everything, uniformly. A conjunction is")
    print("  a step like any other once the machine can express one, and")
    print("  the things it collects are the previous links.")
    print()
    print("  One genuine import appears and is listed rather than folded in:")
    print("  the inner growth asymptotic comes from Theorem 3.1, which is")
    print("  outside Section 10 and is not one of the eleven. Replacing a")
    print("  step by its proof keeps pulling in hypotheses from elsewhere,")
    print("  and the honest report says which.")


if __name__ == "__main__":
    main()
