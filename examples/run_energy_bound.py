"""
Experiment 1c: does one of the eleven decompose into simple prior rules?

The claim under test, and it is a good one. Every decomposition
terminates, and at each leaf there is either a rule the machine holds or
an assertion somebody made. My position has been that for the eleven
imported steps the leaves carrying the analytic content will be
assertions however finely you decompose. The counter-claim is that they
will be SIMPLE PRIOR FACTS, the sort of thing any textbook has, and that
registering those as rules turns the leaves into rule applications.

That is falsifiable, so this file tests it on the fairest case available.

Lemma 10.4 bounds the energy by the force. Of the eleven it has the most
elementary proof, and if the counter-claim holds anywhere it holds here.
The argument is the one everybody writes: pair the equation with the
velocity, watch the nonlinearity drop out because the field is
divergence-free, throw away the dissipation because it has a sign, apply
Cauchy-Schwarz, divide by the norm, integrate from rest.

Six facts are registered as prior rules. Every one is general, reusable
and citable -- none mentions the construction, the profiles, the pulses
or the correction cycle. Then the decomposition is submitted to
`check_proof` and the machine validates each link on its own or reports
that it cannot.

WHAT COUNTS AS WHICH. The lemma's own hypotheses are inputs, and that is
not a concession: a lemma is an implication and its antecedents are
supposed to be given. What would settle the question against me is the
leaves BEYOND those being rule applications rather than assertions about
the paper's objects.

Run: python examples/run_energy_bound.py
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
    """A standard fact, registered as a rule. General, never about this paper."""
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


#: Six standard facts. Read them and check that not one mentions the
#: construction: that is the whole point of calling them prior.
PRIOR = [
    prior("trilinear_antisymmetry", r"divfree\((\w+)\)",
          "trilinear_vanishes({0})",
          "for a divergence-free field the trilinear term <(u.grad)u, u> "
          "vanishes, by integration by parts"),
    prior("dissipation_has_a_sign", r"divfree\((\w+)\)",
          "dissipation_nonneg({0})",
          "<-laplacian u, u> = ||grad u||^2 >= 0"),
    prior("cauchy_schwarz", r"forced\((\w+),(\w+)\)",
          "pairing_bounded({0},{1})",
          "Cauchy-Schwarz: <f, u> <= ||f|| ||u||"),
    join("energy_pairing",
         ["forced(?u,?f)", "trilinear_vanishes(?u)"],
         "energy_identity(?u,?f)",
         "pairing the equation with the solution gives "
         "(1/2) d/dt ||u||^2 + ||grad u||^2 = <f, u>"),
    join("drop_the_dissipation",
         ["energy_identity(?u,?f)", "dissipation_nonneg(?u)"],
         "energy_inequality(?u,?f)",
         "a term with a sign may be discarded from the larger side of an "
         "inequality"),
    join("divide_and_integrate",
         ["energy_inequality(?u,?f)", "pairing_bounded(?u,?f)",
          "starts_from_rest(?u)"],
         "energy_bounded_by_force_integral(?u,?f)",
         "dividing by ||u|| gives d/dt ||u|| <= ||f||, and integrating from "
         "zero initial data gives ||u(t)|| <= integral of ||f||"),
]


#: Lemma 10.4, written out. The first three are the lemma's hypotheses.
STEPS = [
    Claim("H1", "forced(u,f)", (), "u solves the forced equation with force f",
          "hypothesis of the lemma"),
    Claim("H2", "divfree(u)", (), "u is divergence-free",
          "hypothesis of the lemma"),
    Claim("H3", "starts_from_rest(u)", (), "u vanishes at time zero",
          "hypothesis of the lemma"),

    Claim("1", "trilinear_vanishes(u)", ("H2",),
          "the nonlinearity contributes nothing to the energy",
          "integration by parts against a divergence-free field"),
    Claim("2", "dissipation_nonneg(u)", ("H2",),
          "the dissipation term has a sign",
          "it is a squared norm"),
    Claim("3", "pairing_bounded(u,f)", ("H1",),
          "the forcing pairing is bounded by the product of the norms",
          "Cauchy-Schwarz"),
    Claim("4", "energy_identity(u,f)", ("H1", "1"),
          "pairing the equation with u gives the energy identity",
          "the nonlinearity having dropped out"),
    Claim("5", "energy_inequality(u,f)", ("4", "2"),
          "discarding the dissipation gives an inequality",
          "the discarded term is non-negative"),
    Claim("6", "energy_bounded_by_force_integral(u,f)", ("5", "3", "H3"),
          "so the energy is bounded by the time integral of the force",
          "divide by the norm and integrate from rest"),
]


#: The three hypotheses, instantiated for the construction's own velocity.
#: Each is classified, and the classification is the thing to read: a
#: definition, a stipulation, and a one-line consequence of how the field
#: is built. None of them is the lemma wearing a disguise, and a reader
#: can check that by reading three sentences instead of trusting a count.
INSTANTIATION = [
    ("forced(U,F)",
     "the force F is DEFINED as the residual of the constructed velocity, "
     "so U solves the forced equation by construction rather than by any "
     "theorem",
     "definition"),
    ("divfree(U)",
     "U is assembled from curls of potentials, and the divergence of a "
     "curl is zero; the prior fact is standard and what is specific here "
     "is only that the construction builds it that way",
     "standard fact applied to a construction detail"),
    ("starts_from_rest(U)",
     "the statement being proved is about a solution starting from rest, "
     "so this is stipulated by the theorem rather than established",
     "stipulation of the theorem"),
]

INSTANTIATED = [
    Claim("C1", "forced(U,F)", (), "the constructed velocity and its force",
          INSTANTIATION[0][1]),
    Claim("C2", "divfree(U)", (), "the constructed velocity is solenoidal",
          INSTANTIATION[1][1]),
    Claim("C3", "starts_from_rest(U)", (), "it begins at rest",
          INSTANTIATION[2][1]),
    Claim("1", "trilinear_vanishes(U)", ("C2",),
          "the nonlinearity contributes nothing", "integration by parts"),
    Claim("2", "dissipation_nonneg(U)", ("C2",),
          "the dissipation has a sign", "it is a squared norm"),
    Claim("3", "pairing_bounded(U,F)", ("C1",),
          "the forcing pairing is bounded", "Cauchy-Schwarz"),
    Claim("4", "energy_identity(U,F)", ("C1", "1"),
          "the energy identity", "the nonlinearity dropped out"),
    Claim("5", "energy_inequality(U,F)", ("4", "2"),
          "discarding the dissipation", "it is non-negative"),
    Claim("6", "energy_bounded_by_force_integral(U,F)", ("5", "3", "C3"),
          "the energy is bounded by the force integral",
          "divide by the norm and integrate from rest"),
]


def report_instantiation(machine) -> None:
    """Apply the general lemma to the construction's own objects.

    The general lemma was proved and inert: an implication whose
    antecedents nothing here establishes. Supplying them is the step that
    makes it bite, and it is also where the hard content could be
    smuggled in, so every one is granted through `assume` with its
    standing declared and every proof through them is counted.
    """
    print("\n" + "=" * 78)
    print("Instantiating it: the same lemma, applied to the construction")
    print("=" * 78)
    print("  The general lemma is an implication and nothing here establishes")
    print("  its antecedents for the constructed field. That is the step the")
    print("  paper supplies and a reader supplies instantly. It is also")
    print("  exactly where the hard content could be smuggled in, so each is")
    print("  granted on the record with its standing declared:\n")
    for cell, because, standing in INSTANTIATION:
        machine.assume(cell, because, source="the paper", standing=standing)
        print(f"    {cell:22} [{standing}]")
        print(f"      {because}")

    audit = machine.check_proof(INSTANTIATED)
    print("\n--- the audit, instantiated ---")
    print(audit.report())

    print("\n--- what this settles ---")
    print(f"  derived steps validated: "
          f"{len(audit.derived) - len(audit.unvalidated)} of {len(audit.derived)}")
    print("  The energy bound now holds FOR THE CONSTRUCTION'S FIELD, modulo")
    print("  a list a reader can check in a minute: six textbook facts and")
    print("  three instantiations that are a definition, a stipulation, and")
    print("  a curl being divergence-free.")
    print()
    print("  One thing this does not do, and it is the last honest gap here.")
    print("  Identifying the cell it reached with the imported step's cell")
    print("  is a judgement that two sentences say the same thing, in two")
    print("  different languages. No machinery here makes that judgement,")
    print("  and asserting it myself would be the unaudited claim this")
    print("  package exists to catch.")


def main() -> None:
    machine = RenMachine(device="cpu")
    for rule in PRIOR:
        machine.library.add(rule)

    print("=" * 78)
    print("Lemma 10.4, decomposed to its leaves")
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

    hypotheses = [v for v in audit.inputs]
    print("\n--- the count that settles it ---")
    print(f"  derived steps:                       {len(audit.derived)}")
    print(f"  validated by a rule the machine holds: "
          f"{len(audit.derived) - len(audit.unvalidated)}")
    print(f"  leaves that are assertions about this construction: "
          f"{len([v for v in audit.unvalidated])}")
    print(f"  leaves that are the lemma's own hypotheses: {len(hypotheses)}")
    print(f"  prior facts the whole thing rests on: {len(audit.assumptions)}")

    print("\n--- reading it honestly ---")
    if not audit.unvalidated:
        print("  The counter-claim holds here, and I said it was falsifiable,")
        print("  so: every derived step went through a registered prior fact,")
        print("  and not one leaf is an assertion about the paper's objects.")
        print("  The only inputs are the lemma's own hypotheses, which is what")
        print("  a lemma is supposed to have. Lemma 10.4 is an untrusted")
        print("  import in the main run and is derived here from six textbook")
        print("  facts, which is a real change in its standing.")
    else:
        print("  Some leaf is still an assertion about the construction.")
    print()
    print("  What it does NOT show. Those six facts are registered, not")
    print("  proved here, so the status is 'complete modulo six classical")
    print("  facts'. That is what mathematics does with a standard library")
    print("  and it is a legitimate place to stand, provided the list is")
    print("  printed, which it is.")
    print()
    print("  And it is the easiest of the eleven by a distance. Its proof is")
    print("  one integration by parts and a Cauchy-Schwarz. The three")
    print("  existentials assert an object into being, and no chain of")
    print("  textbook facts produces one; the remaining bounds are about")
    print("  objects the construction builds, so their leaves reach for")
    print("  properties of those objects rather than general facts. The")
    print("  honest generalisation from this experiment is that the eleven")
    print("  are not alike, and I have been treating them as though they")
    print("  were.")


    report_instantiation(machine)


if __name__ == "__main__":
    main()
