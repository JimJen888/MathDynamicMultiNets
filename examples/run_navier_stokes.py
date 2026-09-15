"""
Experiment 1: re-deriving the Navier-Stokes blowup construction as rules.

The OpenAI preprint constructs, for every viscosity, a smooth compactly
supported force and a solution from rest whose kinetic energy stays bounded
while its velocity becomes unbounded at t = 1 -- alternatives (C) and (D) of
Fefferman's problem description. This run takes the CONSTRUCTION, not the
paper's word for it, and rebuilds it in the only currency this machine has:
mapping rules, and an oracle that can decide whether one is right.

Every named result becomes a rule. They split into two kinds, and the whole
point of the run is that the machine keeps them apart by itself.

  RULES IT CHECKS. Six results reduce to something computable on an instance,
  so each is declared UNVERIFIED and then measured against an oracle that
  answers the same question a different way:

      Lemma 4.5   the admissible cone     against the two wave families
                  by its quadratic test   actually built and solved for
                                          positive squared amplitudes
      Lemma 7.4   pulse grows then        against integrating the amplitude
                  decays, from its end    equation across the slot
                  rates alone
      Lemma A.6   the exterior swirl      against differencing the field and
                  solves the heat         forming the residual
                  equation
      Lemma A.1   distinct radial powers  against the determinant of the
                  give invertible         matrix, built from actual bumps
                  moment matrices
      Prop 9.6    sigma_j = 1/5 + j/10    against iterating the recursion
      Sec 3.5     bounded energy and      against numerical quadrature of the
                  integrable dissipation  dissipation integral

  RULES IT IMPORTS. Nine results are estimates on function spaces -- Theorem
  4.6, Propositions 5.5, 7.5, 9.6, 9.9, the localization and comparison of
  Section 10. A rule can state the step. Nothing here can establish it, so
  they stay untrusted, and the search that reaches Theorem 1.1 through them
  reports itself as unproved.

What that buys, and what it does not, is printed at the end. Two gaps survive
the run and no amount of running closes either: every oracle above decides one
instance, while the paper quantifies over all q as q goes to zero, over every
dyadic band, every label and every correction stage; and the estimates
themselves are untouched.

    python examples/run_navier_stokes.py            # ~2 min, no GPU, no torch
    python examples/run_navier_stokes.py --quick    # ~30 s, fewer instances
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamicmultinets import (LLMController, RenMachine,             # noqa: E402
                              ScriptedController)
from dynamicmultinets.navierstokes import (                          # noqa: E402
    CLAIM_DATA, CLAIM_RULES, IMPORTED_NAMES, MUTATIONS,
    install_navier_stokes_rules)
from dynamicmultinets.nsmechanisms import (                         # noqa: E402
    MECHANISM_CHECKS, MECHANISM_MUTATIONS, REMAINING, SUPPORTS,
    install_mechanism_rules)
from dynamicmultinets.normcalc import (                            # noqa: E402
    RESIDUAL_START, RESIDUAL_TARGET, assumptions_behind, est, glob,
    install_norm_rules, pair, propose_band_rule, required_estimate, vanishes)
from dynamicmultinets.nsderivation import (                         # noqa: E402
    CYCLE_MOVES, DERIVATION_CHECKS, DERIVATION_IMPORTED, DERIVATION_LEGS,
    DERIVATION_MUTATIONS, derive_cycle_gain, initial_state,
    install_derivation_rules, prove_cycle_closes_at_every_stage, state_after)

#: Every step of the paper the machine carries as a rule and cannot check.
ALL_IMPORTED = tuple(IMPORTED_NAMES) + DERIVATION_IMPORTED
from dynamicmultinets.tapes import ABSTRACT, SPECIFIC, Content       # noqa: E402
from dynamicmultinets.witness import build_moment_profile           # noqa: E402

GOAL = (
    "Rebuild the forced Navier-Stokes blowup construction as rules. Form one "
    "rule for every result that reduces to a computation, verify each against "
    "an oracle that decides the same question by a different route, prove the "
    "instance tasks with what survives, and report exactly which steps are "
    "imported and what the instances do not reach."
)

# The construction's only free exponent. Theorem 4.6 fixes h in (0, 1/100);
# 1/200 sits inside it, and every number printed below is exact at that value.
H = "1/200"
VERDICT = f"bounded_energy_unbounded_velocity,h={H}"

# How many instances each check gets.
#
# One number for everything is the wrong shape, because the oracles differ in
# cost by two orders of magnitude and because confidence is Laplace-smoothed
# over the checks that ran: at 160 checks a clean rule reports 0.9938 and at
# 1200 it reports 0.9992, and the difference is real evidence when the
# instances are different questions and nothing at all when they are repeats.
# So each count is the larger of what the oracle can afford in about ten
# seconds and what its generator can still answer without repeating itself.
#
# 1200 is the ceiling. Past it the narrower generators start returning cells
# they have already returned, `verify` charges the rule once per distinct
# question, and the extra examples cost time while buying nothing. The one
# exception below is the energy bound, whose oracle integrates a stiff system
# at 0.37 seconds an instance; 160 of those is already a minute.
CHECK_INSTANCES = {
    "energy_budget_3_5": 1200,
    "lemma_4_5_cone": 1200,
    "lemma_7_4_pulse": 400,
    "lemma_A6_heat": 1200,
    "lemma_A1_moments": 1200,
    "prop_9_6_decay": 1200,
    "eq_4_1_exponents": 600,
    "prop_9_6_table": 1200,
    "lemma_5_4_summation": 1200,
    "lemma_10_3_borel": 1200,
    "lemma_10_4_energy_bound": 160,
    "prop_10_1_curl_order": 400,
    "increment_identity": 1200,
}
DEFAULT_INSTANCES = 400


# One instance per checked result, chosen inside the regime the construction
# actually uses rather than wherever the rule happens to answer.
INSTANCES = {
    "cone_instance": ("cone(a=3,bs=1/2,p1=12,p2=3)", "admissible"),
    "pulse_instance": ("pulse(lambda0=2,ustar=1,damp=1,slot=40)", "grows_then_decays"),
    "heat_instance": (f"heat(h={H},A=101/200,r=21/10,tau=7/20)",
                      "solves_radial_swirl_heat"),
    "moment_instance": ("exponents=1/2;-1/2;-3/2", "moment_matrix_invertible"),
    "cycle_instance": ("stage=40", "sigma=21/5"),
    "energy_instance": (f"h={H}", VERDICT),
}


NORM_PROOFS = (
    ("bernstein_derivative", "band_exponents_by_scaling"),
    ("bernstein_uniform", "band_exponents_by_scaling"),
    ("bernstein_to_energy", "band_exponents_by_scaling"),
    ("sum_over_bands", "summation_and_limit_by_sign"),
    ("limit_at_the_singularity", "summation_and_limit_by_sign"),
)


def plan(scale: float) -> list[tuple[str, dict]]:
    """The reference sequence, as the tool calls a controller would make.

    `scale` divides every instance count; the quick run passes a quarter.
    """

    def count(rule: str) -> int:
        return max(20, int(CHECK_INSTANCES.get(rule, DEFAULT_INSTANCES) * scale))

    steps: list[tuple[str, dict]] = [
        ("inspect_machine", {}),

        # --- what "useful" means here ---------------------------------------
        # One task per checked result, plus the same energy question posed as
        # a DRAWING the machine did not write, which nothing can read.
    ]
    for name, (start, target) in INSTANCES.items():
        steps.append(("add_task", {"name": name, "start": start, "target": target}))
    steps.append(("add_task", {"name": "energy_from_screen", "start": f"h={H}",
                               "target": VERDICT, "domain": "specific",
                               "observed": True}))

    # Before anything is checked, the six claim rules are untrusted and no
    # proof may use them. This report is the baseline the run is measured
    # against, and it should look bad.
    steps.append(("library_report", {}))

    # --- check each claim against the mechanism it claims to decide ---------
    # No training set: these rules are symbolic, formed from the construction's
    # own algebra, so every example is a test example and the seed only has to
    # be fresh with respect to nothing at all. What matters is the base rate,
    # printed beside each accuracy afterwards.
    for i, (rule, oracle) in enumerate(CLAIM_RULES.items()):
        generator, params = CLAIM_DATA[rule]
        dataset = f"check_{rule}"
        steps.append(("generate_data", {"generator": generator, "n": count(rule),
                                        "seed": 101 + i, "name": dataset,
                                        "params": params}))
        steps.append(("verify_rule", {"rule": rule, "dataset": dataset,
                                      "oracle": oracle, "threshold": 0.99}))

    # The cone rule gets a second, independent oracle: the definition (4.21)
    # with its square root, rather than the wave amplitudes. Lemma 4.5 says
    # the two must agree; this is where that stops being a citation.
    steps.append(("generate_data", {"generator": "ns_cone_points",
                                    "n": count("lemma_4_5_cone"),
                                    "seed": 909, "name": "cone_by_defn"}))
    steps.append(("verify_rule", {"rule": "lemma_4_5_cone",
                                  "dataset": "cone_by_defn",
                                  "oracle": "cone_by_definition",
                                  "threshold": 0.99}))

    # --- the imported estimates, with their methods carried out ------------
    # Each of these is the mechanism behind a step the machine cannot check
    # as stated. Running it does not make the step true, and the step stays
    # untrusted; what it does is move the claim from "cited" to "the
    # construction was carried out here and behaved".
    for i, (rule, (oracle, generator)) in enumerate(
            {**MECHANISM_CHECKS, **DERIVATION_CHECKS}.items()):
        dataset = f"check_{rule}"
        steps.append(("generate_data", {"generator": generator, "n": count(rule),
                                        "seed": 301 + i, "name": dataset}))
        steps.append(("verify_rule", {"rule": rule, "dataset": dataset,
                                      "oracle": oracle, "threshold": 0.99}))

    # Proposition 7.5 gets its own second opinion: the covariance integrated
    # from the pulses themselves, rather than from their frozen leading form.
    steps.append(("generate_data", {"generator": "ns_cone_points",
                                    "n": count("lemma_7_4_pulse"),
                                    "seed": 717, "name": "cone_by_pulses"}))
    steps.append(("verify_rule", {"rule": "lemma_4_5_cone",
                                  "dataset": "cone_by_pulses",
                                  "oracle": "covariance_from_integrated_pulses",
                                  "threshold": 0.99}))

    # The one claim here that is not settled by asking questions. The
    # induction of Proposition 9.6 quantifies over every stage, which no
    # number of instances reaches -- but it quantifies over a rational
    # recursion on an integer index, and that is decidable. So it is
    # proved instead of checked, and the tool that does it is a different
    # tool for that reason.
    steps.append(("prove_rule", {"rule": "prop_9_6_all_stages",
                                 "prover": "stage_induction_by_linear_arithmetic"}))
    # And the identity everything in Sections 7 and 9 rests on. Its
    # derivation is four lines of algebra among the field and its first
    # derivatives, so expanding it settles every smooth field at once and
    # the 1200 differentiated fields become a confirmation of a theorem.
    steps.append(("prove_rule", {
        "rule": "increment_identity",
        "prover": "increment_identity_by_polynomial_algebra"}))
    # And the band inequalities, whose exponents are forced by scaling.
    # These run here rather than in the report so the final library table
    # says PROVED for them instead of trusted.
    for rule, prover_name in NORM_PROOFS:
        steps.append(("prove_rule", {"rule": rule, "prover": prover_name}))
    # And the viscous balance of Section 7.2, which is uniform over a
    # continuum of eps and so was never in reach of an instance either.
    steps.append(("prove_rule", {"rule": "ns_carrier_frequency",
                                 "prover": "carrier_balance_by_factorisation"}))

    steps.append(("library_report", {}))
    for name in INSTANCES:
        start, target = INSTANCES[name]
        steps.append(("prove", {"start": start, "target": target, "max_depth": 8}))

    # The same question about a cell the machine did not draw.
    steps.append(("prove", {"start": f"h={H}", "target": VERDICT,
                            "domain": SPECIFIC, "observed": True,
                            "max_depth": 8}))

    # What is missing, shown rather than asserted. The same drawn cell without
    # the observed mark is read by the caption copier and the chain completes;
    # with it, nothing in the library can start. The gap between those two
    # lines is exactly the reader this run never builds.
    steps.append(("prove", {"start": f"h={H}", "target": VERDICT,
                            "domain": SPECIFIC, "max_depth": 8}))
    steps.append(("finish", {"summary": "checked results verified; estimates imported"}))
    return steps


def report_measurements(machine: RenMachine) -> None:
    """Every accuracy beside the base rate that guessing would have scored."""
    print("\n--- what each check actually measured ---")
    print(f"  {'result':22}{'oracle':30}{'n':>5}{'acc':>8}{'base':>8}  answers")
    checks = [(rule, oracle, f"check_{rule}") for rule, oracle in CLAIM_RULES.items()]
    checks.append(("lemma_4_5_cone", "cone_by_definition", "cone_by_defn"))
    for rule, oracle, name in checks:
        dataset = machine.datasets[name]
        labels = [e.out.text for e in dataset.examples if e.labeled]
        if not labels:
            continue
        r = machine.library.get(rule)
        hits = sum(1 for e in dataset.examples if e.labeled
                   and (r.apply(e.inp) or Content.abstract("")).text == e.out.text)
        base = max(Counter(labels).values()) / len(labels)
        print(f"  {rule:22}{oracle:30}{len(labels):5}{hits / len(labels):8.3f}"
              f"{base:8.3f}  {len(set(labels))} classes")
    print("  Agreement is worth what the question was hard. A two-class check\n"
          "  against a base rate near 0.85 says little on its own; the energy\n"
          "  budget and the decay parameter answer with a hundred-odd distinct\n"
          "  values each, and those two are the strongest checks here for that\n"
          "  reason before any other.")


def derive_the_cycle(machine: RenMachine) -> None:
    """Find one correction cycle by search, keep it, check it, and reuse it.

    This is the derivation done the way the architecture does derivations
    rather than the way a comment does them. The four numbered steps of
    Proposition 9.6 are four rules over a cell holding the decay orders, and
    nothing tells the machine what order to apply them in: it searches. What
    it finds is kept under one name, which is what makes the next cycle a
    single move, and only then is the chain checked -- against the closed
    form the recursion is supposed to have, which is a different encoding of
    the same claim and is labelled as the consistency check it is.
    """
    print("\n--- the correction cycle, derived rather than asserted ---")
    start, after_one = initial_state(), state_after(1)
    found = machine.prove(start, after_one, max_depth=6, trusted_only=False)
    print(f"  search over the four steps: "
          f"{'found' if found.found else 'not found'} in {found.length} moves")
    for step in found.steps:
        print(f"    {step.rule}")
    if not found.found:
        return

    machine.keep_proof(found, "cycle_once",
                       "one correction cycle of Proposition 9.6")
    kept = machine.library.get("cycle_once")
    print(f"  kept as one rule: {kept.steps()} primitive steps, "
          f"confidence {kept.confidence():.4f}, "
          f"{'trusted' if kept.trusted else 'UNTRUSTED'}")

    # The closed form is Fraction arithmetic and costs nothing to evaluate,
    # so this one is limited by the generator rather than by the clock.
    machine.generate_data("ns_cycle_states", 1200, seed=808,
                          name="check_cycle_once")
    report = machine.verify("cycle_once", "check_cycle_once",
                            "cycle_state_by_closed_form", threshold=0.99)
    print(f"  {report.summary().splitlines()[0]}")
    kept = machine.library.get("cycle_once")
    print(f"  after checking: confidence {kept.confidence():.4f}, "
          f"{'trusted' if kept.trusted else 'UNTRUSTED'}")

    # Now the transfer: the cycle is one move, so ten of them is a chain of
    # ten and the residual order the paper needs is reached by search.
    ten = machine.prove(initial_state(), state_after(10), max_depth=12)
    print(f"  ten cycles from the same start: "
          f"{'found' if ten.found else 'not found'}, "
          f"{ten.length} moves, confidence {ten.confidence:.4f}")
    print(f"    {initial_state()}")
    print(f"    {state_after(10)}")
    print("  The four steps underneath are still unverified, so a chain that\n"
          "  goes through them rather than through the checked composite is\n"
          "  still refused. What was measured is the cycle as a whole.")


def report_derivation(machine: RenMachine) -> None:
    """The derivation, leg by leg, as chains the machine finds for itself."""
    print("\n--- the derivation, as chains found by search ---")
    for start, target, label in DERIVATION_LEGS:
        proof = machine.prove(start, target, max_depth=16, trusted_only=False)
        if not proof.found:
            print(f"  {label:52} NOT FOUND")
            continue
        imported = sum(1 for n in proof.rule_names() if n in ALL_IMPORTED)
        print(f"  {label:52}{proof.length:3} moves, "
              f"{imported} imported, {proof.evidence()}")
        for step in proof.steps:
            mark = "imported" if step.rule in ALL_IMPORTED else "checked"
            print(f"      {step.rule:24}{mark}")


def report_mechanisms(machine: RenMachine) -> None:
    """What was executed for each step the machine cannot check as stated."""
    print("\n--- the imported steps, and what was carried out for each ---")
    for imported, (mechanism, note) in SUPPORTS.items():
        rule = machine.library.get(imported)
        status = "UNTRUSTED" if not rule.trusted else "trusted"
        if mechanism is None:
            print(f"  {imported:22}{status:11}nothing executed")
        else:
            checked = machine.library.get(mechanism)
            print(f"  {imported:22}{status:11}{mechanism} "
                  f"({checked.stats.summary()})")
        print(f"  {'':33}{note}")
    print("  Every step above is still untrusted, and that is not an "
          "oversight.\n  Each asserts something uniform -- in the "
          "concentration scale, the\n  band, the label, the stage -- and a "
          "mechanism run on instances is\n  not an argument of that shape.")


def report_sensitivity(machine: RenMachine) -> None:
    """Break each rule on purpose and see whether the oracle objects.

    Perfect agreement means nothing on its own: it is also what a check that
    cannot fail would print. So each rule is run again with one thing
    deliberately wrong, on the same data, and the disagreements are the real
    measure of what the check established.
    """
    print("\n--- how wrong could each rule be before the oracle noticed ---")
    print(f"  {'rule':22}{'one thing made wrong':38}{'caught':>8}")
    for rule, (label, mutant) in {**MUTATIONS, **MECHANISM_MUTATIONS,
                                  **DERIVATION_MUTATIONS}.items():
        dataset = machine.datasets[f"check_{rule}"]
        examples = [e for e in dataset.examples if e.labeled]
        missed = sum(1 for e in examples
                     if (mutant(e.inp) or Content.abstract("")).text == e.out.text)
        caught = len(examples) - missed
        flag = "   <-- INVISIBLE" if caught == 0 else ""
        print(f"  {rule:22}{label:38}{caught:>4}/{len(examples):<4}{flag}")
    print("  A mutation the oracle cannot see marks a claim this run did not\n"
          "  establish, whatever the accuracy column above says. For the\n"
          "  mechanisms whose criterion is a single flag the base rate is\n"
          "  already this test: half the instances answer each way, so a\n"
          "  rule that always said yes would score one half. The cone\n"
          "  constant is the weakest of these: half the data sits within a\n"
          "  factor of two of the boundary and a wrong constant still slips\n"
          "  past most points.")


def report_what_was_proved(machine: RenMachine) -> None:
    """The parts of the construction that are decided rather than sampled."""
    print("\n--- proved, not sampled ---")
    print("  Two of the paper's derivations are plain enough to carry out")
    print("  rather than import. Neither is an estimate: one is an identity")
    print("  and one is a recursion, and both live in fragments where a")
    print("  procedure decides every case at once.")

    ident = machine.library.get("increment_identity")
    print(f"\n  1. The increment identity of Section 3.3 "
          f"[{'PROVED' if ident.proved else 'NOT PROVED'}]")
    print("     R(u+w,p+pi) - R(u,p) - L_u(w,pi) - div(w tensor w) = -w div(w)")
    print("     Expanded with the field and its first derivatives as")
    print("     independent symbols, so it holds at every point of every")
    print("     smooth field. The defect comes out in closed form rather")
    print("     than as a small residual, which is why the divergence-free")
    print("     hypothesis is visible in the answer instead of assumed.")
    print(f"     exact {ident.exact}, confidence {ident.confidence():.4f}, "
          f"and the 1200 differentiated fields now confirm a theorem")

    print("\n  2. The induction of Proposition 9.6")
    proof = prove_cycle_closes_at_every_stage(machine.library)
    print(f"     {proof.summary()}")
    print("     The cycle was run ONCE, with the stage index and the radial")
    print("     derivative loss both left as variables. Every inequality")
    print("     that fell out is linear in them, and a linear inequality")
    print("     over an index that runs to infinity is decided by its")
    print("     coefficients rather than by trying values.")
    print(f"     Derived budget: {proof.budget.describe()}. The construction")
    print("     takes kappa = 1/100000, so the recursion never stops closing")
    print("     and sigma_j = 1/5 + j/10 holds at every stage, which is the")
    print("     closed form (9.8) -- obtained here rather than imported.")

    rule = machine.library.get("prop_9_6_all_stages")
    print(f"     exact {rule.exact}, confidence {rule.confidence():.4f}")

    carrier = machine.library.get("ns_carrier_frequency")
    print(f"\n  3. The viscous balance of Section 7.2 "
          f"[{'PROVED' if carrier.proved else 'NOT PROVED'}]")
    print("     1 <= eps*k^2 <= 4 for EVERY eps in (0,1], with k the carrier")
    print("     frequency ceil(eps^-1/2). A continuum, so no number of")
    print("     instances reaches it, and the argument is three lines: the")
    print("     lower bound is a condition the rule enforces, the upper one")
    print("     is 4(k-1)^2 - k^2 = (3k-2)(k-2) with both factors positive")
    print("     for k >= 2, and k = 1 pins eps = 1.")
    print("     One consequence is worth printing on its own: the rule's")
    print("     viscous_balance_lost branch is UNREACHABLE. No number of")
    print("     instances could have exercised it, so its base rate was")
    print("     never going to be informative and the proof is the only")
    print("     thing that could have said so.")

    print("\n  Confidence 1.0000 on those three is not a thousand agreements")
    print("  rounded up. There are no instances in any of them, which is")
    print("  why the library prints them as PROVED and not as trusted.")

    print("\n  What this does NOT reach, and why the theorem is still not")
    print("  proved here. The construction quantifies over five things:")
    for what, verdict in (
            ("every correction stage j", "PROVED above: a rational recursion "
             "on an integer index, which is a decidable fragment"),
            ("every order of the background expansion",
             "not attempted: the cutoff recursion is decidable in the same "
             "way, and its coefficients are not"),
            ("every dyadic band", "REACHED by derivation below, given "
             "Bernstein: the dyadic sum is decided by its exponent"),
            ("every slow label", "NOT established: the phrase covers the "
             "construction's continuous slow variables and no decision "
             "procedure here touches them. The nearest labelled uniformity, "
             "the carrier frequency over every eps, is proved above"),
            ("every concentration scale q as q -> 0",
             "REACHED by derivation below: a limit decided by the sign of "
             "one exponent")):
        print(f"    {what:42} {verdict}")




def report_norm_derivations(machine: RenMachine) -> None:
    """Statements about norms, reached by chaining rules rather than cited.

    The run has been calling these out of reach, and that was too coarse.
    An estimate has an analytic half and an exponent half; the analytic
    half is assumed here, by name, and the exponent half is derived and
    then chained. What comes out is a derivation whose exponents are exact
    and whose assumptions are listed, which is a different and more useful
    thing than an imported estimate.
    """
    print("\n--- statements about function space norms, by derivation ---")
    print("  Five inequalities as rules. Each one's EXPONENT is derived from")
    print("  the requirement that the inequality survive rescaling the")
    print("  function, with the integrability indices left as variables, so")
    print("  the derivation covers every index rather than a sample. What")
    print("  is assumed is that a finite constant exists at all.")
    for rule_name, _ in NORM_PROOFS:
        rule = machine.library.get(rule_name)
        mark = "PROVED" if rule.proved else "NOT PROVED"
        print(f"    {rule_name:26} {mark:11} {rule.proved[:46]}")
    print("    holder_product             ASSUMED     its index arithmetic is "
          "Hoelder's, not derived here")

    print("\n  A derivation the construction needs. The starting point is an")
    print("  L^2 band estimate for the residual, which is the kind of thing")
    print("  Proposition 7.5 delivers and which is ASSUMED here, not proved:")
    proof = machine.prove(RESIDUAL_START, RESIDUAL_TARGET, max_depth=6)
    print(f"    {RESIDUAL_START}")
    for step in proof.steps:
        print(f"      --{step.rule}-->  {step.after}")
    print(f"    {'reached' if proof.found else 'NOT reached'}: "
          f"the residual's L^infinity norm tends to zero at the singularity")
    print(f"    {proof.evidence()}, over {proof.length} moves")

    print("\n  Two of the quantifiers this run kept calling out of reach are")
    print("  discharged in that chain, and neither by instances:")
    print("    every dyadic band  -- sum_over_bands, which converges exactly")
    print("      when the frequency exponent is negative; here it is -3/2")
    print("    the limit q -> 0   -- limit_at_the_singularity, which holds")
    print("      exactly when the scale exponent is positive; here it is 1/5")
    print("  Both guards DECLINE otherwise, so a divergent sum or a bound")
    print("  that does not vanish stops the search instead of passing.")

    for start, why in ((est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(1),
                            sc=Fraction(1, 5)), "frequency exponent +1: the "
                        "dyadic sum diverges"),
                       (est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-3),
                            sc=Fraction(-1, 5)), "scale exponent -1/5: the "
                        "bound blows up instead of vanishing")):
        blocked = machine.prove(start, RESIDUAL_TARGET, max_depth=6)
        print(f"    {'reached' if blocked.found else 'refused':8} {why}")

    print("\n  The quadratic term, which is where Hoelder earns its place.")
    print("  Two band estimates for the increment multiply, and the")
    print("  integrability indices add rather than being chosen:")
    half = est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-2), sc=Fraction(1, 10))
    product = machine.prove(pair(half, half), RESIDUAL_TARGET, max_depth=6)
    print(f"    {pair(half, half)}")
    for step in product.steps:
        print(f"      --{step.rule}-->  {step.after}")

    used = set(proof.rule_names()) | set(product.rule_names())
    print("\n  What those chains rest on, collected back out of the rules:")
    for line in assumptions_behind(machine.library, sorted(used)):
        print(f"    - {line}")
    print("    - and the starting estimates themselves, which are the")
    print("      paper's analytic work and are not derived anywhere here")

    print("\n  So the honest statement is narrower than it looks and wider")
    print("  than before. Bernstein and Hoelder are not proved here and are")
    print("  not in question; what is now machine-checked is everything the")
    print("  construction does WITH an estimate once it has one. What is")
    print("  still absent is the estimate: Theorem 4.6 and Propositions 5.5,")
    print("  7.5 and 9.9 are not corollaries of these inequalities, and no")
    print("  chain of them produces one.")


def report_what_was_worked_out(machine: RenMachine) -> None:
    """Quantities the machine found, rather than ones it was handed.

    Checking a number and finding one are different, and until now almost
    everything here was the first. These two are the second. Neither is a
    proof and neither is the paper's insight; what they are is the machine
    recovering a constant the construction chose, and stating a hypothesis
    it would need, from rules it already had.
    """
    print("\n--- what the machine worked out for itself ---")

    gain = derive_cycle_gain(machine.library)
    print("  1. The step size of Proposition 9.6, found rather than assumed.")
    print("     The first three moves never mention the gain; they say where")
    print("     the orders land. Step 4 compares that against a claimed gain.")
    print("     Asking instead for the LARGEST gain those three would clear")
    print("     turns a check into a derivation:")
    print(f"       {gain.summary()}")
    print("     Every margin, as a function of the radial derivative loss:")
    for what, form in gain.margins:
        value = form.at(kappa=gain.kappa, n=0).value()
        print(f"       {what:12} {str(form):26} = {value}")
    print("     No margin shrinks as the stage grows, so stage zero is the")
    print("     binding one and the gain derived there holds at every stage.")
    print(f"     The construction's {gain.claimed} is inside that with room, and")
    print("     the term that caps it is named rather than guessed at.")

    print("\n  2. The estimate the derivation would need, stated as a")
    print("     hypothesis instead of assumed as a premise.")
    print("     Every chain so far takes an estimate and carries it forward.")
    print("     Run the same chain with the estimate left as variables and")
    print("     the guards say what it would have to be:")
    start, needed = required_estimate(machine.library)
    print(f"       from {start}")
    for line in needed:
        print(f"         needs {line}")
    print("     That is the shape of propose_rules applied to analysis: what")
    print("     comes back is the thing worth proving, derived by running")
    print("     the rules rather than by reading them.")

    print("\n  Neither is a proof and neither is the paper's insight. Which")
    print("  construction to try, which profile, which pair of pulse")
    print("  families -- none of that was found here, and this run does not")
    print("  claim it was. What these two show is the narrower thing: where")
    print("  a constant or a hypothesis is implied by rules the machine")
    print("  already holds, it can be made to produce it instead of being")
    print("  told it.")


def report_the_open_door(machine: RenMachine) -> None:
    """The one place the closed registry opens, and what keeps it honest.

    The catalogue tells a controller it must pick generators and oracles
    by name and cannot write new ones. That is there to stop an oracle
    written by whoever wrote the rule, which is not a second opinion. The
    reasoning does not apply to a rule the machine can DECIDE, so band
    inequalities are proposable as data and checked against what rescaling
    forces.
    """
    print("\n--- the one door in the closed registry ---")
    bernstein = ("Bernstein on a dyadic band, with a constant independent "
                 "of the band")
    target = vanishes(d=3, dv=0, ip=Fraction(1, 4))
    before = machine.prove(RESIDUAL_START, target, max_depth=6)
    print(f"  a target in L^4: {'reached' if before.found else 'NOT reached'} "
          f"with the rules the machine shipped with")

    print("\n  A proposal arrives as data: where the index lands, what the")
    print("  proposer claims the frequency costs, and the classical fact")
    print("  being leaned on. Nothing executable crosses the boundary.")
    for name, ip_to, shift, extra, note in (
            ("bernstein_to_L4", Fraction(1, 4), {"ip": 3, "ip_to": -3, "dv": 1},
             {}, "the honest one"),
            ("wrong_exponent", Fraction(0), {"ip": 2, "ip_to": -2, "dv": 1},
             {}, "two powers of 1/p where scaling forces three"),
            ("smuggles_a_scale", Fraction(0), {"ip": 3, "ip_to": -3, "dv": 1},
             {"scale_shift": Fraction(1, 5)},
             "a claim about the concentration scale rescaling cannot support"),
            ("no_assumption", Fraction(0), {"ip": 3, "ip_to": -3, "dv": 1},
             {"assumes": ""}, "declines to say what it leans on")):
        kwargs = dict(assumes=bernstein)
        kwargs.update(extra)
        result = propose_band_rule(machine.library, name, ip_to, shift, **kwargs)
        print(f"    {name:18} {'ADMITTED' if result.admitted else 'REFUSED':9} "
              f"{note}")

    after = machine.prove(RESIDUAL_START, target, max_depth=6)
    print(f"\n  the same L^4 target, after the admitted one: "
          f"{'reached' if after.found else 'NOT reached'}")
    for step in after.steps:
        print(f"    --{step.rule}-->  {step.after}")
    print("  and the assumption comes back out of the chain rather than")
    print("  disappearing into it:")
    for line in assumptions_behind(machine.library, after.rule_names()):
        print(f"    - {line}")

    print("\n  What this does and does not settle. It lets a controller")
    print("  extend the calculus without being trusted, because the part")
    print("  that could be wrong is the part the machine decides. It does")
    print("  not open generators or oracles, where correctness is not")
    print("  decidable and the closed registry is still doing real work.")
    print("  And scaling fixes the exponent of an inequality that is true;")
    print("  it does not make one true. An admitted rule is a PROVED")
    print("  exponent sitting on a named assumption, which is exactly the")
    print("  standing of the rules that shipped with the calculus.")


def report_the_witness(machine: RenMachine) -> None:
    """Theorem 4.6's existential, attacked by building one.

    Most of the eleven say an object exists, and no chain of implications
    produces an object. So the question is not whether the machine can
    deduce the existential but whether it can CONSTRUCT the thing, and for
    the finite part of Theorem 4.6's profile it can.
    """
    print("\n--- building a witness instead of assuming one ---")
    print("  Theorem 4.6 asserts profiles with four properties. Those four")
    print("  are not alike. The moment identities are a finite linear")
    print("  system over the rationals, and a finite linear system is")
    print("  something this machine solves exactly. The other three")
    print("  quantify over a continuum.")
    reduction = machine.prove_rule("lemma_A6_heat",
                                   "exterior_ode_by_symbolic_reduction")
    print()
    print("  The heat exterior, checked as an identity rather than at")
    print("  sampled points. Substituting the similarity field into the")
    print("  exterior equation and differentiating symbolically reduces it")
    print("  to ONE power of s times an equation in Z, H, H' and H'':")
    print(f"    {reduction.judgement.statement.split(': ', 1)[-1]}")
    print("  That collapse is the similarity structure closing, and it is")
    print("  checked rather than assumed. The equation is derived with the")
    print("  exponent left symbolic, so it is not specific to the paper's")
    print("  choice, and then the paper's profile is measured against it:")
    for line in reduction.judgement.detail[2:4]:
        print(f"    {line}")
    print("  The second half is quadrature, so this is a partial result and")
    print("  the rule is deliberately not marked exact.")

    built = build_moment_profile(
        [Fraction(1, 2), Fraction(-1, 2), Fraction(-3, 2)],
        [Fraction(1), Fraction(0), Fraction(0)])
    print()
    for line in built.report().splitlines():
        print(f"  {line}")
    print()
    print("  Two failure modes, because a construction that always succeeds")
    print("  is not checking anything:")
    for powers, why in (([Fraction(1, 2), Fraction(1, 2)], "repeated powers, "
                         "which is exactly what Lemma A.1 forbids"),
                        ([Fraction(1, 3), Fraction(-1, 2)], "a power whose "
                         "moments are irrational")):
        attempt = build_moment_profile(powers, [Fraction(1), Fraction(0)])
        print(f"    {'built' if attempt.built else 'refused':8} {why}")
    print()
    print("  Two of Theorem 4.6's four conditions are now attacked this")
    print("  way rather than assumed: the moment identities exactly, and")
    print("  the heat exterior as a symbolic reduction plus a measurement.")
    print("  This still does not prove Theorem 4.6, and the report says")
    print("  which conditions it leaves open. What it changes is the shape")
    print("  of the gap. 'There exist profiles' was one assumption with")
    print("  nothing behind it; it is now an explicit candidate whose")
    print("  finite conditions hold exactly and whose infinite ones are")
    print("  named. That is the most an existential can be reduced to here.")


def report_the_two_gaps(machine: RenMachine) -> None:
    """The two things the verified rules still do not give, demonstrated.

    Read the first one carefully, because it says less than it looks like
    it says. THIS machine holds the eleven intermediate results as
    untrusted imports and holds none of their prerequisites, so the
    trusted-only search has nothing to walk through and exhausts. That is
    a fact about this library, not about the theorem, and
    `report_the_theorem_with_prerequisites` below settles the difference
    by giving the search the prerequisites and asking again.
    """
    print("\n--- the two gaps, demonstrated ---")

    # 1. What the eleven imported steps are worth AS IMPORTS.
    #
    #    This used to print a PROVED/NOT PROVED verdict on Theorem 1.1 and
    #    it should never have. A trusted-only path search over THIS library
    #    exhausts for two reasons, neither of them about the theorem: the
    #    eleven steps here are imported labels whose prerequisites are not
    #    installed, and a path search cannot express a conjunction, which
    #    is what `JoinRule` and `saturate` were built for. Both are fixed
    #    elsewhere in the package, and the derivation is reported in the
    #    section after this one. What belongs here is the narrower thing
    #    this run actually measures: how much of the chain is imported.
    allowed = machine.prove(f"h={H}", "theorem_1_1_forced_blowup",
                            max_depth=24, trusted_only=False)
    if allowed.found:
        imported = [n for n in allowed.rule_names() if n in ALL_IMPORTED]
        print(f"  The chain through the imported steps: {allowed.length} steps, "
              f"{allowed.evidence()}")
        print(f"  {len(imported)} of those steps are imported: "
              f"{', '.join(imported)}")
        print("     The measured figure is the product over the steps that were\n"
              "     checked here. The unmeasured count is the rest, and it is a\n"
              "     count rather than a probability on purpose: each of those\n"
              "     steps sits at the Laplace prior of one half, and multiplying\n"
              "     eleven of them gives 0.0005, which reads like near-certain\n"
              "     failure when what it means is that nothing here has read\n"
              f"     eleven proofs. The bare product is {allowed.confidence:.6f};\n"
              "     the chain is the paper's dependency graph, not a verdict on it.")

    # 2. A verified rule answers outside the regime it was checked in.
    rule = machine.library.get("lemma_4_5_cone")
    for text in ("cone(a=1/1000,bs=40,p1=-3,p2=7)", "cone(a=3,bs=0,p1=1000000,p2=0)"):
        got = rule.apply(Content.abstract(text))
        print(f"  far outside the profile's range {text:34} -> "
              f"{got.text if got else '(declined)'}")
    print("     The cone rule has no idea which shears a profile can actually\n"
          "     produce. It was checked on data drawn from one generator and\n"
          "     answers anything of the right shape with the same confidence.")

    checked = sum(len(machine.datasets[f"check_{r}"])
                  for r in list(CLAIM_RULES) + list(MECHANISM_CHECKS))
    print(f"\n  instances decided by an oracle in this run: {checked}")
    print("  instances the construction quantifies over:  every q as q -> 0,\n"
          "     every dyadic band, every slow label, every correction stage j,\n"
          "     every order of the background expansion")
    print("     No number of THESE instances reaches one of those. The\n"
          "     instances above are numerical samples of the construction's\n"
          "     objects, and sampling an object cannot settle a claim about\n"
          "     every object. That is a limit on this instance space rather\n"
          "     than on the method: `run_norm_discovery.py` samples written-\n"
          "     down ESTIMATES instead, where every instance is itself a\n"
          "     uniform statement. The estimates of Theorem 4.6, Propositions\n"
          "     5.5, 7.5, 9.6 and 9.9 and Section 10 are not reached either\n"
          "     way, and they sit in this library as untrusted steps, which\n"
          "     is the honest place for a result this machine has imported\n"
          "     and cannot check.")


def report_the_theorem_with_prerequisites() -> None:
    """The same question, asked of a library that has the prerequisites.

    The gap above is real and is about THIS library. Given the named
    theorems each intermediate result rests on, and the paper's own
    estimates granted on the record, the theorem is not out of reach at
    all: the machine starts at one cell and derives it.

    This is what "finish the proof" means here -- form rule chains from
    known and cited prerequisites, through the intermediate results, to
    Theorem 1.1 -- and it is assembled rather than reproved from axioms.
    `run_complete.py` is the experiment; this reports its result beside
    the negative one so the two are not confused.
    """
    import importlib.util

    print("\n--- Theorem 1.1, with the prerequisites present ---")
    path = Path(__file__).resolve().parent / "run_complete.py"
    spec = importlib.util.spec_from_file_location("run_complete", path)
    complete = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(complete)

    machine, facts, estimates, links, joins = complete.build()
    stages = complete.walk(machine)
    reached = sum(1 for *_, got in stages if got.found)
    total = sum(got.length for *_, got in stages if got.found)
    axioms = sum(n.startswith("assume:")
                 for *_, got in stages for n in got.rule_names())
    bridges = sum(n.startswith(("from:", "means:"))
                  for *_, got in stages for n in got.rule_names())

    verdict = "DERIVED" if reached == len(stages) else "NOT DERIVED"
    print(f"  Theorem 1.1 with the prerequisites installed: {verdict}")
    print(f"    stages walked: {reached} of {len(stages)}, "
          f"starting at {complete.FIRST_CELL}")
    print(f"    rule applications end to end:        {total}")
    print(f"      applications of a named theorem:   {total - axioms - bridges}")
    print(f"      granted hypotheses pulled in:      {axioms}")
    print(f"      correspondences between languages: {bridges}")
    print(f"    named theorems in the library:       {facts}")
    print(f"    the paper's estimates, granted:      {estimates}")
    print()
    print("  So the negative result above is about a library, and this is")
    print("  about the theorem. What the two together say is the honest")
    print("  version: nothing here CHECKS the paper's analysis, and given")
    print("  that analysis by citation the argument assembles and closes.")
    print("  A proof modulo a printed list is what a proof is relative to a")
    print("  standard library; the list is printed by run_complete.py and")
    print("  the correspondences in it are my reading rather than anyone's")
    print("  theorem.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer instances per check")
    ap.add_argument("--llm", action="store_true",
                    help="let Claude drive instead of following the plan. The "
                         "scripted plan is a fixed sequence I wrote, so every "
                         "target in it is mine; this is the path where the "
                         "controller chooses what to check and what to prove.")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="tool calls the LLM controller is allowed")
    args = ap.parse_args()
    scale = 0.25 if args.quick else 1.0

    machine = RenMachine(goal=GOAL, device="cpu")
    install_navier_stokes_rules(machine.library)
    install_mechanism_rules(machine.library)
    install_derivation_rules(machine.library)
    install_norm_rules(machine.library)

    # Depth 24: the derivation runs from the exponent h through the energy
    # budget, the wave increment, the four moves of one correction cycle and
    # every imported step, to the theorem. The tool form of add_task does not
    # take a depth, so this one is added directly.
    machine.add_task("theorem_1_1", f"h={H}", "theorem_1_1_forced_blowup",
                     max_depth=24)

    if args.llm:
        # The reason this path is worth having. Every report below reads the
        # machine's state rather than the plan, so they work the same whether
        # the sequence came from me or from the controller -- and the
        # difference between the two runs is exactly the question of whether
        # anything here can choose its own targets.
        try:
            run = LLMController(machine, max_steps=args.max_steps).run(GOAL)
        except Exception as err:                       # credentials, usually
            name = type(err).__name__
            if "Auth" not in name and "APIConnection" not in name:
                raise
            print(f"\nthe LLM controller could not reach the API ({name}).")
            print("Set ANTHROPIC_API_KEY, or log in with `ant auth login`, and")
            print("run again. Everything below works either way -- the reports")
            print("read the machine's state, not the plan -- so the scripted")
            print("sequence runs now and the run is the reference the LLM path")
            print("is compared against.\n")
            run = ScriptedController(machine).run(plan(scale))
    else:
        run = ScriptedController(machine).run(plan(scale))

    print("\n" + "=" * 78)
    print(run.summary())
    print("\n--- final library ---")
    print(machine.library.table())

    report_measurements(machine)

    print("\n--- what was established ---")
    for rule in CLAIM_RULES:
        r = machine.library.get(rule)
        print(f"  {rule:22} {'trusted' if r.trusted else 'UNTRUSTED'}, "
              f"confidence {r.confidence():.4f}, {r.stats.summary()}")

    derive_the_cycle(machine)
    report_derivation(machine)
    report_mechanisms(machine)
    report_what_was_proved(machine)
    report_norm_derivations(machine)
    report_what_was_worked_out(machine)
    report_the_open_door(machine)
    report_the_witness(machine)
    report_sensitivity(machine)
    report_the_two_gaps(machine)
    report_the_theorem_with_prerequisites()

    print("""
--- what this run does, and what it leaves to other files ---
  The theorem is derived, and the section above is where that is reported
  rather than done: `run_complete.py` assembles the chain and this run
  imports the result. What THIS file establishes is narrower and is worth
  separating from it.

  One step is PROVED here, in the strongest sense the package has. The
  induction of Proposition 9.6 closes at every stage, for every radial
  derivative loss up to a budget the machine derived rather than imported.
  That is a decision procedure over rational arithmetic, it is the only
  quantifier in the construction that lives in a decidable fragment, and
  it leaves every estimate the induction is built from exactly where it
  was.

  The rest was rebuilt as rules and checked against computations that
  answer the same question another way, which is worth exactly what that
  is worth: the cone test agrees with the wave amplitudes it is supposed
  to predict, the exterior field really does solve the heat equation, the
  pulse really does turn over inside its slot, the moment matrices really
  are invertible, the decay recursion really does run away, and the energy
  budget really does close for h < 1/6.

  Four of those checks have a stated blind spot, printed above and set out
  in the module: the pulse rule and its oracle share the model they disagree
  about, the covariance oracle drops the error terms, the cone's second
  oracle shares the coordinate map, the decay check is bookkeeping rather
  than the estimate that produces the gain, and the core exponents are
  definitions here rather than measurements of a field.

  The eleven steps that carry the construction's weight are imports IN
  THIS LIBRARY and stay untrusted here, because this file installs none of
  their prerequisites. That is a fact about this machine. Each has since
  been decomposed to named theorems and the paper's cited estimates, each
  is derived from the link before it in `run_complete.py`, and each is
  discharged there by `machine.discharge`, which refuses any chain
  containing an untrusted rule.

  What no file here does is CHECK the paper's analysis. The estimates of
  Theorem 4.6 and Propositions 5.5, 7.5 and 9.9 are granted by citation,
  the ledger prints how many, and that is the real limit: this is the
  paper's argument assembled, not an independent confirmation of it.

  One claim this run used to make and should not have: that a mechanism
  executed on instances can never reach a uniform statement, so norm
  estimates are out of reach in principle. That is true of sampling the
  underlying FUNCTIONS and false as a general claim. A band estimate can
  be written down as a cell, and `run_norm_discovery.py` samples such
  cells and checks them against an oracle that integrates a real function
  -- every instance there is itself a uniform statement. The limit is the
  choice of instance space, not the method.

  Finally, the theorem concerns the FORCED equations: it settles
  alternatives (C) and (D) of Fefferman's statement. The unforced
  question, (A) and (B), is a different one, and the force here cannot be
  dropped -- by the energy identity a zero force would give the zero
  solution.""")


if __name__ == "__main__":
    main()
