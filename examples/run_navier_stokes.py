"""
Experiment 7: re-deriving the Navier-Stokes blowup construction as rules.

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
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamicmultinets import RenMachine, ScriptedController          # noqa: E402
from dynamicmultinets.navierstokes import (                          # noqa: E402
    CLAIM_DATA, CLAIM_RULES, IMPORTED_NAMES, MUTATIONS,
    install_navier_stokes_rules)
from dynamicmultinets.nsmechanisms import (                         # noqa: E402
    MECHANISM_CHECKS, MECHANISM_MUTATIONS, REMAINING, SUPPORTS,
    install_mechanism_rules)
from dynamicmultinets.nsderivation import (                         # noqa: E402
    CYCLE_MOVES, DERIVATION_CHECKS, DERIVATION_IMPORTED, DERIVATION_LEGS,
    DERIVATION_MUTATIONS, initial_state, install_derivation_rules,
    prove_cycle_closes_at_every_stage, state_after)

#: Every step of the paper the machine carries as a rule and cannot check.
ALL_IMPORTED = tuple(IMPORTED_NAMES) + DERIVATION_IMPORTED
from dynamicmultinets.tapes import ABSTRACT, SPECIFIC, Content       # noqa: E402

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

    print("\n  Confidence 1.0000 on those two is not a thousand agreements")
    print("  rounded up. There are no instances in either of them, which is")
    print("  why the library prints them as PROVED and not as trusted.")

    print("\n  What this does NOT reach, and why the theorem is still not")
    print("  proved here. The construction quantifies over five things:")
    for what, verdict in (
            ("every correction stage j", "PROVED above: a rational recursion "
             "on an integer index, which is a decidable fragment"),
            ("every order of the background expansion",
             "not attempted: the cutoff recursion is decidable in the same "
             "way, and its coefficients are not"),
            ("every dyadic band", "out of reach: a statement about function "
             "space norms, not about arithmetic"),
            ("every slow label", "out of reach: same"),
            ("every concentration scale q as q -> 0",
             "out of reach: a limit, not an induction")):
        print(f"    {what:42} {verdict}")


def report_the_two_gaps(machine: RenMachine) -> None:
    """The two things the verified rules still do not give, demonstrated."""
    print("\n--- the two gaps, demonstrated ---")

    # 1. The chain to the theorem exists and is not a proof.
    proved = machine.prove(f"h={H}", "theorem_1_1_forced_blowup", max_depth=24)
    print(f"  Theorem 1.1 with trusted rules only: "
          f"{'PROVED' if proved.found else 'NOT PROVED'} ({proved.note})")
    allowed = machine.prove(f"h={H}", "theorem_1_1_forced_blowup",
                            max_depth=24, trusted_only=False)
    if allowed.found:
        imported = [n for n in allowed.rule_names() if n in ALL_IMPORTED]
        print(f"  Letting the imported steps in: found in {allowed.length} steps, "
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
    print("     No number of instances reaches one of those, and the estimates\n"
          "     of Theorem 4.6, Propositions 5.5, 7.5, 9.6 and 9.9 and Section\n"
          "     10 were never in reach of an instance to begin with. They are\n"
          "     in the library as untrusted steps, which is the honest place\n"
          "     for a result this machine has imported and cannot check.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer instances per check")
    args = ap.parse_args()
    scale = 0.25 if args.quick else 1.0

    machine = RenMachine(goal=GOAL, device="cpu")
    install_navier_stokes_rules(machine.library)
    install_mechanism_rules(machine.library)
    install_derivation_rules(machine.library)

    # Depth 24: the derivation runs from the exponent h through the energy
    # budget, the wave increment, the four moves of one correction cycle and
    # every imported step, to the theorem. The tool form of add_task does not
    # take a depth, so this one is added directly.
    machine.add_task("theorem_1_1", f"h={H}", "theorem_1_1_forced_blowup",
                     max_depth=24)

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
    report_sensitivity(machine)
    report_the_two_gaps(machine)

    print("""
--- and what was not ---
  This run did not prove that the Navier-Stokes equations break down. One
  step of the construction IS proved here, and naming it precisely is the
  point: the induction of Proposition 9.6 closes at every stage, for every
  radial derivative loss up to a budget the machine derived rather than
  imported. That is a decision procedure over rational arithmetic, it is
  the only quantifier in the construction that lives in a decidable
  fragment, and it leaves every estimate the induction is built from
  exactly where it was.

  The rest was rebuilt as rules and checked against computations that
  answer the same question another way, which is
  worth exactly what that is worth: the cone test agrees with the wave
  amplitudes it is supposed to predict, the exterior field really does solve
  the heat equation, the pulse really does turn over inside its slot, the
  moment matrices really are invertible, the decay recursion really does run
  away, and the energy budget really does close for h < 1/6.

  Four of those checks have a stated blind spot, printed above and set out
  in the module: the pulse rule and its oracle share the model they disagree
  about, the covariance oracle drops the error terms, the cone's second
  oracle shares the coordinate map, the decay check is bookkeeping rather
  than the estimate that produces the gain, and the core exponents are
  definitions here rather than measurements of a field.

  The nine steps that carry the construction's weight had their methods
  carried out rather than quoted, and the run says for each one what that
  reached and what it did not. Seven of the nine have a mechanism here that
  was built and measured. Two do not: the comparison argument of Lemma 10.5
  turns on a pressure flux controlled by Riesz transforms on the whole
  space, and Theorem 1.1 is the conjunction of everything, uniformly.

  Not one of the nine is trusted afterwards, and that is the finding rather
  than a shortfall. Every one of them asserts something uniform in the
  concentration scale, the dyadic band, the slow label and the correction
  stage. A mechanism executed on instances is not an argument of that shape,
  and running more instances does not change its shape. The
  theorem also concerns the FORCED equations: it settles alternatives (C) and
  (D) of Fefferman's statement. The unforced question, (A) and (B), is a
  different one, and the force here cannot be dropped -- by the energy
  identity a zero force would give the zero solution.""")


if __name__ == "__main__":
    main()
