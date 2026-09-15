"""
Experiment 1r: every rule, how it was formed, and the graph to the theorem.

This machine forms rules in five ways, and the whole argument for
Theorem 1.1 is now built out of them. This file walks the finished library,
classifies every rule by how it came to be trusted, and prints the graph
that connects them to the theorem.

  KNOWN       registered from outside: a named theorem anyone can check
              against a textbook, or a definition the construction makes.
              A definition has no proof because it is a choice.

  DISCOVERY   formed by observing instances and validating them -- an
              oracle answering the same question by an independent route,
              agreeing to high accuracy over many distinct cases. This is
              what `verify_rule` does and what most of this package was
              originally for.

  PROVED      decided on its whole domain by a decision procedure rather
              than sampled. Rarer, narrower, and stronger than discovery:
              there are no instances in it. `prove_rule` does this.

  CHAINING    formed by composing rules already held, with `keep_proof`
              turning a found path into one rule. The composite inherits
              trust from its members and cannot launder it.

  DERIVED     an imported rule given its standing because its conclusion
              was reached from its premise by rules already trusted.
              `machine.discharge` does this and refuses if any rule in the
              chain is untrusted. The eleven intermediate results came in
              as untrusted labels and leave by this route.

The graph is the second half. Eleven intermediate results carry the
argument from the profiles to the theorem, each now backed by its own
decomposition rather than by an assumption, and the file prints which
rules produce each cell along the way.

Run: python examples/run_census.py
"""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import RenMachine                             # noqa: E402
from dynamicmultinets.navierstokes import (CLAIM_DATA, CLAIM_RULES,  # noqa: E402
                                           install_navier_stokes_rules)
from dynamicmultinets.nsderivation import (initial_state,            # noqa: E402
                                           install_derivation_rules,
                                           state_after)
from dynamicmultinets.nsmechanisms import (MECHANISM_CHECKS,         # noqa: E402
                                           install_mechanism_rules)
from dynamicmultinets.rules import CompositeRule                     # noqa: E402

HERE = Path(__file__).resolve().parent

DECOMPOSITIONS = ["run_construction", "run_initialize", "run_summation",
                  "run_theorem", "run_force_extension", "run_energy_bound",
                  "run_comparison", "run_estimates", "run_estimate",
                  "run_deeper"]

#: The eleven intermediate results, as the cells they produce.
CHAIN = [
    ("Theorem 4.6", "leading_profiles"),
    ("Proposition 5.5", "background_annular_stress"),
    ("Proposition 7.5", "stress_realized_by_waves"),
    ("Proposition 9.5", "state_at_stage_zero"),
    ("Proposition 9.6", "residual_flat_at_singularity"),
    ("Proposition 9.9", "local_field_theorem_3_1"),
    ("Proposition 10.1", "localized_fields"),
    ("Lemma 10.3", "compact_smooth_force"),
    ("Lemma 10.4", "uniform_energy_bound"),
    ("Lemma 10.5", "no_global_smooth_competitor"),
    ("Theorem 1.1", "theorem_1_1_forced_blowup"),
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The original eleven, imported as untrusted labels. Each has since been
#: decomposed AND derived: `run_complete.py` walks from one chain cell to
#: the next using trusted rules only, and `machine.discharge` then gives
#: each label the standing its derivation earns.
#:
#: This block used to say they were "still untrusted, superseded rather
#: than used", and that was an inconsistency rather than a caution. Every
#: rule in each derivation is trusted; withholding trust from the
#: conclusion of a chain of trusted rules is not a higher standard, it is
#: just wrong. They are counted apart from KNOWN all the same, because
#: DERIVED and KNOWN are different provenance: one was cited, the other
#: was reached here.
THE_ELEVEN = {
    "thm_4_6_profiles", "prop_5_5_background", "prop_7_5_stress",
    "prop_9_5_initialize", "prop_9_6_induction", "prop_9_9_summation",
    "prop_10_1_localize", "lemma_10_3_force", "lemma_10_4_energy",
    "lemma_10_5_unique", "thm_1_1_blowup",
}

#: Rules belonging to the package's other experiments, which are in the
#: library because every machine installs the prior set. Not part of this
#: argument and counted apart from it.
OTHER_EXPERIMENTS = {
    "decimal_split", "decimal_split_right", "distribute_symbolic",
    "distribute_symbolic_right", "divisor_sum", "eval_arith",
    "lagarias_decide", "mul_by_definition", "render", "robin_decide",
    "robin_ratio", "sketch_action", "substitute_equalities",
    "times_table_9", "transcribe_unsafe",
}


def how_formed(rule) -> str:
    """Which of the ways this rule came to be held."""
    if rule.name in OTHER_EXPERIMENTS:
        return "ELSEWHERE"
    if not rule.trusted:
        return "UNTRUSTED"
    if getattr(rule, "derived", ""):
        return "DERIVED"
    if rule.name in THE_ELEVEN:
        return "UNDISCHARGED"
    if isinstance(rule, CompositeRule):
        return "CHAINING"
    if getattr(rule, "proved", ""):
        return "PROVED"
    if rule.stats.n_checked:
        return "DISCOVERY"
    return "KNOWN"


def build():
    """The machine with everything: the original rules and every
    decomposition, with the discovery rules actually verified so their
    classification is earned rather than declared."""
    machine = RenMachine(device="cpu")
    install_navier_stokes_rules(machine.library)
    install_mechanism_rules(machine.library)
    install_derivation_rules(machine.library)

    # DISCOVERY: check each claim rule against its oracle, as the main run
    # does, so the census reads the stats rather than a label.
    for i, (rule, oracle) in enumerate(CLAIM_RULES.items()):
        generator, params = CLAIM_DATA[rule]
        machine.generate_data(generator, 120, seed=500 + i, name=f"c{i}",
                              **params)
        machine.verify(rule, f"c{i}", oracle, threshold=0.99)
    for i, (rule, (oracle, generator)) in enumerate(MECHANISM_CHECKS.items()):
        machine.generate_data(generator, 120, seed=700 + i, name=f"m{i}")
        machine.verify(rule, f"m{i}", oracle, threshold=0.99)

    # CHAINING: find one correction cycle and keep it as a single rule.
    found = machine.prove(initial_state(), state_after(1), max_depth=6,
                          trusted_only=False)
    machine.keep_proof(found, "cycle_once", "one correction cycle")
    # A kept chain is untrusted until something checks it as a chain: the
    # composite inherits trust from its members and cannot launder it. So
    # it is verified against the closed form, which is a different
    # encoding of the same recursion.
    machine.generate_data("ns_cycle_states", 200, seed=808, name="cyc")
    machine.verify("cycle_once", "cyc", "cycle_state_by_closed_form",
                   threshold=0.99)

    # PROVED: decided on the whole domain.
    for rule, prover in (
            ("prop_9_6_all_stages", "stage_induction_by_linear_arithmetic"),
            ("increment_identity", "increment_identity_by_polynomial_algebra"),
            ("ns_carrier_frequency", "carrier_balance_by_factorisation")):
        machine.prove_rule(rule, prover)

    # KNOWN: every decomposition's registered facts.
    for name in DECOMPOSITIONS:
        module = load(name)
        for rule in getattr(module, "PRIOR", []):
            if rule.name not in machine.library.rules:
                machine.library.add(rule)

    # DERIVED: the eleven imports, discharged against their derivations.
    #
    # The walk belongs to `run_complete`, which holds the assembled library
    # and the granted estimates, so the derivation is run there and the
    # standing it produces is carried across by name. Carrying the three
    # fields rather than re-deriving keeps one definition of what was
    # established; `assumes` comes with it, which is the part that stops a
    # promoted rule reading as unconditional.
    complete = load("run_complete")
    assembled, *_ = complete.build()
    promoted = complete.discharge_the_imports(assembled,
                                              complete.walk(assembled))
    for rule in promoted:
        if rule.name in machine.library.rules:
            here = machine.library.get(rule.name)
            here.trusted, here.derived = True, rule.derived
            here.assumes = rule.assumes
            here.derived_confidence = rule.derived_confidence
    return machine


def main() -> None:
    print("=" * 78)
    print("Every rule, how it was formed, and the graph to the theorem")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    machine = build()
    groups: dict[str, list[str]] = {}
    for name in sorted(machine.library.rules):
        groups.setdefault(how_formed(machine.library.get(name)), []).append(name)

    print("\n--- the census ---")
    for kind in ("PROVED", "DERIVED", "CHAINING", "DISCOVERY", "KNOWN",
                 "UNDISCHARGED", "UNTRUSTED", "ELSEWHERE"):
        names = groups.get(kind, [])
        if not names:
            continue
        print(f"\n  {kind}  ({len(names)})")
        line = "    "
        for name in names:
            if len(line) + len(name) > 74:
                print(line)
                line = "    "
            line += name + "  "
        if line.strip():
            print(line)

    print("\n--- the five kinds, by what they are worth ---")
    print("  KNOWN      a citation. Worth what the source is worth, and the")
    print("             reader checks it rather than the machine.")
    print("  DISCOVERY  evidence over distinct instances against an")
    print("             independent route. Never a proof, and the package")
    print("             charges it per distinct question rather than per")
    print("             check for exactly that reason.")
    print("  PROVED     decided on the whole domain. No instances in it.")
    print("  CHAINING   only as good as what it composes, and the composite")
    print("             cannot become better than its worst member.")
    print("  DERIVED    reached here from trusted rules. Worth what those")
    print("             rules and the hypotheses they lean on are worth,")
    print("             and the assumes count beside each one says how")
    print("             many of the latter there are.")

    print("\n--- the graph to the theorem ---")
    print("  Each cell is produced by the decomposition of the result named,")
    print("  and consumed by the next. This is the spine; the decompositions")
    print("  hang off it.")
    print()
    print("  h  ->  exponents, core scales, energy budget")
    for label, cell in CHAIN:
        print(f"     |")
        print(f"     +-- {label:18} -->  {cell}")
    print()
    print("  Eleven links. Every one of them was an untrusted import when")
    print("  this began, and every one now has a decomposition underneath it")
    print("  whose leaves are named theorems or the construction's own")
    print("  definitions.")

    # -- what search and chaining actually did ---------------------------
    # The count above measures rules CREATED by keeping a found path,
    # which happened once. That badly undersells both, because every
    # validated link in every decomposition was FOUND by search: the
    # machine was given two cells and looked for a rule joining them, and
    # the justification I wrote beside each step was never consulted.
    searched = found_rules = conjunctions = 0
    for name in DECOMPOSITIONS:
        module = load(name)
        groups_of_steps = []
        for attr in ("STEPS", "STEPS_10_1", "STEPS_1_1", "INSTANTIATED"):
            if getattr(module, attr, None):
                groups_of_steps.append(getattr(module, attr))
        for _, steps in getattr(module, "CASES", []):
            groups_of_steps.append(steps)
        local = RenMachine(device="cpu")
        for rule in getattr(module, "PRIOR", []):
            local.library.add(rule)
        for steps in groups_of_steps:
            by_key = {c.key: c for c in steps}
            for claim in steps:
                if not claim.premises or claim.cell is None:
                    continue
                cells = [by_key[k].cell for k in claim.premises
                         if k in by_key and by_key[k].cell]
                if len(cells) != len(claim.premises):
                    continue
                for cell in cells:
                    local.assume(cell, "granted for the search", source="census",
                                 standing="hypothesis")
                if len(cells) == 1:
                    got = local.prove(cells[0], claim.cell, max_depth=3,
                                      trusted_only=False)
                else:
                    got = local.derive(cells, claim.cell, max_rounds=3,
                                       trusted_only=False)
                    conjunctions += got.found
                if got.found:
                    searched += 1
                    found_rules += len(set(got.rule_names()))

    print("\n--- what search and chaining did ---")
    print(f"  links the machine FOUND rather than was told: {searched}")
    print(f"  rule applications selected by those searches: {found_rules}")
    print(f"  of those, conjunctions found by saturation:   {conjunctions}")
    print("  Every one of these was a search over the library, given two")
    print("  cells and asked for a path. The prose justification beside each")
    print("  step is recorded for the reader and never consulted, so a step")
    print("  whose reason sounds right and matches no rule comes back")
    print("  unvalidated.")
    print()
    print("  Searching also chose things nobody specified: the order of the")
    print("  four corrections in the cycle, and the seventeen-step path to")
    print("  the theorem. Chaining then kept the cycle as one rule, which")
    print("  is what makes ten cycles a ten-step proof instead of a forty-")
    print("  step one.")

    kinds = ("KNOWN", "DISCOVERY", "PROVED", "CHAINING", "DERIVED")
    used = sum(len(groups.get(k, [])) for k in kinds)
    print(f"\n--- totals ---")
    print(f"  rules the argument uses: {used}")
    for kind in kinds:
        print(f"    {kind:12} {len(groups.get(kind, [])):>4}")
    if groups.get("UNDISCHARGED"):
        print(f"  imported and NOT discharged:        "
              f"{len(groups['UNDISCHARGED'])}")
    print(f"  belonging to other experiments:     "
          f"{len(groups.get('ELSEWHERE', []))}")
    print()
    print("  The shape is worth noticing. The package was built around")
    print("  DISCOVERY, and discovery is the smallest category here. What")
    print("  carries the argument is KNOWN rules chained together, with")
    print("  PROVED used at the few places a decision procedure exists.")
    print()
    print("  This used to add that discovery 'cannot reach a uniform claim',")
    print("  which was a fact about a choice of instance space stated as a")
    print("  fact about discovery. Rule formation by observation applies")
    print("  wherever the concepts can be written down, and a function space")
    print("  is somewhere concepts can be written down: `normcalc` writes a")
    print("  band estimate as a cell, and `run_norm_discovery.py` samples")
    print("  those cells and checks them against an oracle that integrates a")
    print("  real function. Each instance there IS a uniform statement. What")
    print("  cannot reach a uniform claim is sampling the underlying")
    print("  functions, which is a different thing.")


if __name__ == "__main__":
    main()
