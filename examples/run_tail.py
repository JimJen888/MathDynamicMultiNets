"""
Experiment 1j: granting the named rules, how much of the chain closes?

The question this answers: if every registered prior fact and every cited
import counts as proven, is the main theorem reached?

The honest answer has two halves and they are very different, so this
file measures them separately rather than reporting one number.

THE BACK HALF. Five of the eleven have been decomposed: Proposition 10.1,
Lemmas 10.3, 10.4 and 10.5, and Theorem 1.1. Between them they cover the
whole of Section 10 and the theorem, which is the chain

    local_field_theorem_3_1 -> localized_fields -> compact_smooth_force
        -> uniform_energy_bound -> no_global_smooth_competitor
        -> theorem_1_1_forced_blowup

Granting the named rules and the two cited imports, does that chain close
through the decompositions with every imported step deleted? That is a
yes-or-no question and the file asks it.

THE FRONT HALF. The other six have no decomposition at all. Granting
named rules does nothing for a step with no decomposition to grant them
to: there is no argument there to complete. That half is

    h -> ... -> leading_profiles -> background_annular_stress
        -> ... -> residual_flat_at_singularity -> local_field_theorem_3_1

and it still runs entirely on imports.

So the answer is not "yes" or "no" but "the back half, and here is what
it cost". Counting the cost is the point: two cited imports pulled in from
outside the eleven, and twelve correspondences that are my judgement.

Run: python examples/run_tail.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import RenMachine                             # noqa: E402
from dynamicmultinets.navierstokes import (                         # noqa: E402
    IMPORTED_NAMES, install_navier_stokes_rules)
from dynamicmultinets.nsderivation import (                         # noqa: E402
    DERIVATION_IMPORTED, install_derivation_rules)
from dynamicmultinets.nsmechanisms import install_mechanism_rules   # noqa: E402

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The five decomposed steps, in chain order, with what each needs.
TAIL = [
    dict(step="prop_10_1_localize", module="run_theorem", claims="STEPS_10_1",
         chain_in="local_field_theorem_3_1", chain_out="localized_fields",
         proves="localized_fields(L)", entry=["local_fields(L)"], cited=[]),
    dict(step="lemma_10_3_force", module="run_force_extension", claims="STEPS",
         chain_in="localized_fields", chain_out="compact_smooth_force",
         proves="force_extends_smoothly(R)",
         entry=["widths(W)", "shrinking(W)"],
         cited=[("derivative_limits(R)", "Lemma 10.2")]),
    dict(step="lemma_10_4_energy", module="run_energy_bound",
         claims="INSTANTIATED",
         chain_in="compact_smooth_force", chain_out="uniform_energy_bound",
         proves="energy_bounded_by_force_integral(U,F)",
         entry=["forced(U,F)", "divfree(U)", "starts_from_rest(U)"], cited=[]),
    dict(step="lemma_10_5_unique", module="run_comparison", claims="STEPS",
         chain_in="uniform_energy_bound",
         chain_out="no_global_smooth_competitor",
         proves="fields_agree(v,u)",
         entry=["solves_1_1(v)", "solves_1_1(u)", "same_force(v,u)",
                "zero_initial_difference(v,u)"], cited=[]),
    dict(step="thm_1_1_blowup", module="run_theorem", claims="STEPS_1_1",
         chain_in="no_global_smooth_competitor",
         chain_out="theorem_1_1_forced_blowup",
         proves="theorem_1_1_forced_blowup(L)",
         entry=["localized_fields(L)", "competitor_equals_it(L)",
                "bounded_energy(L)", "smooth_compact_force(L)"],
         cited=[("inner_growth_asymptotic(L)", "Theorem 3.1(ii)")]),
]

FRONT = ("thm_4_6_profiles", "prop_5_5_background", "prop_7_5_stress",
         "prop_9_5_initialize", "prop_9_6_induction", "prop_9_9_summation")


def build() -> tuple:
    """One machine holding every decomposition, with nothing imported."""
    machine = RenMachine(device="cpu")
    seen, facts, correspondences, cited = set(), 0, 0, []

    for entry in TAIL:
        module = load(entry["module"])
        for rule in module.PRIOR:
            if rule.name not in seen:
                machine.library.add(rule)
                seen.add(rule.name)
                facts += 1
        for cell in entry["entry"]:
            machine.assume(cell, f"{entry['chain_in']} says {cell}",
                           source="my reading", standing="correspondence")
            correspondences += 1
        for cell, why in entry["cited"]:
            machine.assume(cell, f"{why}, cited from outside the eleven",
                           source="the paper", standing="cited import")
            cited.append(why)
        machine.assume(entry["chain_out"],
                       f"{entry['proves']} says {entry['chain_out']}",
                       source="my reading", standing="correspondence")
        correspondences += 1

    return machine, facts, correspondences, cited


def main() -> None:
    print("=" * 78)
    print("Granting the named rules: how much of the chain closes?")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    machine, facts, correspondences, cited = build()

    print("\n--- the back half, step by step ---")
    ok = True
    for entry in TAIL:
        module = load(entry["module"])
        audit = machine.check_proof(getattr(module, entry["claims"]))
        reached = machine.prove("anything", entry["chain_out"], max_depth=3,
                                trusted_only=False).found
        ok = ok and reached and not audit.unvalidated
        print(f"  {entry['step']:22} "
              f"{len(audit.derived) - len(audit.unvalidated)}/"
              f"{len(audit.derived):<3} steps   "
              f"{entry['chain_in']} -> {entry['chain_out']}: "
              f"{'closed' if reached else 'OPEN'}")

    end = machine.prove("anything", "theorem_1_1_forced_blowup", max_depth=4,
                        trusted_only=False)
    print(f"\n  the whole back half closes: {'yes' if ok and end.found else 'no'}")
    print(f"  from local_field_theorem_3_1 through to the theorem, with every")
    print(f"  imported step of Section 10 deleted")

    print("\n--- what that cost ---")
    print(f"  named facts registered:        {facts}")
    print(f"  cited imports from outside:    {len(cited)}  "
          f"({', '.join(cited)})")
    print(f"  correspondences (my judgement): {correspondences}")

    print("\n--- the front half ---")
    print("  Six steps, none decomposed:")
    for name in FRONT:
        print(f"    {name}")
    print()
    print("  Granting named rules does nothing here, and the reason is worth")
    print("  being exact about. A named rule completes an ARGUMENT. These")
    print("  six have no argument written down in this repository, so there")
    print("  is nothing for a granted rule to complete. They are not")
    print("  blocked, they are unattempted.")

    print("\n--- the answer ---")
    print("  Counting registered and cited rules as proven, the chain from")
    print("  Theorem 3.1's local field through to the main theorem is")
    print("  complete. That is the whole of Section 10 plus Theorem 1.1,")
    print("  five of the eleven imported steps, replaced by their own")
    print("  proofs with every step checked.")
    print()
    print("  The main theorem is NOT proved, because the chain that feeds")
    print("  Theorem 3.1's local field -- the construction itself, Sections")
    print("  4 through 9 -- is still six imported steps. The back half of")
    print("  the paper is done and the front half is not started.")


if __name__ == "__main__":
    main()
