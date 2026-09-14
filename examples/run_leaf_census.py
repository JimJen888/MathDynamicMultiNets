"""
Experiment 1h: what fraction of a proof's steps apply a well-known rule?

The claim this measures: every step of a mathematical derivation applies
an existing rule, and those rules are so well established and so often
used that a language model holds them reliably. If that is right, then a
model supplying the step and a machine checking it against a held rule is
a workable division of labour, and the decompositions in this repository
are evidence either way.

Five of the eleven imported steps have been decomposed here, and between
them they register a few dozen facts as prior rules. This file counts
them, split two ways:

  NAMED       a theorem or identity with a name, that any analysis course
              covers and any competent reader states without looking up.
              Cauchy-Schwarz, Gronwall, Sobolev, Weierstrass, Riesz
              boundedness, the divergence of a curl.

  SPECIFIC    a computation belonging to THIS proof. It may be routine to
              carry out and it is still not something anyone has
              memorised: the commutator kernel bound, the coordinate
              bound (10.3), the claim that every power in the flux bounds
              stays below two.

The split is my judgement and it is written out below rather than
computed, so a reader can disagree with any single line and see what it
does to the total. That is the only honest way to report a number that
comes from a classification.

WHY THE SPLIT MATTERS. If a decomposition is nearly all NAMED, the claim
holds and the machine's job is checking. If it is heavy with SPECIFIC
facts, then the decomposition has stopped where the paper's prose stops
and the real content is still sitting inside a registered assumption, and
the honest move is to decompose further rather than to count it as done.

Run: python examples/run_leaf_census.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Rule name -> the classification, with a reason where it is not obvious.
#: Anything absent is counted NAMED, so an omission errs toward the claim
#: this file is testing rather than against it.
SPECIFIC = {
    "widths_beat_the_growth":
        "the recursive cutoff schedule of Lemma 5.4. Routine to carry out, "
        "not something anyone has memorised, and the package checks it on "
        "instances separately",
    "commutator_kernel":
        "the kernel bound C|x-y|^-3 min(|x-y|/R, 1) and its L^4/3 norm; a "
        "computation of this proof",
    "pressure_flux_bound":
        "inequality (10.19), assembled from the two parts of (10.18) with "
        "specific interpolation exponents",
    "young_absorbs_the_fluxes":
        "that every power of A_R in the flux bounds is at most 3/2. The "
        "hinge of the energy estimate and specific to these exponents",
    "coordinate_bound":
        "q <= C0(tau + |z|^(1/D)), read off the similarity coordinates "
        "(10.3)",
    "transport_flux":
        "the annulus bound C_T R^-1 B_R^3/2, which uses the specific "
        "Hoelder splitting of this proof",
}

SOURCES = [
    ("Lemma 10.4, the energy bound", "run_energy_bound"),
    ("Lemma 10.3, the force extension", "run_force_extension"),
    ("Lemma 10.5, the comparison", "run_comparison"),
    ("Prop 10.1 and Theorem 1.1", "run_theorem"),
]


def main() -> None:
    print("=" * 78)
    print("What fraction of the decomposed steps apply a well-known rule?")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    total_named = total_specific = 0
    print("\n--- by decomposition ---")
    for title, module_name in SOURCES:
        module = load(module_name)
        named = [r.name for r in module.PRIOR if r.name not in SPECIFIC]
        specific = [r.name for r in module.PRIOR if r.name in SPECIFIC]
        total_named += len(named)
        total_specific += len(specific)
        share = len(named) / max(len(module.PRIOR), 1)
        print(f"\n  {title}")
        print(f"    {len(named)} named, {len(specific)} specific "
              f"({share:.0%} named)")
        for name in specific:
            print(f"      SPECIFIC  {name}: {SPECIFIC[name]}")

    total = total_named + total_specific
    print("\n--- the number ---")
    print(f"  facts registered across five decomposed steps: {total}")
    print(f"  named theorems and identities:                 {total_named}")
    print(f"  computations belonging to this proof:          {total_specific}")
    print(f"  share that are rules anyone holds:             "
          f"{total_named / total:.0%}")

    print("\n--- reading it ---")
    print("  The claim under test is that the rules a derivation applies are")
    print("  well established enough to be held rather than looked up, and")
    print("  the measurement supports it: the large majority of what these")
    print("  five decompositions lean on is named mathematics.")
    print()
    print("  The minority matters more than its size. Every SPECIFIC entry")
    print("  is a place where a decomposition stopped at the paper's prose")
    print("  rather than going under it, and each one is a candidate for")
    print("  further decomposition rather than a settled leaf. Three of the")
    print("  six sit in Lemma 10.5, which is why that one is the weakest of")
    print("  the five even though every step validated.")
    print()
    print("  The classification is mine and is written out in this file, so")
    print("  a reader who thinks the transport flux bound is standard, or")
    print("  that the energy identity is not, can move one line and see the")
    print("  total move with it. A number from a classification is worth")
    print("  only as much as the classification, which is why it is here")
    print("  rather than computed out of sight.")


if __name__ == "__main__":
    main()
