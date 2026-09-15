"""
Experiment 1s: discovery in a norm space.

The package kept saying that discovery works on instances and therefore
cannot reach a statement about every case, and that norm estimates are
consequently out of its reach. The first half is a fact about a CHOICE OF
INSTANCE SPACE and was being stated as a fact about discovery. Rule
formation by observation applies wherever the concepts can be written
down, and a function space is somewhere concepts can be written down.

`normcalc.py` already writes them down:

    est(d,dv,ip,fr,sc)   ||grad^dv P_M u||_{L^p} <= C M^fr q^sc, uniformly
                         in the dyadic frequency M, with ip = 1/p

That cell is not a number. It is a uniform statement about every function
in a band, and a rule mapping one such cell to another is a theorem about
function spaces. So the instances a discovery run samples here are whole
estimates, and each check is a check of a uniform claim rather than of one
numerical case.

WHAT MAKES IT A CHECK RATHER THAN A RESTATEMENT. The rule does exponent
arithmetic on a written-down estimate. The oracle does not: it builds the
bump concentrated at width 1/M that saturates the inequality, integrates
its norms on a grid at four frequencies, and fits the exponent as a slope
in log M. Numerical quadrature on an actual function and symbolic
bookkeeping about a norm are not the same route to the answer, which is
the whole requirement verification has.

WHAT THIS DOES NOT DO. It does not make Bernstein's inequality true. The
oracle measures the EXPONENT of an inequality already believed to hold,
which is exactly the split `normcalc` draws: that some finite constant
works is analysis and is assumed by name; what the exponent has to be is
forced, and is now measurable as well as provable.

Run: python examples/run_norm_discovery.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import RenMachine                             # noqa: E402
from dynamicmultinets.normcalc import (BERNSTEIN, _estimate_rule,   # noqa: E402
                                       install_norm_rules)

#: The two band rules and the integrability index each moves to.
UNDER_TEST = [("bernstein_uniform", "0"), ("bernstein_to_energy", "1/2")]


def mutant():
    """The same rule with the dimension factor dropped.

    A check nothing can fail is not a check, so the run ends by breaking
    the rule in a way that is wrong and looks right. Bernstein costs
    d * (1/p) powers of the frequency; this one charges 1/p. In one
    dimension the two agree exactly, which is what makes it the honest
    mutant to try: a reader who tested only on the line would see nothing.
    """
    def shift(f):
        f["fr"] = f["fr"] + (f["ip"] - Fraction(0))
        f["ip"] = Fraction(0)
        return f

    return _estimate_rule(
        "bernstein_no_dimension",
        "MUTANT: charges 1/p powers of the frequency instead of d * (1/p)",
        "est(ip,fr)->est(0,fr+ip)", (BERNSTEIN,),
        lambda f: f["ip"] > 0, shift)


def main() -> None:
    print("=" * 78)
    print("Discovery in a norm space: the instances are estimates")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    machine = RenMachine(device="cpu")
    install_norm_rules(machine.library)

    print("\n--- what one instance looks like ---")
    from dynamicmultinets.generators import generate
    from dynamicmultinets.oracles import label
    sample = label(generate("norm_band_estimates", 3, seed=5, ip_to="0"),
                   "band_exponent_by_quadrature")
    for ex in sample.examples:
        print(f"  {ex.inp.text}")
        print(f"     measured -> {ex.out.text if ex.out else '(unlabelled)'}")
    print("  Each line is a uniform statement in, a uniform statement out.")
    print("  Nothing here is a number the rule was asked about.")

    print("\n--- the checks ---")
    print(f"  {'rule':24}{'n':>5}{'distinct':>10}{'accuracy':>10}  trusted")
    for name, target in UNDER_TEST:
        machine.generate_data("norm_band_estimates", 300, seed=11,
                              name=f"n_{name}", ip_to=target)
        report = machine.verify(name, f"n_{name}",
                                "band_exponent_by_quadrature", threshold=0.99)
        print(f"  {name:24}{report.n_checked:>5}{report.n_distinct:>10}"
              f"{report.accuracy:>10.4f}  "
              f"{'yes' if machine.library.get(name).trusted else 'no'}")
    print("  Charged per distinct estimate, not per draw, as everywhere else")
    print("  in this package.")

    print("\n--- the check can fail ---")
    machine.library.add(mutant())
    machine.generate_data("norm_band_estimates", 300, seed=11, name="mut",
                          ip_to="0")
    bad = machine.verify("bernstein_no_dimension", "mut",
                         "band_exponent_by_quadrature", threshold=0.99)
    print(f"  bernstein_no_dimension   accuracy {bad.accuracy:.4f} over "
          f"{bad.n_distinct} distinct estimates")
    if bad.counterexamples:
        print(f"  first disagreement: {bad.counterexamples[0]}")
    print("  It agrees on the line and fails in the plane, which is where")
    print("  the factor it dropped lives.")

    print("\n--- reading it ---")
    print("  So discovery is not confined to arithmetic on instances. What")
    print("  it needs is a space whose elements can be written down and a")
    print("  second route to the answer, and a norm space has both. The")
    print("  earlier claim that a uniform statement is beyond reach was")
    print("  about sampling the underlying FUNCTIONS. Sampling the")
    print("  ESTIMATES is a different thing and it is what this does.")
    print()
    print("  The estimates the Navier-Stokes construction needs are still")
    print("  not established by this. They are not corollaries of")
    print("  Bernstein, and measuring an exponent does not supply a")
    print("  constant. What changes is the scope of the method, not the")
    print("  standing of those estimates.")


if __name__ == "__main__":
    main()
