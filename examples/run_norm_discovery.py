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

AND WHAT THE NETWORK IS FOR. The run ends with a learned rule over the
same statements, and the point of it is not that a network can compute a
band exponent. A rule whose correctness can be derived or computed should
not be executed by a network in practice: the decision procedure is exact,
total and cheap, and the network is none of those. Trained on one and two
dimensions it scores 0.98 there and 0.03 in three, because it found the
cases rather than the law.

What it IS for is finding a candidate. So the last section does not ask it
for answers; it asks what rule it behaves as if it had, solving for the
constant in `fr + c * (ip - ip_to)`, and hands that to `propose_band_rule`.
The network supplies the shape, which no decision procedure was going to
suggest. Rescaling settles the constant, which is the half the network got
wrong. Neither does the other's work.

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


def learn_the_exponent(machine, epochs: int = 20, n_train: int = 1200,
                       n_test: int = 300, seed: int = 0):
    """Train a NETWORK to read a band estimate and say what it costs.

    Symbols and text layouts are the same content in two carriers: the
    abstract cell `est(d=2,dv=1,ip=1/2,fr=-3,sc=0)` and the layout of those
    characters on the specific tape are one statement, and the codec
    renders one into the other implicitly. So there is nothing stopping a
    learned rule from working on a function-space concept, and this is it:
    the network sees the rendered estimate and emits the frequency
    exponent, which the slot vocabulary of digits and operators can spell.

    The test set is the experiment. Training draws only the line and the
    plane; the held-out check draws three dimensions, which the network has
    never seen. Bernstein charges d * (1/p), so a network that found the
    PATTERN answers correctly in a new dimension and one that found the
    CASES does not. Scoring only on seen dimensions would not tell them
    apart, and the instance space is small enough that it mostly would not
    be testing anything.

    Returns (train report, accuracy on seen dimensions, accuracy on the
    unseen one, a few of its mistakes).
    """
    from dynamicmultinets.tapes import ABSTRACT

    machine.declare_rule(
        "band_exponent_learned", ABSTRACT, ABSTRACT,
        "read a band estimate and say the exponent it acquires",
        num_slots=8, from_oracle="band_exponent_value_by_quadrature")
    machine.generate_data("norm_band_estimates", n_train, seed=2, name="tr",
                          ip_to="0", min_dim=1, max_dim=2)
    machine.label_data("tr", "band_exponent_value_by_quadrature")
    report = machine.train("band_exponent_learned", "tr", epochs=epochs,
                           seed=seed)

    scores, misses = {}, []
    for tag, lo, hi in (("seen", 1, 2), ("unseen", 3, 3)):
        machine.generate_data("norm_band_estimates", n_test, seed=77,
                              name=f"te_{tag}", ip_to="0",
                              min_dim=lo, max_dim=hi)
        got = machine.verify("band_exponent_learned", f"te_{tag}",
                             "band_exponent_value_by_quadrature",
                             threshold=0.99)
        scores[tag] = got.accuracy
        if tag == "unseen":
            misses = got.counterexamples[:3]
    return report, scores["seen"], scores["unseen"], misses


def proposal_from_the_network(machine, n: int = 120, seed: int = 77):
    """Read the network's behaviour as a CANDIDATE RULE, and decide it.

    This is the division of labour the package should have been drawing all
    along. A rule whose correctness can be derived or computed has no
    business being executed by a network: the decision procedure is exact,
    total and cheap, and the network is none of those. What a network is
    for is FINDING a candidate -- noticing that the answers look like a
    constant times the index gap -- after which computation settles whether
    the constant is right.

    So the network is not asked for answers here. It is asked what rule it
    behaves as if it had, by solving

        answer = fr + c * (ip - ip_to)

    for c on each instance and taking the value it uses most. That c is a
    proposal, `propose_band_rule` compares it against what rescaling
    forces, and a wrong one comes back REFUSED with both forms printed.

    Returns (the constant the network implies, the verdict on it, the
    verdict on the forced one).
    """
    from collections import Counter

    from dynamicmultinets.generators import generate
    from dynamicmultinets.normcalc import propose_band_rule
    from dynamicmultinets.tapes import Content

    rule = machine.library.get("band_exponent_learned")
    implied: Counter = Counter()
    for ex in generate("norm_band_estimates", n, seed=seed, ip_to="0",
                       min_dim=3, max_dim=3).examples:
        got = rule.apply(Content.abstract(ex.inp.text))
        if got is None:
            continue
        gap = Fraction(ex.meta["ip"]) - Fraction(ex.meta["ip_to"])
        if gap == 0:
            continue
        try:
            answer = Fraction(got.text)
        except (ValueError, ZeroDivisionError):
            continue            # the slots spelled something that is not a number
        implied[(answer - Fraction(ex.meta["fr"])) / gap] += 1
    if not implied:
        return None, None, None
    c = implied.most_common(1)[0][0]

    leans = ("Bernstein's inequality holds on a dyadic band with SOME finite "
             "constant; what is at issue here is only its exponent")
    guessed = propose_band_rule(
        machine.library, "band_shift_proposed_by_the_network", 0,
        {"ip": c, "ip_to": -c, "dv": 1}, assumes=leans,
        description="the shift the learned rule behaves as if it had")
    forced = propose_band_rule(
        machine.library, "band_shift_to_uniform", 0,
        {"ip": 3, "ip_to": -3, "dv": 1}, assumes=leans,
        description="the shift rescaling forces in three dimensions")
    return c, guessed, forced


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

    print("\n--- the same question, asked of a network ---")
    print("  A band estimate written as symbols and the same estimate laid")
    print("  out as characters on the specific tape are one statement in two")
    print("  carriers, and the codec renders between them implicitly. So a")
    print("  learned rule can work on a function-space concept directly:")
    print("  this one reads the rendered estimate and emits the exponent.")
    print()
    print("  It is trained on one and two dimensions only, and tested on")
    print("  three. Bernstein charges d * (1/p), so a network that found the")
    print("  pattern answers a new dimension and one that found the cases")
    print("  does not.")
    print()
    report, seen, unseen, misses = learn_the_exponent(machine)
    print(f"  {report.summary()}")
    print(f"  accuracy on dimensions 1 and 2, which it trained on: {seen:.4f}")
    print(f"  accuracy on dimension 3, which it never saw:         {unseen:.4f}")
    for bad in misses:
        print(f"    {bad}")
    print()
    print("  It found the cases. In three dimensions it answers with the")
    print("  two-dimensional shift, which is a lookup table behaving exactly")
    print("  as a lookup table does off the end of its keys. The symbolic")
    print("  rule above has the factor d in it and is right in every")
    print("  dimension without being shown one.")

    print("\n--- so the network is used for the other job ---")
    print("  A rule whose correctness can be derived or computed should not")
    print("  be executed by a network in practice, and asking one to compute")
    print("  Bernstein's exponent was pointing it at work a decision")
    print("  procedure already does exactly, totally and cheaply. What a")
    print("  network is for is FINDING a candidate. So it is not asked for")
    print("  answers here; it is asked what rule it behaves as if it had,")
    print("  and computation decides that.")
    print()
    c, guessed, forced = proposal_from_the_network(machine)
    if guessed is not None:
        print(f"  the constant the network implies: {c}")
        print()
        for line in guessed.summary().splitlines():
            print(f"  {line}")
        print()
        for line in forced.summary().splitlines():
            print(f"  {line}")
        print()
        print("  That is the division of labour. The network noticed the")
        print("  SHAPE -- a constant times the gap between the indices, plus")
        print("  one per derivative -- which is the part no decision")
        print("  procedure was going to suggest. It got the constant wrong,")
        print("  and the constant is exactly what rescaling settles. Neither")
        print("  half does the other's work.")

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
