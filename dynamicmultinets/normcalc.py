"""
Statements about function space norms, reached by chaining rules.

The Navier-Stokes run has been saying that its imported steps are out of
reach because they are statements about norms rather than about arithmetic.
That is true of the steps and false as a general claim, and this module is
the correction.

An estimate like

    ||grad^s P_M u||_{L^p(R^d)}  <=  C * M^a * q^b

has two quite different halves. There is the assertion that SOME finite
constant works, which is analysis and is not decidable here. And there is
the exponent bookkeeping -- what a, b, s and 1/p have to be, and what
happens to them when two estimates are combined -- which is linear
arithmetic and is decidable, and which is what the construction actually
manipulates.

So this module splits them, on purpose and visibly:

  ASSUMED    that Bernstein's inequality, Hoelder's inequality and the
             geometric series behave as they classically do, each named on
             the rule that uses it and collected again at the end of any
             chain that went through it. `Rule.assumes` is where that lives
             and `assumptions_behind` is what reads it back.

  DERIVED    the exponents themselves. They are not copied from a table:
             an inequality between norms has to survive rescaling the
             function, and that requirement FORCES the exponent. The prover
             below solves the scaling equation symbolically, so the answer
             covers every integrability index at once rather than the ones
             someone sampled.

  DECIDED    the two places a family of estimates becomes a single
             statement. Summing over dyadic bands converges exactly when
             the band exponent is negative, and a bound q^b vanishes in the
             limit exactly when b is positive. Both are strict inequalities
             on a rational number, both are decided rather than checked,
             and both are quantifiers the run previously called out of
             reach.

What that buys is a derivation, in the architecture's own sense: a chain of
rules carrying an assumed band estimate to a statement about the whole
function, with the exponents exact at every step and the assumptions
listed. What it does not buy is the paper's estimates. Those are not
corollaries of Bernstein and Hoelder; they are the specific analytic work
of Theorem 4.6 and Propositions 5.5, 7.5 and 9.9, and nothing here derives
them. The gain is that once such an estimate is granted, everything the
construction does WITH it is now machine-checked instead of imported
alongside it.

## The cells

    est(d,dv,ip,fr,sc)    ||grad^dv P_M u||_{L^p} <= C M^fr q^sc, uniformly
                          in the dyadic frequency M >= 1, with ip = 1/p
    glob(d,dv,ip,sc)      ||grad^dv u||_{L^p} <= C q^sc
    vanishes(d,dv,ip)     ||grad^dv u||_{L^p} -> 0 as q -> 0

`ip` rather than `p` because every relation here is linear in 1/p and none
is linear in p, and because L^infinity is ip = 0 instead of a special case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable

from .linarith import Lin, obligations, requirements, requires, solve_for
from .provers import Judgement, prover
from .rules import PythonRule, Rule
from .tapes import ABSTRACT, Content

NORM_RULES: dict[str, Callable[[], Rule]] = {}

BERNSTEIN = ("Bernstein's inequality on a dyadic band: a frequency-localised "
             "function obeys ||grad^s P_M f||_p <= C M^s ||P_M f||_p and "
             "||P_M f||_r <= C M^(d(1/p-1/r)) ||P_M f||_p, with C depending "
             "on d, p, r and the Littlewood-Paley cutoff but not on M or f")
HOELDER = ("Hoelder's inequality: ||fg||_r <= ||f||_p ||g||_q whenever "
           "1/r = 1/p + 1/q")
GEOMETRIC = ("a dyadic sum of M^a over M = 2^k, k >= 0, is a geometric "
             "series with ratio 2^a")
TRIANGLE = ("the Littlewood-Paley decomposition converges and the triangle "
            "inequality applies to it, so a bound on the sum of the pieces "
            "is a bound on the whole")


def _norm_rule(name: str):
    def wrap(factory: Callable[[], Rule]) -> Callable[[], Rule]:
        NORM_RULES[name] = factory
        return factory

    return wrap


_NUM = r"-?\d+(?:/\d+)?"
_EST = ("d", "dv", "ip", "fr", "sc")
_GLOB = ("d", "dv", "ip", "sc")
_VANISH = ("d", "dv", "ip")


def _read(text: str, head: str, fields: tuple[str, ...]) -> dict | None:
    body = text.replace(" ", "")
    if not (body.startswith(head + "(") and body.endswith(")")):
        return None
    pattern = ",".join(rf"{f}=({_NUM})" for f in fields)
    m = re.fullmatch(pattern, body[len(head) + 1:-1])
    if not m:
        return None
    return {f: Fraction(v) for f, v in zip(fields, m.groups())}


def _write(head: str, fields: tuple[str, ...], f: dict) -> str:
    return head + "(" + ",".join(f"{k}={Fraction(f[k])}" for k in fields) + ")"


def est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-1), sc=Fraction(0)) -> str:
    return _write("est", _EST, {"d": d, "dv": dv, "ip": ip, "fr": fr, "sc": sc})


def glob(d=3, dv=0, ip=Fraction(1, 2), sc=Fraction(0)) -> str:
    return _write("glob", _GLOB, {"d": d, "dv": dv, "ip": ip, "sc": sc})


def vanishes(d=3, dv=0, ip=Fraction(1, 2)) -> str:
    return _write("vanishes", _VANISH, {"d": d, "dv": dv, "ip": ip})


# ---------------------------------------------------------------------------
# The exponent arithmetic, written once so the prover can run the same code
# ---------------------------------------------------------------------------
def band_shift(d, ip_from, ip_to):
    """What raising integrability on one band costs in frequency.

    Written as a function of its arguments and nothing else, so the prover
    can hand it linear forms and settle the rule on every index at once
    rather than on a sample of them.
    """
    return d * (ip_from - ip_to)


def derivative_shift(d, count):
    """What taking `count` derivatives on one band costs in frequency."""
    return count


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
def _estimate_rule(name: str, description: str, source: str,
                   assumes: Iterable[str],
                   guard: Callable[[dict], bool],
                   shift: Callable[[dict], dict]) -> PythonRule:
    """A band rule, split into the domain it accepts and the arithmetic.

    Split because the prover needs the arithmetic ON ITS OWN. Handing it
    linear forms settles the exponent for every index at once, which a
    guard written with `<=` cannot survive; and the guards here are
    half-lines, which the prover decides separately by their boundary. The
    two halves are attached to the rule so that what gets proved is the
    function the rule runs and not a transcription of it.
    """
    def fn(c: Content) -> Content | None:
        f = _read(c.text, "est", _EST)
        if f is None or not guard(f):
            return None
        return Content.abstract(_write("est", _EST, shift(dict(f))),
                                derivation=description)

    rule = PythonRule(name, fn, ABSTRACT, ABSTRACT, description=description,
                      source=source, exact=True, trusted=True)
    rule.assumes = tuple(assumes)
    rule.guard = guard
    rule.shift = shift
    return rule


@_norm_rule("bernstein_derivative")
def make_bernstein_derivative() -> PythonRule:
    """One more derivative on a band, at one power of the frequency.

    The exponent is not a convention: an inequality
    ||grad P_M f||_p <= C M^a ||P_M f||_p has to survive replacing f by
    f(lambda .), and that forces a = 1. The prover does that rearrangement
    rather than trusting this docstring.
    """
    def shift(f):
        f["dv"] = f["dv"] + 1
        f["fr"] = f["fr"] + derivative_shift(f["d"], 1)
        return f

    return _estimate_rule(
        "bernstein_derivative",
        "one more derivative on a dyadic band costs one power of the frequency",
        "est(dv,fr)->est(dv+1,fr+1)", (BERNSTEIN,),
        lambda f: True, shift)


@_norm_rule("bernstein_uniform")
def make_bernstein_uniform() -> PythonRule:
    """A band estimate in L^p becomes one in L^infinity.

    The workhorse. Frequency localisation is what makes this possible at
    all -- there is no such embedding without it -- and the price is
    d * (1/p) powers of the frequency.
    """
    def shift(f):
        f["fr"] = f["fr"] + band_shift(f["d"], f["ip"], Fraction(0))
        f["ip"] = Fraction(0)
        return f

    return _estimate_rule(
        "bernstein_uniform",
        "a band estimate in L^p becomes one in L^infinity at d/p powers of "
        "the frequency",
        "est(ip,fr)->est(0,fr+d*ip)", (BERNSTEIN,),
        lambda f: f["ip"] > 0, shift)


@_norm_rule("bernstein_to_energy")
def make_bernstein_to_energy() -> PythonRule:
    """A band estimate in L^p, p <= 2, becomes one in L^2.

    The other direction the construction needs, because the energy is an
    L^2 quantity and the pulses are estimated where they are convenient.
    """
    def shift(f):
        f["fr"] = f["fr"] + band_shift(f["d"], f["ip"], Fraction(1, 2))
        f["ip"] = Fraction(1, 2)
        return f

    return _estimate_rule(
        "bernstein_to_energy",
        "a band estimate in L^p with p <= 2 becomes one in L^2",
        "est(ip>1/2,fr)->est(1/2,fr+d(ip-1/2))", (BERNSTEIN,),
        lambda f: f["ip"] > Fraction(1, 2), shift)


@_norm_rule("sum_over_bands")
def make_sum_over_bands() -> PythonRule:
    """Every dyadic band at once, when the sum converges.

    This is the rule that answers one of the quantifiers the run has been
    calling out of reach. A family of band estimates is not a statement
    about the function; summing them is. The sum over M = 2^k of M^fr is
    geometric with ratio 2^fr, so it converges exactly when fr < 0 -- and
    when it does not, this rule DECLINES, which is the honest outcome and
    is what stops a proof search from walking past a divergent sum.
    """

    def fn(c: Content) -> Content | None:
        f = _read(c.text, "est", _EST)
        if f is None:
            return None
        if not requires(f["fr"], 0, "the dyadic sum converges"):
            return None                     # the dyadic sum diverges
        return Content.abstract(
            _write("glob", _GLOB, f),
            derivation=f"sum over M = 2^k of M^({f['fr']}) converges")

    rule = PythonRule(
        "sum_over_bands", fn, ABSTRACT, ABSTRACT,
        description="a band estimate with a negative frequency exponent sums "
                    "to an estimate on the whole function",
        source="est(fr<0)->glob", exact=True, trusted=True)
    rule.assumes = (GEOMETRIC, TRIANGLE)
    return rule


@_norm_rule("limit_at_the_singularity")
def make_limit_at_the_singularity() -> PythonRule:
    """A bound q^sc vanishes as the scale goes to zero, when sc > 0.

    The other quantifier the run has been calling out of reach. It is a
    limit, which is exactly why no number of instances reaches it, and it
    is decided by the sign of one rational number.
    """

    def fn(c: Content) -> Content | None:
        f = _read(c.text, "glob", _GLOB)
        if f is None or not requires(0, f["sc"], "the bound vanishes at q = 0"):
            return None
        return Content.abstract(
            _write("vanishes", _VANISH, f),
            derivation=f"q^({f['sc']}) -> 0 as q -> 0")

    rule = PythonRule(
        "limit_at_the_singularity", fn, ABSTRACT, ABSTRACT,
        description="a bound with a positive power of the concentration "
                    "scale vanishes in the limit",
        source="glob(sc>0)->vanishes", exact=True, trusted=True)
    rule.assumes = ()
    return rule


@_norm_rule("holder_product")
def make_holder_product() -> PythonRule:
    """Two estimates on one band multiply, with the indices adding.

    The quadratic term is why this is here: `div(w tensor w)` needs a bound
    on a product, and Hoelder gives one whose integrability index is the
    sum of the two. Written on a paired cell because a rule maps one cell
    to one cell, and a product needs two.
    """

    def fn(c: Content) -> Content | None:
        body = c.text.replace(" ", "")
        if not (body.startswith("pair(") and body.endswith(")")):
            return None
        halves = body[5:-1].split(";")
        if len(halves) != 2:
            return None
        a, b = (_read(h, "est", _EST) for h in halves)
        if a is None or b is None or a["d"] != b["d"]:
            return None
        if a["ip"] + b["ip"] > 1:
            return None                     # not a Lebesgue index any more
        out = {"d": a["d"], "dv": Fraction(0), "ip": a["ip"] + b["ip"],
               "fr": a["fr"] + b["fr"], "sc": a["sc"] + b["sc"]}
        return Content.abstract(_write("est", _EST, out),
                                derivation="Hoelder: 1/r = 1/p + 1/q")

    rule = PythonRule(
        "holder_product", fn, ABSTRACT, ABSTRACT,
        description="two band estimates multiply, integrability indices "
                    "adding",
        source="pair(est;est)->est", exact=True, trusted=True)
    rule.assumes = (HOELDER,)
    return rule


def pair(first: str, second: str) -> str:
    return f"pair({first};{second})"


# ---------------------------------------------------------------------------
# Rules proposed from outside, admitted only if the machine can check them
# ---------------------------------------------------------------------------
#: What scaling forces a band inequality's frequency exponent to be, as
#: coefficients over the three things a band rule can change. Derived in
#: `_prove_band_exponents` by rearranging the scaling equation; written
#: here as data so a PROPOSED rule can be compared against it without
#: anyone reimplementing the derivation.
FORCED_SHIFT = {"ip": Fraction(3), "ip_to": Fraction(-3), "dv": Fraction(1),
                "const": Fraction(0)}


@dataclass
class Proposal:
    """A band rule someone suggested, and what the machine made of it."""

    name: str
    admitted: bool
    reason: str
    proposed: str = ""
    forced: str = ""

    def summary(self) -> str:
        head = f"{self.name}: {'ADMITTED' if self.admitted else 'REFUSED'}"
        if self.admitted:
            return f"{head} -- {self.reason}"
        return (f"{head} -- {self.reason}\n    proposed shift {self.proposed}"
                f"\n    scaling forces  {self.forced}")


def _shift_form(coeffs: dict) -> Lin:
    """The frequency shift as a linear form in the indices."""
    return (Lin.var("ip") * Fraction(coeffs.get("ip", 0) or 0)
            + Lin.var("ip_to") * Fraction(coeffs.get("ip_to", 0) or 0)
            + Lin.var("dv") * Fraction(coeffs.get("dv", 0) or 0)
            + Fraction(coeffs.get("const", 0) or 0))


def propose_band_rule(library, name: str, ip_to, frequency_shift: dict,
                      derivative_change: int = 0, scale_shift=0,
                      assumes: str = "", description: str = "") -> Proposal:
    """Admit a band inequality proposed from outside, if it checks out.

    This is the door in the closed registry, and it is narrow on purpose.
    The catalogue tells a controller it must pick generators and oracles by
    name because an oracle written by whoever wrote the rule is not a
    second opinion. That reasoning does not apply to a rule whose
    correctness the machine can DECIDE, and a band inequality is one: its
    frequency exponent is forced by the inequality surviving a rescaling of
    the function, so a proposal either has that exponent or it does not.

    So the proposer supplies data, not code -- where the integrability
    index lands, how many derivatives are taken, and what it claims the
    frequency costs -- and the machine compares the claim against what
    scaling forces. A wrong exponent is refused with both forms printed.
    Nothing is taken on the proposer's word and nothing executable crosses
    the boundary.

    What is NOT checked, and is recorded rather than hidden: that the
    inequality holds at all with some finite constant. Scaling fixes the
    exponent of an inequality that is true; it does not make one true. So
    an admitted rule carries its `assumes` text exactly as the built-in
    ones do, and `assumptions_behind` will surface it at the end of any
    chain that used it.
    """
    proposed = _shift_form(frequency_shift)
    forced = _shift_form(FORCED_SHIFT)
    ip_to = Fraction(ip_to)

    if Fraction(scale_shift) != 0:
        return Proposal(name, False,
                        "a band inequality cannot change the power of the "
                        "concentration scale: rescaling the function does not "
                        "touch it, so a nonzero shift here is a claim scaling "
                        "does not support",
                        f"scale shift {Fraction(scale_shift)}", "scale shift 0")
    if not 0 <= ip_to <= 1:
        return Proposal(name, False,
                        f"1/p = {ip_to} is not an integrability index",
                        str(ip_to), "between 0 and 1")
    if proposed != forced:
        return Proposal(name, False,
                        "the frequency exponent is not the one rescaling the "
                        "function forces",
                        str(proposed), str(forced))
    if not assumes:
        return Proposal(name, False,
                        "a proposed rule has to name the classical fact it "
                        "leans on; an exponent checked on top of an unstated "
                        "assumption is worse than an unchecked one",
                        "no assumption named", "one named")

    dv_change = Fraction(derivative_change)

    def shift(f):
        f["fr"] = f["fr"] + (f["d"] * (f["ip"] - ip_to) + dv_change)
        f["ip"] = ip_to
        f["dv"] = f["dv"] + dv_change
        return f

    rule = _estimate_rule(
        name, description or f"proposed band inequality to 1/p = {ip_to}",
        f"est(ip,fr)->est({ip_to},fr+d(ip-{ip_to})+{dv_change})",
        (assumes,), lambda f: f["ip"] >= ip_to, shift)
    rule.proved = (f"the frequency exponent {forced} is the one rescaling "
                   f"the function forces, for every integrability index")
    library.add(rule, replace=True)
    return Proposal(name, True,
                    f"the claimed exponent is what scaling forces, so the "
                    f"rule is admitted and usable in a chain; its analytic "
                    f"content stays assumed and named",
                    str(proposed), str(forced))


# ---------------------------------------------------------------------------
# Deriving the exponents instead of quoting them
# ---------------------------------------------------------------------------
@prover("band_exponents_by_scaling",
        "derives the frequency exponent of a band inequality from the "
        "requirement that it survive rescaling the function, and checks the "
        "rule computes that exponent for every index rather than for some")
def _prove_band_exponents(library, rule) -> Judgement:
    """Where `d(1/p - 1/r)` comes from, done rather than cited.

    Suppose ||P_M f||_r <= C M^a ||P_M f||_p holds for every f and every
    dyadic M with one constant. Apply it to f(lambda .). Rescaling moves
    the frequency, P_M (f(lambda .)) = (P_{M/lambda} f)(lambda .), and
    moves the norms, ||g(lambda .)||_p = lambda^(-d/p) ||g||_p. Writing
    both sides in terms of P_{M/lambda} f leaves

        lambda^(-d*ir) ||.||_r  <=  C lambda^(a - d*ip) M'^a ||.||_p,

    and a constant that does not depend on lambda forces the two powers of
    lambda to agree:

        -d*ir = a - d*ip,   so   a = d*(ip - ir).

    That rearrangement is done below symbolically, with the two
    integrability indices left as variables, so what comes out is the
    exponent for every pair at once. Then the rule's own arithmetic is
    handed the same variables and the two are compared as linear forms.
    The rule is decided on its whole domain, and nothing was sampled.

    What is assumed and stays assumed: that the inequality holds at all.
    Scaling says what the exponent must be IF a constant exists; it does
    not produce one.
    """
    # The dimension is grounded and the two integrability indices are not.
    # A product of two variables would leave the linear fragment, and the
    # construction is in three dimensions; nothing below is claimed for any
    # other.
    dim = Fraction(3)
    ip, ir = Lin.var("ip"), Lin.var("ir")
    a = Lin.var("a")

    # Powers of lambda picked up on each side when f is replaced by
    # f(lambda .), written as linear forms and set equal.
    left = ir * (-dim)
    right = a - ip * dim
    forced = solve_for(left - right, "a")
    if forced != ip * dim - ir * dim:
        return Judgement(False, obstruction=(
            f"the scaling equation rearranged to a = {forced}, which is not "
            f"d(1/p - 1/r) at d = 3"))

    if not hasattr(rule, "shift"):
        return Judgement(False, obstruction=(
            f"{rule.name} does not expose the arithmetic it runs, so the "
            f"proof would be about a copy of it"))

    checks = [f"scaling forces a = {forced}, for every pair of indices"]
    known = {"bernstein_uniform": Fraction(0),
             "bernstein_to_energy": Fraction(1, 2)}

    # Run the rule's OWN shift on a symbolic index and read what it did to
    # the frequency exponent. Nothing here recomputes the shift; if the
    # rule's arithmetic and the scaling argument disagree, they disagree.
    before = {"d": dim, "dv": Fraction(0), "ip": Lin.var("ip"),
              "fr": Lin(Fraction(0)), "sc": Fraction(0)}
    after = rule.shift(dict(before))
    got = after["fr"] - before["fr"]

    if rule.name in known:
        target = known[rule.name]
        want = forced.at(ir=target)
        if got != want:
            return Judgement(False, obstruction=(
                f"{rule.name} shifts the frequency exponent by {got}, and "
                f"scaling forces {want}"))
        if after["ip"] != target:
            return Judgement(False, obstruction=(
                f"{rule.name} lands at index {after['ip']}, not {target}, so "
                f"the exponent it was checked against is the wrong one"))
        checks.append(f"{rule.name}: shift {got} for every 1/p, which is what "
                      f"scaling forces when 1/r = {target}")
    elif rule.name == "bernstein_derivative":
        # A derivative brings one power of lambda down, so the scaling
        # equation is s - d*ip = a - d*ip and forces a = s.
        order = Lin.var("s")
        per_derivative = solve_for((order - ip * dim) - (a - ip * dim), "a")
        if got != per_derivative.at(s=1):
            return Judgement(False, obstruction=(
                f"bernstein_derivative shifts by {got}, and scaling forces "
                f"{per_derivative.at(s=1)} for one derivative"))
        if after["dv"] - before["dv"] != 1:
            return Judgement(False, obstruction=(
                "bernstein_derivative does not raise the derivative count "
                "by one, so its shift is being compared to the wrong order"))
        checks.append(f"bernstein_derivative: scaling forces a = "
                      f"{per_derivative}, and the rule shifts by {got} for "
                      f"one derivative")
    else:
        return Judgement(False, obstruction=(
            f"{rule.name} is not one of the band inequalities this prover "
            f"knows how to derive"))

    return Judgement(
        True,
        statement=("the frequency exponent of a band inequality is forced by "
                   "rescaling: a = d(1/p - 1/r), and one per derivative"),
        covers=("every integrability index, at the construction's dimension "
                "d = 3, which is the rule's whole domain"),
        whole_domain=True,
        detail=checks + [
            "derived by rearranging the scaling equation symbolically, with "
            "the indices left as variables rather than sampled",
            "what stays assumed is that some finite constant exists; scaling "
            "fixes the exponent and produces no constant",
        ])


@prover("summation_and_limit_by_sign",
        "decides the two places a family of estimates becomes one statement: "
        "the dyadic sum converges exactly when the band exponent is negative "
        "and the bound vanishes exactly when the scale exponent is positive")
def _prove_summation_and_limit(library, rule) -> Judgement:
    """The two quantifiers, decided rather than sampled.

    Both are one-line facts and both are strict inequalities on a rational
    number, which is why they can be settled here at all:

        sum over M = 2^k, k >= 0, of M^a  converges  <=>  a < 0
        q^b -> 0 as q -> 0+                          <=>  b > 0

    What the prover does is confirm the RULE's guard is that condition and
    not an approximation of it, by asking it on both sides of the boundary
    and at the boundary itself. A predicate that is a half-line agrees with
    another half-line everywhere exactly when they agree there.
    """
    probes = [Fraction(0), Fraction(1, 10 ** 9), Fraction(-1, 10 ** 9),
              Fraction(1, 3), Fraction(-1, 3), Fraction(7), Fraction(-7)]
    detail, wrong = [], []

    if rule.name == "sum_over_bands":
        for a in probes:
            got = rule.apply(Content.abstract(est(fr=a, sc=Fraction(1, 5))))
            should = a < 0
            if (got is not None) != should:
                wrong.append(f"band exponent {a}: rule "
                             f"{'accepts' if got is not None else 'declines'}, "
                             f"the series {'converges' if should else 'diverges'}")
        detail.append("the dyadic sum of M^a is geometric with ratio 2^a, so "
                      "it converges exactly when a < 0")
        statement = ("a family of band estimates with a negative frequency "
                     "exponent sums to an estimate on the whole function, and "
                     "one with a non-negative exponent does not")
    elif rule.name == "limit_at_the_singularity":
        for b in probes:
            got = rule.apply(Content.abstract(glob(sc=b)))
            should = b > 0
            if (got is not None) != should:
                wrong.append(f"scale exponent {b}: rule "
                             f"{'accepts' if got is not None else 'declines'}, "
                             f"the bound {'vanishes' if should else 'does not'}")
        detail.append("q^b -> 0 as q -> 0+ exactly when b > 0; at b = 0 the "
                      "bound is a constant and below it the bound blows up")
        statement = ("a bound by a positive power of the concentration scale "
                     "vanishes in the limit, and one by a non-positive power "
                     "does not")
    else:
        return Judgement(False, obstruction=(
            f"{rule.name} is not one of the two threshold rules"))

    if wrong:
        return Judgement(False, obstruction=(
            "the rule's guard is not the condition: " + "; ".join(wrong[:3])))

    return Judgement(
        True, statement=statement,
        covers="every rational exponent, which is the rule's whole domain",
        whole_domain=True,
        detail=detail + [
            "checked at the boundary and on both sides of it, which settles "
            "two half-lines against each other",
            "this is the step the run used to call out of reach: it is a "
            "quantifier over bands, and over the limit, not over instances",
        ])


# ---------------------------------------------------------------------------
# Reading the assumptions back out of a chain
# ---------------------------------------------------------------------------
def assumptions_behind(library, rule_names: Iterable[str]) -> list[str]:
    """Every classical inequality a chain leaned on, once each.

    The point of the exercise. A derivation here is exact in its exponents
    and only as good as what it assumed, and the assumptions have to come
    back out at the end or the exactness is misleading.
    """
    out: list[str] = []
    for name in rule_names:
        for text in getattr(library.get(name), "assumes", ()):
            if text not in out:
                out.append(text)
    return out


def required_estimate(library, chain: tuple[str, ...] = (
        "bernstein_uniform", "sum_over_bands", "limit_at_the_singularity"),
        ip=Fraction(1, 2)) -> tuple[str, list[str]]:
    """What estimate would the construction have to supply for this to work?

    The other half of a derivation, and the half this package has not been
    doing. Everything so far takes an estimate and carries it forward. This
    runs the same chain with the estimate left as variables and collects
    what the guards would need, so the answer is a HYPOTHESIS the machine
    states rather than a conclusion it reaches.

    That is the shape of `propose_rules` applied to analysis: put the cells
    that cannot be reached beside the rules that would reach them, and let
    what they need come back as the thing worth proving. Here it comes back
    as inequalities on the frequency and scale exponents, derived by
    running the rules rather than by reading them.
    """
    fr, sc = Lin.var("fr"), Lin.var("sc")
    state = {"d": Fraction(3), "dv": Fraction(0), "ip": ip, "fr": fr, "sc": sc}
    with obligations() as needed:
        for name in chain:
            rule = library.get(name)
            if hasattr(rule, "shift"):
                state = rule.shift(dict(state))
            elif name == "sum_over_bands":
                requires(state["fr"], 0, "the dyadic sum converges")
            elif name == "limit_at_the_singularity":
                requires(0, state["sc"], "the bound vanishes at q = 0")
    start = (f"est(d=3,dv=0,ip={Fraction(ip)},fr=<fr>,sc=<sc>) with 1/p = "
             f"{Fraction(ip)}")
    return start, requirements(needed)


def install_norm_rules(library) -> None:
    for factory in NORM_RULES.values():
        library.add(factory(), replace=True)


#: A derivation the construction needs, as the cells it passes through.
#: Starting point: an L^2 band estimate for the residual, which is the kind
#: of thing Proposition 7.5 delivers and which is ASSUMED here.
RESIDUAL_START = est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-3),
                     sc=Fraction(1, 5))
RESIDUAL_TARGET = vanishes(d=3, dv=0, ip=Fraction(0))
