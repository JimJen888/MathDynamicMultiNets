"""
The construction's order calculus, from Proposition 6.6.

Last turn I stopped short of the front half, on the grounds that its
proofs reason in a private apparatus rather than in public mathematics,
and that registering that apparatus would mean writing my own paraphrase
of notation defined a hundred pages earlier. Reading further, that was
half wrong in a way that matters.

The apparatus is not informal notation. It is Proposition 6.6, a stated
result with a proof, which says exactly how the three coefficient classes
compose and how the differential operators move between them. Registering
it is citing a proposition, the same act as citing Cauchy-Schwarz, and a
reader can check it against page 71 rather than against my prose.

## The three classes

A coefficient sits in one of three classes, each carrying a real order
recording the power of the small parameter it is bounded by:

    M_alpha   angular mean coefficients
    W_alpha   wave amplitudes, carrying an integer harmonic
    S_alpha   radial moments

## What Proposition 6.6 says

Products:

    M_a M_b  in  M_(a+b)
    M_a W_b  in  W_(a+b)
    W_a W_b  in  W_(a+b) when the harmonics do not cancel, and in
                 M_(a+b) when they do, after the temporal cutoff

and each class is closed under finite sums at a fixed order. Operators,
from the table (6.32):

    coefficient derivatives in R, Z, T     order unchanged
    the radial derivative D_r              order drops by kappa_s
    the axial derivative D_z = eps d_Z     order rises by one
    the slow time derivative -eps d_T      order rises by one
    the absorption operator                order unchanged

That is the whole of it, and it is rational arithmetic on one exponent,
which is the fragment this package is built to do exactly. Nothing here
is approximated and nothing is sampled: an order is a `Fraction` and the
rules add and subtract.

## Why this is the prerequisite rather than a detour

Proposition 9.5's proof is a sequence of statements of the form "this
increment is M at that order", chained by these laws. With the laws
registered, that reasoning becomes a derivation the machine can check. It
was not that the front half used unknown mathematics; it was that it used
a lemma this repository had not encoded.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Callable

from .rules import JoinRule, PythonRule, Rule
from .tapes import ABSTRACT, Content

APPARATUS_RULES: dict[str, Callable[[], Rule]] = {}

#: Proposition 6.6 and the operator table (6.32), cited on every rule that
#: uses them so a chain reports what it leaned on.
PROP_6_6 = ("Proposition 6.6: the coefficient classes are closed under "
            "finite sums at a fixed order, and products satisfy "
            "M_a M_b in M_(a+b), M_a W_b in W_(a+b), and W_a W_b in "
            "W_(a+b) or M_(a+b) according to whether the harmonics cancel")
TABLE_6_32 = ("the operator table (6.32): coefficient derivatives preserve "
              "the order, the radial derivative costs kappa_s, and the axial "
              "and slow-time derivatives each gain one")

#: The radial derivative loss. The construction takes it very small; it is
#: a parameter here so the calculus is not tied to one value.
KAPPA_S = Fraction(1, 100_000)

_NUM = r"-?\d+(?:/\d+)?"


def _apparatus(name: str):
    def wrap(factory: Callable[[], Rule]) -> Callable[[], Rule]:
        APPARATUS_RULES[name] = factory
        return factory

    return wrap


def mean(order) -> str:
    return f"M({Fraction(order)})"


def wave(order, harmonic: int) -> str:
    return f"W({Fraction(order)},{harmonic})"


def moment(order) -> str:
    return f"S({Fraction(order)})"


def pair(first: str, second: str) -> str:
    return f"pair({first};{second})"


def read(text: str) -> tuple | None:
    """`M(a)`, `W(a,m)` or `S(a)` as (class, order, harmonic)."""
    body = text.replace(" ", "")
    m = re.fullmatch(rf"([MS])\(({_NUM})\)", body)
    if m:
        return m.group(1), Fraction(m.group(2)), None
    m = re.fullmatch(rf"W\(({_NUM}),(-?\d+)\)", body)
    if m:
        return "W", Fraction(m.group(1)), int(m.group(2))
    return None


def write(kind: str, order: Fraction, harmonic: int | None = None) -> str:
    if kind == "W":
        return wave(order, harmonic if harmonic is not None else 0)
    return f"{kind}({Fraction(order)})"


# ---------------------------------------------------------------------------
# Products (6.30) and (6.31)
# ---------------------------------------------------------------------------
@_apparatus("class_product")
def make_class_product() -> PythonRule:
    """`pair(X;Y)` -> the class and order of the product.

    All three product laws in one rule, because they are one statement in
    the paper and splitting them would invite the reader to check three
    things where Proposition 6.6 asserts one. The harmonic bookkeeping is
    the only part with a case in it: two waves whose harmonics cancel
    produce a mean, and otherwise a wave at the summed harmonic.
    """

    def fn(c: Content) -> Content | None:
        body = c.text.replace(" ", "")
        if not (body.startswith("pair(") and body.endswith(")")):
            return None
        halves = body[5:-1].split(";")
        if len(halves) != 2:
            return None
        left, right = (read(h) for h in halves)
        if left is None or right is None:
            return None
        (lk, la, lm), (rk, ra, rm) = left, right
        order = la + ra
        if lk == "M" and rk == "M":
            return Content.abstract(mean(order), derivation=PROP_6_6)
        if {lk, rk} == {"M", "W"}:
            harmonic = lm if lk == "W" else rm
            return Content.abstract(wave(order, harmonic), derivation=PROP_6_6)
        if lk == "W" and rk == "W":
            total = (lm or 0) + (rm or 0)
            return Content.abstract(
                mean(order) if total == 0 else wave(order, total),
                derivation=PROP_6_6)
        return None                      # moments do not multiply here

    rule = PythonRule("class_product", fn, ABSTRACT, ABSTRACT,
                      description="the product laws of Proposition 6.6",
                      source="pair(X;Y)->X*Y", exact=True, trusted=True)
    rule.assumes = (PROP_6_6,)
    return rule


# ---------------------------------------------------------------------------
# The operator table (6.32)
# ---------------------------------------------------------------------------
def _operator(name: str, shift, description: str) -> PythonRule:
    """One row of the table: an operator that moves the order by `shift`."""

    def fn(c: Content) -> Content | None:
        parsed = read(c.text)
        if parsed is None:
            return None
        kind, order, harmonic = parsed
        return Content.abstract(write(kind, order + shift(), harmonic),
                                derivation=description)

    rule = PythonRule(name, fn, ABSTRACT, ABSTRACT, description=description,
                      source=f"C(a)->C(a{'+' if shift() >= 0 else ''}"
                             f"{shift()})",
                      exact=True, trusted=True)
    rule.assumes = (TABLE_6_32,)
    return rule


@_apparatus("coefficient_derivative")
def make_coefficient_derivative() -> PythonRule:
    return _operator("coefficient_derivative", lambda: Fraction(0),
                     "a coefficient derivative in R, Z or T leaves the order "
                     "unchanged")


@_apparatus("radial_derivative")
def make_radial_derivative() -> PythonRule:
    return _operator("radial_derivative", lambda: -KAPPA_S,
                     "the radial derivative D_r costs kappa_s in the order")


@_apparatus("axial_derivative")
def make_axial_derivative() -> PythonRule:
    return _operator("axial_derivative", lambda: Fraction(1),
                     "the axial derivative D_z = eps d_Z gains one order")


@_apparatus("slow_time_derivative")
def make_slow_time_derivative() -> PythonRule:
    return _operator("slow_time_derivative", lambda: Fraction(1),
                     "the slow time derivative -eps d_T gains one order")


@_apparatus("absorption")
def make_absorption() -> PythonRule:
    return _operator("absorption", lambda: Fraction(0),
                     "the absorption operator leaves the order unchanged")


@_apparatus("finite_sum")
def make_finite_sum() -> JoinRule:
    """Two coefficients of the same class and order sum within it."""
    rule = JoinRule("finite_sum", ["M(?a)", "M(?a)"], "M(?a)",
                    description="each class is closed under finite sums at a "
                                "fixed order")
    rule.assumes = (PROP_6_6,)
    return rule


#: Table (9.19): what a PHYSICAL derivative costs in powers of the band
#: scale Q, as against the coefficient derivatives of (6.32). A second
#: operator table with the same shape, which is why the same machinery
#: reads it. The exponents are the paper's.
H = Fraction(1, 100)
PHYSICAL_LOSSES = {
    "radial": Fraction(1, 2) + H * KAPPA_S,
    "axial": Fraction(1, 2) - H,
    "time": 1 + H,
    "frame": Fraction(1, 2),
}
TABLE_9_19 = ("table (9.19): a physical radial derivative costs 1/2 + h "
              "kappa_s in the band scale, an axial one D = 1/2 - h, a time "
              "derivative 1 + h, and a frame derivative 1/2; each further "
              "derivative contributes a further fixed reduction")


def physical_loss(kinds) -> Fraction:
    """What a composition of physical derivatives costs, added up.

    The atomic law underneath is that losses COMPOSE ADDITIVELY, which is
    what makes a table of single derivatives enough to price any
    composition of them. That is associativity of composition together
    with the product rule for the bounds, and it is the whole reason an
    order calculus works at all.
    """
    total = Fraction(0)
    for kind in kinds:
        if kind not in PHYSICAL_LOSSES:
            raise ValueError(f"no physical derivative called {kind!r}")
        total += PHYSICAL_LOSSES[kind]
    return total


def stage_gain(j: int) -> Fraction:
    """`g_j = h j / 10`, the gain Lemma 9.8 records at stage j.

    Linear in the stage and therefore unbounded, which is the only
    property the summation needs: every fixed power of q is eventually
    beaten.
    """
    return H * Fraction(j, 10)


@_apparatus("physical_derivative")
def make_physical_derivative() -> PythonRule:
    """`phys(kinds)` -> the total loss in the band scale.

    Reads a composition such as `phys(radial,axial,time)` and prices it by
    adding the table's entries. Exact, and the addition IS the content:
    a table of four numbers prices every composition because losses add.
    """

    def fn(c: Content) -> Content | None:
        body = c.text.replace(" ", "")
        m = re.fullmatch(r"phys\(([a-z,]+)\)", body)
        if not m:
            return None
        kinds = [k for k in m.group(1).split(",") if k]
        if not kinds or any(k not in PHYSICAL_LOSSES for k in kinds):
            return None
        return Content.abstract(f"loss({physical_loss(kinds)})",
                                derivation=TABLE_9_19)

    rule = PythonRule("physical_derivative", fn, ABSTRACT, ABSTRACT,
                      description="physical derivative losses, added",
                      source="phys(kinds)->loss(total)",
                      exact=True, trusted=True)
    rule.assumes = (TABLE_9_19,)
    return rule


@_apparatus("weakest_term")
def make_weakest_term() -> PythonRule:
    """`pair(M(a);M(b))` -> the weaker of the two orders.

    A sum of terms at different orders is controlled by the worst of
    them. Proposition 6.6 gives closure at a FIXED order, and a sum across
    orders sits at the smallest, because each term is also in every lower
    class. Written as its own rule because the construction's proofs use
    it constantly: they list the contributions and then quote the least.
    """

    def fn(c: Content) -> Content | None:
        body = c.text.replace(" ", "")
        if not (body.startswith("pair(") and body.endswith(")")):
            return None
        halves = body[5:-1].split(";")
        if len(halves) != 2:
            return None
        left, right = (read(h) for h in halves)
        if left is None or right is None or left[0] != right[0]:
            return None
        kind = left[0]
        if kind == "W" and left[2] != right[2]:
            return None                  # different harmonics do not merge
        return Content.abstract(
            write(kind, min(left[1], right[1]), left[2]),
            derivation="a sum across orders sits at the smallest of them, "
                       "since each class contains those above it")

    rule = PythonRule("weakest_term", fn, ABSTRACT, ABSTRACT,
                      description="a sum of terms sits at the weakest order",
                      source="pair(C(a);C(b))->C(min(a,b))",
                      exact=True, trusted=True)
    rule.assumes = ("each coefficient class contains those of higher order, "
                    "so a finite sum sits at the smallest order present",)
    return rule


def _pairing(name: str, left: str, right: str) -> JoinRule:
    """Two established coefficients, offered to the product law as a pair.

    `class_product` reads a `pair(X;Y)` cell, because a product needs two
    inputs and a rule maps one cell to one cell. Nothing built that cell,
    so a proof step of the form "these two multiply" had both its
    premises established and no way to combine them -- which is exactly
    the gap `JoinRule` exists to close, and it went unnoticed because the
    only test of it was a decomposition that never multiplied anything.
    """
    rule = JoinRule(name, [left, right], f"pair({left};{right})",
                    description="offer two coefficients to the product law")
    rule.assumes = (PROP_6_6,)
    return rule


@_apparatus("pair_means")
def make_pair_means() -> JoinRule:
    return _pairing("pair_means", "M(?a)", "M(?b)")


@_apparatus("pair_mean_wave")
def make_pair_mean_wave() -> JoinRule:
    return _pairing("pair_mean_wave", "M(?a)", "W(?b,?n)")


@_apparatus("pair_waves")
def make_pair_waves() -> JoinRule:
    return _pairing("pair_waves", "W(?a,?m)", "W(?b,?n)")


def clears(achieved: str, required) -> bool:
    """Does a derived order meet what the next stage requires?

    The comparison the construction's proofs end on: a list of achieved
    exponents against the thresholds a state at the next stage must have.
    Exact, because the orders are rationals and a threshold missed by a
    hundred-thousandth is missed.
    """
    parsed = read(achieved)
    return parsed is not None and parsed[1] >= Fraction(required)


def install_apparatus(library) -> None:
    for factory in APPARATUS_RULES.values():
        library.add(factory(), replace=True)
