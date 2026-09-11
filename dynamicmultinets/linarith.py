"""
Linear arithmetic over the rationals, so a claim about EVERY case can be
decided instead of sampled.

Everything else in this package verifies a rule by asking it questions and
comparing its answers to an oracle's. That establishes a rule on the
instances it was asked about and nothing more, which is why the
Navier-Stokes run reports its imported steps as untrusted however many
instances agree: each of them asserts something uniform -- in the
concentration scale, the dyadic band, the slow label, the correction stage
-- and no number of instances reaches a statement of that shape.

One of those quantifiers is different from the others. "Every correction
stage j" ranges over the non-negative integers, and every order the
correction cycle moves through is *linear in j* with rational coefficients.
A universally quantified linear inequality over such variables is
decidable, and deciding it is a proof rather than a sample. So that
quantifier, alone among the five, can be discharged here -- and with it the
induction of Proposition 9.6, which is the one place the construction is a
recursion rather than an estimate.

The fragment:

    Lin        a rational constant plus rational multiples of named variables
    MinSet     the pointwise minimum of finitely many Lin, kept UNRESOLVED
    obligations()   a scope in which guards are recorded, not decided
    solve()    the whole conjunction, decided over every value at once

`MinSet` is the reason this stays small. A minimum of linear forms is
piecewise linear, and resolving which piece applies would need a case split
at every crossover. It is never necessary here, because the only thing
asked of a minimum is whether it clears a bound, and

    min(t_1, ..., t_k) >= L   everywhere   <=>   t_i >= L everywhere, each i

turns one piecewise question into k straight ones.

## Variables, and the one that is solved for rather than quantified

`solve` treats every variable as ranging over [0, infinity) and asks whether
the inequality holds throughout -- except one, which may be nominated as the
PARAMETER. Instead of being quantified, the parameter is solved for: the
answer comes back as the interval of values that make every inequality hold.
That is how the correction cycle's budget stops being a number quoted from
the paper and becomes something the machine derives.

## Why comparisons are recorded rather than evaluated

The rules whose claims are being proved are ordinary Python: they compute
with `Fraction`, take minima, and return `None` when a guard fails. Proving
something about a *copy* of that code written for the prover would prove
nothing about the rule. So the rules call `least` and `short` instead of
`min` and `<`, and those two do the ordinary thing on ordinary numbers.
Inside an `obligations()` scope, handed symbolic values, `least` builds a
`MinSet` and `short` records the guard it was asked about and answers
False -- letting the rule run on as if the guard had held, and leaving the
prover to discharge what was assumed.

That is exactly the shape of the argument being formalised. The cycle
closes at stage n PROVIDED its three gains reach 1/10; the prover runs the
cycle to get the conclusion and then proves the proviso for every n at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

Number = Fraction | int


def _clean(terms: dict[str, Fraction]) -> dict[str, Fraction]:
    return {k: v for k, v in sorted(terms.items()) if v}


@dataclass
class Lin:
    """`const + sum(coeff * variable)`, all coefficients rational."""

    const: Fraction = Fraction(0)
    terms: dict[str, Fraction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.const = Fraction(self.const)
        self.terms = _clean({k: Fraction(v) for k, v in self.terms.items()})

    # -- construction --------------------------------------------------------
    @staticmethod
    def var(name: str, coeff: Number = 1) -> "Lin":
        return Lin(Fraction(0), {name: Fraction(coeff)})

    # -- arithmetic ----------------------------------------------------------
    def _coerce(self, other: Any) -> "Lin | None":
        if isinstance(other, Lin):
            return other
        if isinstance(other, (int, Fraction)):
            return Lin(Fraction(other), {})
        return None

    def __add__(self, other: Any) -> "Lin":
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        terms = dict(self.terms)
        for k, v in o.terms.items():
            terms[k] = terms.get(k, Fraction(0)) + v
        return Lin(self.const + o.const, terms)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Lin":
        o = self._coerce(other)
        return NotImplemented if o is None else self + (-o)

    def __rsub__(self, other: Any) -> "Lin":
        o = self._coerce(other)
        return NotImplemented if o is None else o + (-self)

    def __mul__(self, other: Any) -> "Lin":
        # Only by a constant: a product of two non-constant forms would
        # leave the fragment, and nothing in the cycle needs one.
        if isinstance(other, (int, Fraction)):
            f = Fraction(other)
            return Lin(self.const * f, {k: v * f for k, v in self.terms.items()})
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self) -> "Lin":
        return Lin(-self.const, {k: -v for k, v in self.terms.items()})

    def __eq__(self, other: Any) -> bool:
        o = self._coerce(other)
        return o is not None and self.const == o.const and self.terms == o.terms

    def coeff(self, name: str) -> Fraction:
        return self.terms.get(name, Fraction(0))

    def variables(self) -> set[str]:
        return set(self.terms)

    def at(self, **values: Number) -> "Lin":
        """Substitute values for some variables; a fully ground form has
        an empty `terms` and its `const` is the value."""
        out = Lin(self.const, dict(self.terms))
        for name, value in values.items():
            if name in out.terms:
                out = Lin(out.const + out.terms[name] * Fraction(value),
                          {k: v for k, v in out.terms.items() if k != name})
        return out

    def value(self) -> Fraction:
        if self.terms:
            raise ValueError(f"{self} is not a number")
        return self.const

    def __str__(self) -> str:
        parts = [str(self.const)] if self.const or not self.terms else []
        for name, c in self.terms.items():
            sign = "+" if c > 0 else "-"
            mag = abs(c)
            body = name if mag == 1 else f"{mag}{name}"
            parts.append(f"{sign} {body}" if parts else
                         (body if c > 0 else f"-{body}"))
        return " ".join(parts)


@dataclass
class MinSet:
    """The pointwise minimum of `terms`, deliberately not resolved."""

    terms: tuple[Lin, ...]

    def __sub__(self, other: Any) -> "MinSet":
        # min(t_i) - c = min(t_i - c) for any c, so a shift distributes over
        # an unresolved minimum without deciding which branch applies.
        if isinstance(other, (Lin, int, Fraction)):
            return MinSet(tuple(t - other for t in self.terms))
        return NotImplemented

    def __str__(self) -> str:
        return "min(" + ", ".join(str(t) for t in self.terms) + ")"


Symbolic = Lin | MinSet


def _branches(x: Any) -> tuple[Lin, ...]:
    if isinstance(x, MinSet):
        return x.terms
    if isinstance(x, Lin):
        return (x,)
    return (Lin(Fraction(x)),)


#: Public name for the branches of a (possibly unresolved) minimum.
branches = _branches


def is_symbolic(x: Any) -> bool:
    return isinstance(x, (Lin, MinSet))


# ---------------------------------------------------------------------------
# The scope in which guards are recorded instead of decided
# ---------------------------------------------------------------------------
@dataclass
class Obligation:
    """A guard the rule relied on: `lower >= bound`, needed everywhere."""

    lower: Symbolic
    bound: Symbolic
    where: str = ""
    #: True when the guard was a strict inequality. `short` records
    #: non-strict obligations and `requires` records strict ones; the
    #: difference matters when the answer is reported as a hypothesis
    #: rather than solved to an interval.
    strict: bool = False

    def as_requirement(self) -> str:
        relation = ">" if self.strict else ">="
        return f"{self.lower} {relation} {self.bound}"


_open_scopes: list[list[Obligation]] = []


class obligations:
    """Run rule code symbolically, collecting the guards it relied on."""

    def __init__(self) -> None:
        self.collected: list[Obligation] = []

    def __enter__(self) -> list[Obligation]:
        _open_scopes.append(self.collected)
        return self.collected

    def __exit__(self, *exc: Any) -> None:
        _open_scopes.pop()


def recording() -> bool:
    return bool(_open_scopes)


# ---------------------------------------------------------------------------
# What rule code calls instead of `min` and `<`
# ---------------------------------------------------------------------------
def least(*args: Any) -> Any:
    """`min`, except that on symbolic arguments it defers the choice."""
    if not any(is_symbolic(a) for a in args):
        return min(args)
    collected: list[Lin] = []
    for a in args:
        for t in _branches(a):
            if t not in collected:
                collected.append(t)
    return MinSet(tuple(collected)) if len(collected) > 1 else collected[0]


def short(value: Any, bound: Any, where: str = "") -> bool:
    """`value < bound`, except that symbolically it records and answers False.

    Answering False is not optimism, it is the shape of the theorem. The
    rule is being asked "did this gain fall short", and the prover's job is
    to show it never does; letting the rule proceed yields the conclusion
    that holds WHEN it never does, and the recorded obligation is exactly
    the hypothesis that conclusion is under.
    """
    if not (is_symbolic(value) or is_symbolic(bound)):
        return value < bound
    if not recording():
        raise RuntimeError(
            "symbolic values reached `short` outside an obligations() scope: "
            "the guard would have to be decided here and it cannot be"
        )
    _open_scopes[-1].append(Obligation(value, bound, where))
    return False


def requires(value: Any, bound: Any, where: str = "") -> bool:
    """`value < bound`, as a guard that ACCEPTS when it holds.

    The dual of `short`, and the two are not interchangeable. `short` asks
    "did this fall short", so a rule declines when it answers True.
    `requires` asks "is this within the bound", so a rule proceeds when it
    answers True. Symbolically both record and then answer whichever way
    lets the rule carry on, because the point of running a rule on
    symbols is to find out what it would need, not to stop at the first
    thing that cannot be decided.

    Recorded strictly: the dyadic sum converges when the exponent is below
    zero, not at zero, and reporting that as a non-strict requirement
    would be wrong at exactly the boundary the guard exists to exclude.
    """
    if not (is_symbolic(value) or is_symbolic(bound)):
        return value < bound
    if not recording():
        raise RuntimeError(
            "symbolic values reached `requires` outside an obligations() "
            "scope: the guard would have to be decided here and it cannot be"
        )
    _open_scopes[-1].append(Obligation(bound, value, where, strict=True))
    return True


def requirements(obs: Iterable[Obligation]) -> list[str]:
    """The collected guards, as the hypothesis a derivation would need."""
    out: list[str] = []
    for ob in obs:
        for bound in _branches(ob.bound):
            for lower in _branches(ob.lower):
                diff = lower - bound
                relation = "> 0" if ob.strict else ">= 0"
                line = f"{diff} {relation}"
                if ob.where:
                    line += f"   [{ob.where}]"
                if line not in out:
                    out.append(line)
    return out


# ---------------------------------------------------------------------------
# Deciding the whole conjunction
# ---------------------------------------------------------------------------
@dataclass
class Region:
    """Where every recorded inequality holds.

    `feasible` is False when some inequality fails no matter what the
    parameter is -- a variable it grows negatively in is unbounded, or it is
    already negative and the parameter cannot help. Otherwise `lower` and
    `upper` are the closed bounds on the parameter, either possibly None for
    unbounded, and the claim holds for exactly that interval.
    """

    feasible: bool = True
    lower: Fraction | None = None
    upper: Fraction | None = None
    blocking: list[str] = field(default_factory=list)
    parameter: str = ""
    decided: int = 0

    def contains(self, value: Number) -> bool:
        v = Fraction(value)
        if not self.feasible:
            return False
        return ((self.lower is None or v >= self.lower)
                and (self.upper is None or v <= self.upper))

    def describe(self) -> str:
        if not self.feasible:
            return "no value works: " + "; ".join(self.blocking[:3])
        if self.lower is None and self.upper is None:
            return "every value works"
        p = self.parameter or "the parameter"
        if self.lower is None:
            return f"{p} <= {self.upper}"
        if self.upper is None:
            return f"{p} >= {self.lower}"
        return f"{self.lower} <= {p} <= {self.upper}"


def solve_for(equation: Lin, name: str) -> Lin:
    """Rearrange `equation == 0` to `name == ...`, exactly.

    Used where a scaling argument FORCES an exponent rather than bounding
    it. The variable has to appear linearly, which in this fragment it
    always does, and what comes back is a linear form in whatever else was
    in the equation -- so the answer stays symbolic and the derivation
    covers every value of the other exponents at once.
    """
    c = equation.coeff(name)
    if c == 0:
        raise ValueError(f"{name!r} does not appear in {equation}")
    rest = equation - Lin.var(name, c)
    return rest * (Fraction(-1) / c)


def solve(obs: Iterable[Obligation], parameter: str | None = None) -> Region:
    """Decide `lower >= bound` for every recorded guard, everywhere at once.

    Every variable other than `parameter` is taken to range over
    [0, infinity). A linear form is non-negative throughout such a range
    exactly when each of those coefficients is non-negative and the form is
    non-negative with them all at zero, so nothing here searches: it reads
    the coefficients.
    """
    region = Region(parameter=parameter or "")
    for ob in obs:
        for bound in _branches(ob.bound):
            for branch in _branches(ob.lower):
                diff = branch - bound
                region.decided += 1
                bad = False
                for name, c in diff.terms.items():
                    if name == parameter:
                        continue
                    if c < 0:
                        region.feasible = False
                        region.blocking.append(
                            f"{ob.where or 'a guard'}: {diff} falls without "
                            f"bound as {name} grows")
                        bad = True
                if bad:
                    continue
                # Every quantified variable at its minimum, zero.
                b = diff.coeff(parameter) if parameter else Fraction(0)
                c = diff.const
                if b == 0:
                    if c < 0:
                        region.feasible = False
                        region.blocking.append(
                            f"{ob.where or 'a guard'}: {diff} is negative")
                elif b > 0:
                    lo = -c / b
                    region.lower = lo if region.lower is None else max(region.lower, lo)
                else:
                    hi = -c / b
                    region.upper = hi if region.upper is None else min(region.upper, hi)
    if (region.feasible and region.lower is not None
            and region.upper is not None and region.lower > region.upper):
        region.feasible = False
        region.blocking.append("the bounds on the parameter cross")
    return region
