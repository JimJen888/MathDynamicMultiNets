"""
Exact polynomial algebra, for the derivations that are plain enough to do.

`linarith` decides inequalities that are linear in an index. This is the
other decidable fragment the Navier-Stokes construction leans on, and it is
the more useful of the two: several of the paper's derivations are not
estimates at all, they are *identities*, and an identity between polynomial
expressions is settled by expanding both sides.

The point that makes this a proof rather than a spot check is worth stating
plainly, because it is the whole argument.

A pointwise identity among a field and its derivatives -- say

    (u+w).grad(u+w) = u.grad u + u.grad w + w.grad u + w.grad w

-- involves the values of the fields and of their partial derivatives AT ONE
POINT, and involves them algebraically. Nothing in it knows that `du[i][j]`
came from differentiating `u`. So take those values as INDEPENDENT SYMBOLS,
expand both sides as polynomials in them, and subtract: if the difference is
the zero polynomial, the identity holds at every point of every smooth
field, with no sampling anywhere. If it is not zero, what is left over is
the defect, in closed form, and that is usually the more interesting answer.

This is why the increment identity of Section 3.3 can be *proved* here while
the estimates around it cannot. The identity is algebra in the one-jet. The
estimates are statements about norms.

What this module deliberately is not: a computer algebra system. There is no
simplification, no factoring, no substitution of functions for symbols, and
no differentiation. Polynomials go in as sums of products of symbols, and
the only question ever asked of them is whether two are equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping

Number = Fraction | int

#: A monomial is a sorted tuple of (symbol, power), powers positive.
Monomial = tuple[tuple[str, int], ...]

ONE: Monomial = ()


def _times(a: Monomial, b: Monomial) -> Monomial:
    powers: dict[str, int] = {}
    for name, p in a + b:
        powers[name] = powers.get(name, 0) + p
    return tuple(sorted((n, p) for n, p in powers.items() if p))


@dataclass(frozen=True)
class Poly:
    """A polynomial over the rationals in named symbols."""

    terms: Mapping[Monomial, Fraction]

    # -- construction --------------------------------------------------------
    @staticmethod
    def zero() -> "Poly":
        return Poly({})

    @staticmethod
    def constant(value: Number) -> "Poly":
        v = Fraction(value)
        return Poly({} if v == 0 else {ONE: v})

    @staticmethod
    def sym(name: str) -> "Poly":
        return Poly({((name, 1),): Fraction(1)})

    @staticmethod
    def summing(parts: Iterable["Poly"]) -> "Poly":
        out = Poly.zero()
        for p in parts:
            out = out + p
        return out

    # -- arithmetic ----------------------------------------------------------
    def _coerce(self, other: Any) -> "Poly | None":
        if isinstance(other, Poly):
            return other
        if isinstance(other, (int, Fraction)):
            return Poly.constant(other)
        return None

    def __add__(self, other: Any) -> "Poly":
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        terms = dict(self.terms)
        for m, c in o.terms.items():
            got = terms.get(m, Fraction(0)) + c
            if got:
                terms[m] = got
            else:
                terms.pop(m, None)
        return Poly(terms)

    __radd__ = __add__

    def __neg__(self) -> "Poly":
        return Poly({m: -c for m, c in self.terms.items()})

    def __sub__(self, other: Any) -> "Poly":
        o = self._coerce(other)
        return NotImplemented if o is None else self + (-o)

    def __rsub__(self, other: Any) -> "Poly":
        o = self._coerce(other)
        return NotImplemented if o is None else o + (-self)

    def __mul__(self, other: Any) -> "Poly":
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        terms: dict[Monomial, Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in o.terms.items():
                m = _times(m1, m2)
                got = terms.get(m, Fraction(0)) + c1 * c2
                if got:
                    terms[m] = got
                else:
                    terms.pop(m, None)
        return Poly(terms)

    __rmul__ = __mul__

    # -- questions -----------------------------------------------------------
    @property
    def is_zero(self) -> bool:
        return not self.terms

    def __eq__(self, other: Any) -> bool:
        o = self._coerce(other)
        return o is not None and self.terms == o.terms

    def symbols(self) -> set[str]:
        return {name for m in self.terms for name, _ in m}

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for m, c in sorted(self.terms.items()):
            body = "*".join(n if p == 1 else f"{n}^{p}" for n, p in m)
            if not body:
                parts.append(str(c))
            elif c == 1:
                parts.append(body)
            elif c == -1:
                parts.append(f"-{body}")
            else:
                parts.append(f"{c}*{body}")
        return " + ".join(parts).replace("+ -", "- ")


def vector(name: str, dims: int = 3) -> list[Poly]:
    """`name0, name1, name2` as independent symbols."""
    return [Poly.sym(f"{name}{i}") for i in range(dims)]


def jacobian(name: str, dims: int = 3) -> list[list[Poly]]:
    """`d_j name_i` as independent symbols, indexed [i][j].

    Independent is the operative word and the reason this works: a pointwise
    identity cannot tell that these came from differentiating anything, so
    proving it for independent symbols proves it for every field.
    """
    return [[Poly.sym(f"d{name}{i}_{j}") for j in range(dims)]
            for i in range(dims)]


def divergence(jac: list[list[Poly]]) -> Poly:
    return Poly.summing(jac[i][i] for i in range(len(jac)))


def advect(field: list[Poly], jac: list[list[Poly]]) -> list[Poly]:
    """`(field . grad) v`, with `jac` the Jacobian of v: field_j d_j v_i."""
    dims = len(field)
    return [Poly.summing(field[j] * jac[i][j] for j in range(dims))
            for i in range(dims)]


def tensor_divergence(field: list[Poly], jac: list[list[Poly]]) -> list[Poly]:
    """`div(field tensor field)_i = d_j(f_i f_j) = f_j d_j f_i + f_i div f`."""
    dims = len(field)
    div = divergence(jac)
    return [Poly.summing(field[j] * jac[i][j] for j in range(dims))
            + field[i] * div for i in range(dims)]
