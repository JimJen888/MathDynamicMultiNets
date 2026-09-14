"""
Symbolic differentiation, so a candidate can be checked against an equation.

An existential is proved by exhibiting a witness and verifying it meets
the conditions. `witness.py` does that for the part of Theorem 4.6 that is
a finite linear system. This module does it for the part that is a
differential equation: Lemma A.6 says the explicit similarity field

    K(r, tau) = s^-A H(2 tau / s),        s = r^2 / 2

solves the exterior swirl equation

    -d_tau K = d_rr K + r^-1 d_r K - r^-2 K.

Until now that was checked by differencing the field at sampled points,
which establishes it at those points and nowhere else. Differentiating
symbolically settles it as an identity, and produces something a spot
check cannot: the ordinary differential equation H must satisfy, derived
from the partial differential equation rather than quoted from the paper.

## The fragment, and why it is enough

Everything that appears when the ansatz is substituted has the form

    s^(-A + n) * Z^m * H^(k)(Z)

with n and m integers, k a derivative order, and coefficients rational in
the exponent A. The class is closed under both derivatives, which is the
whole reason a general computer algebra system is not needed here:

    d_s [s^(-A+n) Z^m H^(k)]
        = s^(-A+n-1) [ (n - A - m) Z^m H^(k) - Z^(m+1) H^(k+1) ]
    d_tau [s^(-A+n) Z^m H^(k)]
        = s^(-A+n-1) [ 2m Z^(m-1) H^(k) + 2 Z^m H^(k+1) ]

both read off from dZ/ds = -Z/s and dZ/dtau = 2/s. Radial derivatives go
through s because r enters only through it: d_r = r d_s, so
d_rr = d_s + 2s d_ss and r^-1 d_r = d_s and r^-2 = 1/(2s).

Coefficients are polynomials in A, carried exactly, because the question
Lemma A.6 answers is which A makes the identity close and an answer that
depends on A must keep A symbolic to be worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .symalg import Poly

#: A term is keyed by (power of s relative to -A, power of Z, order of H).
Key = tuple[int, int, int]


@dataclass
class Expr:
    """`sum of coeff(A) * s^(-A+n) * Z^m * H^(k)(Z)`, exactly."""

    terms: dict[Key, Poly] = field(default_factory=dict)

    @staticmethod
    def term(n: int, m: int, k: int, coeff=1) -> "Expr":
        c = coeff if isinstance(coeff, Poly) else Poly.constant(coeff)
        return Expr({} if c.is_zero else {(n, m, k): c})

    def __add__(self, other: "Expr") -> "Expr":
        out = dict(self.terms)
        for key, c in other.terms.items():
            got = out.get(key, Poly.zero()) + c
            if got.is_zero:
                out.pop(key, None)
            else:
                out[key] = got
        return Expr(out)

    def scale(self, factor) -> "Expr":
        f = factor if isinstance(factor, Poly) else Poly.constant(factor)
        return Expr({k: v * f for k, v in self.terms.items()
                     if not (v * f).is_zero})

    def shift(self, dn: int = 0, dm: int = 0) -> "Expr":
        """Multiply by s^dn Z^dm."""
        return Expr({(n + dn, m + dm, k): c
                     for (n, m, k), c in self.terms.items()})

    def __sub__(self, other: "Expr") -> "Expr":
        return self + other.scale(-1)

    @property
    def is_zero(self) -> bool:
        return not self.terms

    # -- the two derivatives ------------------------------------------------
    def d_s(self) -> "Expr":
        out = Expr()
        for (n, m, k), c in self.terms.items():
            # (n - A - m) Z^m H^(k) - Z^(m+1) H^(k+1), all at s^(n-1)
            factor = Poly.constant(n - m) - Poly.sym("A")
            out = out + Expr.term(n - 1, m, k, c * factor)
            out = out + Expr.term(n - 1, m + 1, k + 1, c * Poly.constant(-1))
        return out

    def d_tau(self) -> "Expr":
        out = Expr()
        for (n, m, k), c in self.terms.items():
            if m:
                out = out + Expr.term(n - 1, m - 1, k, c * Poly.constant(2 * m))
            out = out + Expr.term(n - 1, m, k + 1, c * Poly.constant(2))
        return out

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for (n, m, k), c in sorted(self.terms.items()):
            piece = f"({c})"
            if n:
                piece += f" s^({n})" if n != 0 else ""
            if m:
                piece += f" Z^{m}" if m != 1 else " Z"
            piece += f" H^({k})" if k else " H"
            parts.append(piece)
        return "  +  ".join(parts)


def heat_operator(field: Expr) -> Expr:
    """`d_rr K + r^-1 d_r K - r^-2 K`, in the radial variable s.

    r enters only through s = r^2/2, so d_r = r d_s and therefore
    d_rr = d_s + 2s d_ss, r^-1 d_r = d_s, and r^-2 = 1/(2s). Written out
    because getting this wrong would make the whole derivation check a
    different equation and still look tidy.
    """
    d1 = field.d_s()
    d2 = d1.d_s()
    return d1.shift(dn=0) + d2.scale(2).shift(dn=1) + d1 \
        - field.scale(Fraction(1, 2)).shift(dn=-1)


@dataclass
class HeatDerivation:
    """The ODE the profile must satisfy, derived from the PDE."""

    residual: Expr
    ode: dict                       # derivative order -> coefficient in A, Z

    def ode_text(self) -> str:
        parts = []
        for (m, k), c in sorted(self.ode.items()):
            z = "" if m == 0 else (" Z" if m == 1 else f" Z^{m}")
            d = "H" if k == 0 else f"H^({k})"
            parts.append(f"({c}){z} {d}")
        return "  +  ".join(parts) + "  =  0"

    def at(self, A) -> dict:
        """The ODE's coefficients at a concrete exponent."""
        return {key: c.terms.get(((("A", 1),)), Fraction(0))
                * Fraction(A) + c.terms.get((), Fraction(0))
                for key, c in self.ode.items()}


def derive_exterior_ode() -> HeatDerivation:
    """Substitute the similarity field and read off what H must satisfy.

    The residual of the exterior equation, with the ansatz in it, comes
    out as a single power of s times an expression in Z, H, H' and H''.
    That factorisation is the content of the similarity reduction: if it
    did not happen the field would not be self-similar and no choice of H
    would work. What is left is the ordinary differential equation, and it
    is derived here rather than copied from Appendix A.
    """
    field = Expr.term(0, 0, 0)                       # s^-A H(Z)
    residual = heat_operator(field) + field.d_tau()  # RHS + d_tau K

    # Everything should sit at one power of s. Checking that rather than
    # assuming it is the difference between a derivation and a rearrangement.
    powers = {n for (n, _, _) in residual.terms}
    if len(powers) > 1:
        raise AssertionError(
            f"the residual did not reduce to one power of s: {sorted(powers)}; "
            f"the ansatz is not self-similar for this operator")

    ode: dict = {}
    for (_, m, k), c in residual.terms.items():
        ode[(m, k)] = c
    return HeatDerivation(residual=residual, ode=ode)
