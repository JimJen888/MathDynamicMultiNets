"""
Building a witness, instead of assuming one exists.

Most of the eleven imported steps are existentials, and the last several
experiments here kept running into the same wall: an implication registry
transforms properties of an object already named and never names one. No
chain of "if P then Q" produces "there exists x with P(x)". You need a
witness, and a witness is built.

So this builds one, as far as it can be built exactly.

Theorem 4.6 asserts profiles with four properties: a regular axis, an exact
heat exterior, four moment identities, and an annular stress inside the
cone. Those four are not alike. The moment identities are a FINITE linear
system -- finitely many radial powers, finitely many conditions, solve for
the coefficients -- and a finite linear system over the rationals is
something this machine can solve exactly and check exactly. The other three
quantify over a continuum.

That split is the whole content of this module. It does not prove Theorem
4.6. What it does is move the existential from a bare assumption to an
explicit candidate with its finite data constructed, its finite conditions
verified exactly, and the remaining conditions named. A reader auditing the
argument can then see which part of "there exists a profile" is witnessed
and which part is still taken on trust, instead of seeing one word.

## Why the arithmetic is exact

The construction's radial powers are half-integers, and a half-integer
power of a rational is irrational, so integrating `x^alpha` over rational
endpoints leaves the rationals immediately. Substituting `x = t^2` fixes
it: `integral x^alpha dx` becomes `2 * integral t^(2 alpha + 1) dt`, the
exponent is an integer whenever alpha is a half-integer, and every moment
is a rational computed in closed form. The substitution is not a trick
imported for this module -- `s = r^2/2` is the construction's own radial
variable and appears in the exterior field already.

Exactness matters here more than usual. The question is whether a linear
system has a solution, and a numerically tiny determinant is the same
thing as a singular one to a float and a completely different thing to the
theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence


def half_integer(value) -> bool:
    """Is this a half-integer, so that the substitution makes it exact?"""
    return Fraction(value).denominator in (1, 2)


def moment(alpha: Fraction, lo: Fraction, hi: Fraction) -> Fraction:
    """`integral from lo to hi of x^alpha dx`, exactly, for half-integer alpha.

    Under x = t^2 this is 2 * integral t^(2a+1) dt between sqrt(lo) and
    sqrt(hi), so the endpoints are taken in t from the start: callers pass
    the interval in the substituted variable and the exponent is an
    integer.
    """
    n = 2 * Fraction(alpha) + 1
    if n.denominator != 1:
        raise ValueError(f"{alpha} is not a half-integer, so the moment is "
                         f"not rational and this module will not guess at it")
    power = int(n) + 1
    if power == 0:
        raise ValueError("a logarithmic moment is outside this fragment")
    return 2 * (hi ** power - lo ** power) / power


def moment_matrix(exponents: Sequence, intervals: Sequence) -> list[list[Fraction]]:
    """`M[j][i]` is the moment of `x^alpha_i` over the jth test interval.

    Test functions are indicators of disjoint intervals, which is the
    crudest choice that works and the only one whose integrals stay
    rational. Lemma A.1's content is that distinct powers make this
    invertible; here it is built and solved rather than characterised.
    """
    return [[moment(a, lo, hi) for a in exponents] for lo, hi in intervals]


def solve_exact(matrix: list[list[Fraction]],
                rhs: Sequence) -> list[Fraction] | None:
    """Gaussian elimination over the rationals. None when singular.

    Exact, so "singular" means singular rather than ill-conditioned.
    """
    n = len(matrix)
    aug = [[Fraction(x) for x in row] + [Fraction(rhs[j])]
           for j, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]
    return [row[n] for row in aug]


@dataclass
class Construction:
    """A witness, with its finite conditions checked and the rest named."""

    exponents: tuple
    intervals: tuple
    targets: tuple
    coefficients: tuple | None = None
    #: conditions verified exactly on the constructed object
    verified: list[str] = field(default_factory=list)
    #: conditions this machine cannot reach, named rather than omitted
    outstanding: list[str] = field(default_factory=list)
    obstruction: str = ""

    @property
    def built(self) -> bool:
        return self.coefficients is not None

    def report(self) -> str:
        if not self.built:
            return f"no witness: {self.obstruction}"
        lines = ["a witness was constructed, not assumed:"]
        lines.append("  coefficients " + ", ".join(
            str(c) for c in self.coefficients))
        lines.append("  over radial powers " + ", ".join(
            str(a) for a in self.exponents))
        lines.append(f"  verified exactly ({len(self.verified)}):")
        for v in self.verified:
            lines.append(f"    - {v}")
        lines.append(f"  still assumed about this candidate "
                     f"({len(self.outstanding)}):")
        for o in self.outstanding:
            lines.append(f"    - {o}")
        lines.append("  so the existential is witnessed in its finite part "
                     "and open in the rest, which is a different and more "
                     "useful thing to report than 'assumed'")
        return "\n".join(lines)


#: Theorem 4.6's four conditions, taken verbatim from the statement as it
#: is recorded in `navierstokes._IMPORTED`, each with what has been done
#: about it. Written as the full four rather than as a list of leftovers
#: so the tally cannot quietly shrink to whatever happened to be
#: checkable: a condition that gets addressed moves from one column to the
#: other in view, and one that does not stays on the page.
CONDITIONS = (
    ("a regular axis",
     "NOT ATTEMPTED. Regularity at r = 0 is a statement about a limit, and "
     "nothing here represents one. The profile's pieces are explicit, so "
     "this looks like the same decomposition-into-prior-facts the other "
     "lemmas took, and it has not been tried.",
     False),
    ("an exact heat exterior",
     "PARTLY DONE. `symdiff` substitutes the similarity field and "
     "differentiates, reducing the exterior equation to one ordinary "
     "differential equation, symbolically and for every exponent. WHICH "
     "exponent solves it depends on the profile, the profile is an "
     "integral, and that half is quadrature.",
     False),
    ("the four moment identities",
     "DONE. A finite linear system over the rationals, solved exactly by "
     "`build_moment_profile` and checked on the nose rather than to a "
     "tolerance.",
     True),
    ("an annular stress inside the cone",
     "NOT ESTABLISHED. `lemma_4_5_cone` agrees with two independent "
     "oracles across thousands of instances, which is evidence about "
     "those instances. The condition is that the stress lies in the cone "
     "THROUGHOUT, and no number of instances reaches that.",
     False),
)

#: What the construction still owes, derived from the four so the two
#: cannot drift apart.
OUTSTANDING = tuple(f"{name}: {status}" for name, status, done in CONDITIONS
                    if not done)


def build_moment_profile(exponents: Sequence, targets: Sequence,
                         intervals: Sequence | None = None) -> Construction:
    """Solve Theorem 4.6's moment conditions and check the solution exactly.

    This is the part of "there exist profiles with four properties" that a
    finite computation can actually deliver: coefficients over the given
    radial powers meeting the given moment conditions, produced rather than
    asserted to exist, with the conditions then verified on the result.

    What comes back always names what it did not establish. A witness whose
    unchecked conditions were left off the report would be worse than no
    witness, because it would read like a proof of the existential.
    """
    exponents = tuple(Fraction(a) for a in exponents)
    targets = tuple(Fraction(t) for t in targets)
    if intervals is None:
        # Disjoint unit intervals in the substituted variable, one per
        # condition, ordered so the matrix is not accidentally symmetric.
        intervals = tuple((Fraction(1 + j), Fraction(2 + j))
                          for j in range(len(targets)))
    intervals = tuple((Fraction(lo), Fraction(hi)) for lo, hi in intervals)

    if len(exponents) != len(targets):
        return Construction(exponents, intervals, targets, obstruction=(
            f"{len(exponents)} powers against {len(targets)} conditions; the "
            f"system is not square and this module does not choose which "
            f"conditions to drop"))
    if len(set(exponents)) != len(exponents):
        return Construction(exponents, intervals, targets, obstruction=(
            "the radial powers repeat, and Lemma A.1's invertibility is "
            "exactly the claim that distinct powers are needed"))
    if not all(half_integer(a) for a in exponents):
        return Construction(exponents, intervals, targets, obstruction=(
            "a power is not a half-integer, so its moments leave the "
            "rationals and nothing here would be exact"))

    matrix = moment_matrix(exponents, intervals)
    coefficients = solve_exact(matrix, targets)
    if coefficients is None:
        return Construction(exponents, intervals, targets, obstruction=(
            "the moment matrix is singular, exactly rather than nearly, so "
            "no combination of these powers meets these conditions"))

    verified = []
    for j, ((lo, hi), want) in enumerate(zip(intervals, targets)):
        got = sum(c * moment(a, lo, hi)
                  for c, a in zip(coefficients, exponents))
        if got != want:
            return Construction(exponents, intervals, targets, obstruction=(
                f"the solved coefficients miss condition {j}: {got} != {want}, "
                f"which in exact arithmetic means the solver is wrong"))
        verified.append(f"moment condition {j} holds exactly: the combination "
                        f"integrates to {want} over [{lo}, {hi}]")
    verified.append(f"the {len(exponents)} radial powers are distinct, which "
                    f"is what Lemma A.1 requires")
    verified.append("the moment matrix is invertible over the rationals, "
                    "decided by elimination rather than by a determinant "
                    "near zero")

    return Construction(exponents, intervals, targets,
                        coefficients=tuple(coefficients),
                        verified=verified, outstanding=list(OUTSTANDING))
