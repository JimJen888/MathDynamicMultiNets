"""
Where the zeta zeros come from.

This module computes the imaginary parts of the nontrivial zeros of the Riemann
zeta function on the critical line, and the spacing statistics built from them.
It is the one place in the package that produces data the machine did not draw
itself and cannot derive: the zeros are a fact about zeta, and a rule checked
against them is being checked against the world rather than against the
renderer. That is what `measured` grounding means in oracles.py.

Method: the Riemann-Siegel formula. Z(t) is real on the critical line and its
sign changes are the zeros, so locating them is a scan plus a bisection rather
than anything delicate. The formula is

    Z(t) = 2 * sum_{n=1}^{N} n^{-1/2} cos(theta(t) - t log n)  +  R(t),
    N    = floor(sqrt(t / 2pi)),

with theta the Riemann-Siegel theta function and R(t) an asymptotic remainder;
the leading correction term is enough here and its accuracy is checked in
tests/ against mpmath's independent `zetazero`, which uses a different method.

Why not just call mpmath: `zetazero(n)` is exact to arbitrary precision and
takes on the order of a millisecond per zero at these heights, which is fine for
ten zeros and not for the thousands a spacing histogram needs. It is used as the
INDEPENDENT CHECK instead, which is the better use for it -- an implementation
verified against a different implementation is worth more than either alone.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def theta(t: float) -> float:
    """The Riemann-Siegel theta function, by its asymptotic expansion.

    Valid for t well above 1, which is the only range zeros live in (the first
    is at 14.13). The expansion is in odd inverse powers of t and the terms
    shown are enough for double precision at any height reachable here.
    """
    return (0.5 * t * math.log(t / TWO_PI) - 0.5 * t - math.pi / 8
            + 1.0 / (48.0 * t) + 7.0 / (5760.0 * t ** 3))


def zeta_z(t: float) -> float:
    """Hardy's Z(t): real-valued, and |Z(t)| = |zeta(1/2 + it)|.

    Working with Z instead of zeta directly is the whole point of the
    Riemann-Siegel formula -- a zero of zeta ON THE CRITICAL LINE is a SIGN
    CHANGE of a real function, so finding one is bisection rather than a search
    in the complex plane. (It says nothing about zeros off the line, which is
    what RH is about, and this module cannot see them.)
    """
    if t <= 0:
        raise ValueError(f"Z(t) is computed for t > 0, not {t}")
    th = theta(t)
    n_terms = int(math.sqrt(t / TWO_PI))
    if n_terms < 1:
        n_terms = 1
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    main = 2.0 * np.sum(np.cos(th - t * np.log(n)) / np.sqrt(n))

    # Leading remainder term. p is the fractional part of sqrt(t/2pi); the
    # cosine ratio is the standard C0 correction and it is what takes the
    # formula from "roughly right between zeros" to "accurate enough that a
    # sign change is where the zero is".
    root = math.sqrt(t / TWO_PI)
    p = root - int(root)
    remainder = ((-1) ** (n_terms - 1) * (TWO_PI / t) ** 0.25
                 * math.cos(TWO_PI * (p * p - p - 1.0 / 16.0)) / math.cos(TWO_PI * p))
    return main + remainder


def scan_step(t: float, fraction: float = 0.02) -> float:
    """Scan step at height t: a small fraction of the LOCAL mean spacing.

    A fixed step is the wrong shape for this problem and was the first bug
    here. Zeros thin out logarithmically, so a step that comfortably resolves
    them at t=100 is too coarse at t=60000, and the failure is silent: zeros
    skipped in PAIRS leave the sign pattern of Z intact, so the scan reports
    success and the missing zeros show up only as two spacings merged into one.

    Worse, the failure is biased in exactly the direction that matters here.
    What gets skipped is the CLOSEST pairs, so the spacing distribution loses
    density at small s -- which is the same signature as level repulsion, and
    would masquerade as the finite-height deviation this module exists to
    measure. A fixed step of 0.05 lost ~9 zeros in 80000 at t~60000: harmless
    for classifying an ensemble, not harmless for measuring a deviation.

    Sizing: the mean spacing is 2pi/log(t/2pi), and taking 2% of it puts the
    expected number of missed pairs below one for samples of this size, since
    GUE spacings vanish like s^2 near the origin and P(s < 0.02) ~ 1e-5.
    """
    return fraction * TWO_PI / math.log(max(t, 8.0) / TWO_PI)


def zeta_zeros(count: int, step: float | None = None,
               start: float = 13.0) -> np.ndarray:
    """The first `count` zero heights, in increasing order.

    A scan for sign changes, then bisection on each. `step=None` (the default)
    adapts it to the local mean spacing via `scan_step`; passing a number
    forces a fixed step, which is what the resolution controls in tests/ do.

    Completeness is not assumed: the count found below a height is checked
    against the Riemann-von Mangoldt formula, because skipping zeros in pairs
    preserves sign and would otherwise be invisible.
    """
    out: list[float] = []
    t, prev = start, zeta_z(start)
    while len(out) < count:
        t_next = t + (scan_step(t) if step is None else step)
        cur = zeta_z(t_next)
        if prev == 0.0:
            out.append(t)
        elif prev * cur < 0.0:
            lo, hi, f_lo = t, t_next, prev
            for _ in range(50):         # to machine precision on the bracket
                mid = 0.5 * (lo + hi)
                f_mid = zeta_z(mid)
                if f_lo * f_mid <= 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            out.append(0.5 * (lo + hi))
        t, prev = t_next, cur
    return np.array(out[:count])


def zero_counting_function(t: float) -> float:
    """N(T), the expected number of zeros with height below T.

    The smooth part of the Riemann-von Mangoldt formula. It is used for two
    different jobs here and they should not be confused: as a CHECK that the
    scan did not miss zeros, and as the UNFOLDING map below -- the zeros thin
    out logarithmically, and a spacing distribution only means anything after
    that trend is divided out.
    """
    return theta(t) / math.pi + 1.0


def unfolded_spacings(zeros: np.ndarray) -> np.ndarray:
    """Consecutive gaps between zeros, rescaled to mean spacing 1.

    Unfolding is not cosmetic. Raw gaps between zeta zeros shrink like
    1/log(t), so a histogram of them measures the height of the sample rather
    than anything about the zeros' arrangement. Mapping each zero through N(t)
    removes exactly that trend, and what is left -- whether the unfolded gaps
    repel like random-matrix eigenvalues or fall where they like -- is the
    question Montgomery and Odlyzko asked.
    """
    unfolded = np.array([zero_counting_function(float(t)) for t in zeros])
    return np.diff(unfolded)


# ---------------------------------------------------------------------------
# The ensembles the spacings are compared against
# ---------------------------------------------------------------------------
def _unfold_eigenvalues(vals: np.ndarray, keep: float = 0.5) -> np.ndarray:
    """Middle-window spacings of one matrix's spectrum, normalised to mean 1.

    The bulk only. Eigenvalue density follows the semicircle and falls to zero
    at the edges, so edge spacings are drawn from a different distribution than
    bulk ones; including them would blur the very feature that separates the
    ensembles.
    """
    vals = np.sort(vals)
    n = len(vals)
    lo = int(n * (1.0 - keep) / 2.0)
    hi = n - lo
    gaps = np.diff(vals[lo:hi])
    return gaps / gaps.mean()


def ensemble_spacings(kind: str, n_spacings: int, rng: np.random.Generator,
                      dim: int = 60) -> np.ndarray:
    """Unfolded level spacings from one of the three reference ensembles.

    GUE and GOE come from ACTUAL random matrices rather than from the Wigner
    surmise. The surmise is a 2x2 result used as an approximation to the
    large-matrix law, and it is a very good one -- but the machine is about to
    be asked whether the zeta spacings look like GUE, and answering that with a
    rule trained on an approximation to GUE would put the approximation's error
    inside the conclusion. Sampling eigenvalues costs a few milliseconds and
    removes the question.

      gue      complex Hermitian Gaussian; levels repel like s^2 at small s
      goe      real symmetric Gaussian; repulsion is linear in s
      poisson  independent uniform points; no repulsion at all, p(s)=exp(-s)
    """
    if kind not in ENSEMBLES:
        raise ValueError(f"unknown ensemble {kind!r}; known: {ENSEMBLES}")
    out: list[np.ndarray] = []
    while sum(len(a) for a in out) < n_spacings:
        if kind == "poisson":
            pts = np.sort(rng.uniform(0.0, 1.0, dim))
            gaps = np.diff(pts)
            out.append(gaps / gaps.mean())
            continue
        if kind == "gue":
            a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            h = (a + a.conj().T) / 2.0
        else:                                   # goe
            a = rng.normal(size=(dim, dim))
            h = (a + a.T) / 2.0
        out.append(_unfold_eigenvalues(np.linalg.eigvalsh(h)))
    return np.concatenate(out)[:n_spacings]


ENSEMBLES = ("gue", "goe", "poisson")
