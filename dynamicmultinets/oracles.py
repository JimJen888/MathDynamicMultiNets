"""
Oracles: what is true, and how we know it.

An oracle labels an experiment. The interesting part is not the labelling, it
is the `kind` field, because a rule is only ever as good as the thing it was
checked against and the machine has to keep track of which is which:

  definitional  bottoms out in a definition the machine holds independently of
                any rule -- multiplication as repeated addition. Slow, narrow,
                and the only thing here that can ground a claim on its own.
  derived       computed by composing rules the machine already trusts. This is
                the paper's "facts reasoned by other existing rules". Sound only
                to the extent those rules are, so `verify.py` reports the
                grounding chain rather than just an accuracy number.
  constructed   the machine drew the data itself and therefore knows what it
                drew. Legitimate supervision for a READER rule (you cannot
                learn to read without knowing what the text said), and useless
                as evidence for anything else -- it can never disagree with the
                renderer.
  measured      came from outside: a camera frame, a physics check, a sim.

`distributive_rewrite` is worth reading closely: it does not assert the
distributive law, it CHECKS each instance numerically before emitting it, which
is what the paper describes the machine doing before it decides the pattern is
worth learning ("it finds that actually the equation holds true").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .dataset import Example, ExampleSet
from .prior import EXP_GAMMA, eval_int_expression, harmonic, sigma
from .tapes import Content

# Quoted so the alias is a ForwardRef rather than a runtime `|` on 3.8/3.9.
OracleFn = Callable[[Example], "Content | None"]


@dataclass
class OracleSpec:
    name: str
    fn: OracleFn
    doc: str
    kind: str                                   # definitional/derived/constructed/measured
    classes: list[str] = field(default_factory=list)   # non-empty => a choice rule


ORACLES: dict[str, OracleSpec] = {}


def oracle(name: str, doc: str, kind: str, classes: Sequence[str] = ()):
    def wrap(fn: OracleFn) -> OracleFn:
        ORACLES[name] = OracleSpec(name, fn, doc, kind, list(classes))
        return fn

    return wrap


def label(example_set: ExampleSet, oracle_name: str) -> ExampleSet:
    """Attach labels in place; unlabelable examples keep out=None and are
    dropped at training time rather than guessed at."""
    if oracle_name not in ORACLES:
        raise KeyError(f"no oracle {oracle_name!r}; known: {sorted(ORACLES)}")
    spec = ORACLES[oracle_name]
    for ex in example_set.examples:
        try:
            ex.out = spec.fn(ex)
        except Exception:
            ex.out = None
        ex.label_source = f"{spec.name}({spec.kind})" if ex.out is not None else ""
    example_set.oracle = oracle_name
    return example_set


def catalogue() -> str:
    rows = []
    for spec in sorted(ORACLES.values(), key=lambda s: s.name):
        extra = f"  classes={spec.classes}" if spec.classes else ""
        rows.append(f"{spec.name} [{spec.kind}]: {spec.doc}{extra}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
@oracle("product_by_definition",
        "a*b evaluated as repeated addition -- the definition, not a rule",
        "definitional")
def _product_by_definition(ex: Example) -> Content | None:
    m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", ex.inp.text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > 2000:
        return None
    total = 0
    for _ in range(a):
        total += b
    return Content.abstract(str(total))


@oracle("arith_value", "the value of an integer expression over + - *", "derived")
def _arith_value(ex: Example) -> Content | None:
    return Content.abstract(str(eval_int_expression(ex.inp.text)))


@oracle("distributive_rewrite",
        "a*b rewritten as a sum of place-value products, each instance checked "
        "numerically before it is emitted",
        "derived")
def _distributive_rewrite(ex: Example) -> Content | None:
    m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", ex.inp.text)
    if not m:
        return None
    a, b = m.group(1), int(m.group(2))
    terms = [int(d) * 10 ** (len(a) - 1 - i) for i, d in enumerate(a)]
    terms = [t for t in terms if t]
    if len(terms) < 2:
        return None
    rhs = "+".join(f"{t}*{b}" for t in terms)
    # The experiment: does this rewrite actually hold for these numbers?
    if eval_int_expression(rhs) != int(a) * b:
        return None
    return Content.abstract(rhs)


@oracle("distributive_rewrite_right",
        "a*b rewritten as a sum over the place values of the RIGHT factor, "
        "each instance checked numerically before it is emitted",
        "derived")
def _distributive_rewrite_right(ex: Example) -> Content | None:
    """The mirror of `distributive_rewrite`, and the reason it exists is worth
    recording: the left-hand rule declines `40*19` and `6*19`, because it
    splits the left factor and those have only one non-zero place there. So a
    decomposition that starts `46*19 -> 40*19+6*19` cannot continue, and the
    parts never reach the 9x9 table. Splitting on the right finishes it.

    Having both also makes a transfer statable rather than merely true: the
    distributive property established for one side, claimed for the other.
    """
    m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", ex.inp.text)
    if not m:
        return None
    a, b = int(m.group(1)), m.group(2)
    terms = [int(d) * 10 ** (len(b) - 1 - i) for i, d in enumerate(b)]
    terms = [t for t in terms if t]
    if len(terms) < 2:
        return None
    rhs = "+".join(f"{a}*{t}" for t in terms)
    # The experiment, exactly as on the left: does this rewrite hold here?
    if eval_int_expression(rhs) != a * int(b):
        return None
    return Content.abstract(rhs)


@oracle("read_back",
        "the text the renderer was asked to draw -- supervision for a reader rule",
        "constructed")
def _read_back(ex: Example) -> Content | None:
    truth = ex.meta.get("truth") or ex.inp.text
    return Content.abstract(truth) if truth else None


# ---------------------------------------------------------------------------
# The Riemann hypothesis, one integer at a time
# ---------------------------------------------------------------------------
# `definitional` is the right kind for both of these and it is worth saying why,
# because it is a stronger claim than any other oracle here makes. These do not
# sample, estimate, or consult a rule: given n they enumerate the divisors of n
# and evaluate an inequality, so their label is correct for that n in the same
# way `mul_by_definition` is correct. That is the paper's "deduction as the
# strict limit of an empirical rule" made literal -- a rule verified at 1.00
# against one of these has been checked exactly, on every instance it was shown.
#
# It also fixes precisely how far that gets. The oracle is exact ABOUT AN
# INSTANCE. Robin's criterion is a statement about all n>5040, and no number of
# exact instance labels closes that gap, which is why the run that uses these
# ends in a transfer claim the machine reports and cannot test.
ROBIN_VERDICTS = ("robin_holds", "robin_fails")
LAGARIAS_VERDICTS = ("lagarias_holds", "lagarias_fails")


def _integer_of(ex: Example) -> int | None:
    truth = ex.meta.get("truth") or ex.inp.text
    return int(truth) if str(truth).strip().isdigit() else None


@oracle("robin_verdict",
        "whether sigma(n) < e^gamma n ln ln n at this n -- Robin's criterion, "
        "which holds for every n>5040 if and only if RH does",
        "definitional", ROBIN_VERDICTS)
def _robin_verdict(ex: Example) -> Content | None:
    n = _integer_of(ex)
    if n is None or n < 3:              # ln ln n <= 0: the criterion is silent
        return None
    ratio = sigma(n) / (n * math.log(math.log(n)))
    return Content.abstract(ROBIN_VERDICTS[0] if ratio < EXP_GAMMA
                            else ROBIN_VERDICTS[1])


@oracle("lagarias_verdict",
        "whether sigma(n) <= H_n + exp(H_n) ln H_n at this n -- Lagarias's "
        "criterion, equivalent to RH and with no exceptional set",
        "definitional", LAGARIAS_VERDICTS)
def _lagarias_verdict(ex: Example) -> Content | None:
    n = _integer_of(ex)
    if n is None or n < 2:
        return None
    h = harmonic(n)
    return Content.abstract(LAGARIAS_VERDICTS[0]
                            if sigma(n) <= h + math.exp(h) * math.log(h)
                            else LAGARIAS_VERDICTS[1])


# ---------------------------------------------------------------------------
# Level spacings
# ---------------------------------------------------------------------------
@oracle("spacing_ensemble",
        "which random-matrix ensemble a spacing histogram was drawn from -- "
        "known because the machine sampled it, and UNDEFINED for the zeta zeros",
        "constructed", ("gue", "goe", "poisson"))
def _spacing_ensemble(ex: Example) -> Content | None:
    """Constructed grounding, with the limit that word carries.

    This oracle is correct about a drawing the machine generated and says
    nothing whatever about a drawing it did not. That is exactly the property
    the zeta experiment needs: the zeta cells have no `ensemble` in their meta,
    so this returns None for them, they stay unlabelled, and no verification
    number can be produced from them by accident.
    """
    kind = ex.meta.get("ensemble") or ex.inp.meta.get("ensemble")
    return Content.abstract(kind) if kind else None


# ---------------------------------------------------------------------------
# Robotics (Appendix A) -- geometry, not learning
# ---------------------------------------------------------------------------
ESCAPE_DIRECTIONS = ("direct", "+x", "-x", "+y", "-y", "+z", "-z")
_AXIS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
         "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def segment_hits_sphere(p0, p1, centre, radius: float) -> bool:
    """Exact segment/sphere test -- the collision check the sketch stands in for."""
    p0, p1, c = np.asarray(p0, float), np.asarray(p1, float), np.asarray(centre, float)
    d = p1 - p0
    n2 = float(d @ d)
    if n2 < 1e-12:
        return bool(np.linalg.norm(p0 - c) <= radius)
    t = float(np.clip((c - p0) @ d / n2, 0.0, 1.0))
    return bool(np.linalg.norm(p0 + t * d - c) <= radius)


def _free(p0, p1, obstacles) -> bool:
    return not any(segment_hits_sphere(p0, p1, o[:3], o[3]) for o in obstacles)


@oracle("best_escape_direction",
        "the escape direction for a sketch: go straight if the path is clear, "
        "otherwise the reachable detour that unblocks the goal and loses least ground",
        "measured", ESCAPE_DIRECTIONS)
def _best_escape_direction(ex: Example) -> Content | None:
    scene = ex.inp.meta.get("scene")
    if not scene or scene.get("kind") != "robot":
        return None
    eef = np.asarray(scene["eef"], float)
    goal = np.asarray(scene["goal"], float)
    obstacles = scene.get("obstacles", ())

    # Priority order copied from detourNet.label_best_detour: an unnecessary
    # detour is a wasted step, so a clear path always wins.
    if _free(eef, goal, obstacles):
        return Content.abstract("direct", label="direct")

    step = 0.35
    reachable, unblocking = [], []
    for name, axis in _AXIS.items():
        wp = eef + step * np.asarray(axis, float)
        if not _free(eef, wp, obstacles):
            continue
        dist = float(np.linalg.norm(goal - wp))
        reachable.append((dist, name))
        if _free(wp, goal, obstacles):
            unblocking.append((dist, name))
    pool = unblocking or reachable
    if not pool:
        return None                       # no valid label: teach nothing, not noise
    best = min(pool)[1]
    return Content.abstract(best, label=best)


# ---------------------------------------------------------------------------
# Geometry (Figure 3)
# ---------------------------------------------------------------------------
CONSTRUCTION_STEPS = ("move_up", "rotate_cw", "rotate_ccw", "done")
ANGLE_FACTS = ("A1=B1,A2=B2,A1+A3+A2=180", "no_facts")

_OFFSET_TOL, _ANGLE_TOL = 0.06, 0.12

# Must match SceneActionCodec's defaults -- these are the sizes of the moves the
# construction actually makes, and the policy below stops by comparing against
# them. A test pins the two together.
_STEP, _ANGLE_STEP = 0.12, 0.15


def _geometry_state(scene: dict) -> tuple[bool, bool, float]:
    """(passes through the apex, parallel to the base, signed angle error)."""
    offset = float(scene.get("line_offset", 0.0))
    err = float(scene.get("line_angle", 0.0)) - float(scene.get("base_angle", 0.0))
    err = (err + np.pi) % (2 * np.pi) - np.pi
    return abs(offset) <= _OFFSET_TOL, abs(err) <= _ANGLE_TOL, err


@oracle("next_construction_step",
        "the next action on the auxiliary line: bring it through the apex first, "
        "then rotate it parallel to the opposite edge",
        "derived", CONSTRUCTION_STEPS)
def _next_construction_step(ex: Example) -> Content | None:
    scene = ex.inp.meta.get("scene")
    if not scene or scene.get("kind") != "geometry":
        return None
    through, parallel, err = _geometry_state(scene)
    if not through:
        step = "move_up"
    elif abs(err) <= _ANGLE_STEP / 2.0:
        # Stop when another rotation would not get CLOSER, not the instant the
        # tolerance is satisfied. The two differ because the step (0.15) is
        # larger than the tolerance (0.12): a construction that halts as soon
        # as it is inside can finish at 0.102 when one more turn would reach
        # -0.048. Nothing downstream is trained on 0.102 -- the generator only
        # ever draws the exact configuration -- and the reader is measured at
        # 1.000 on an exactly parallel line against 0.525 on that one, so the
        # loop was reliably finishing in the one place its own reader could
        # not recognise. Stopping at the nearest reachable point instead of
        # the first acceptable one costs at most one extra rotation.
        step = "done"
    else:
        step = "rotate_cw" if err > 0 else "rotate_ccw"
    return Content.abstract(step, label=step)


@oracle("alternate_angle_facts",
        "the angle equalities the drawing licenses -- only when the auxiliary "
        "line passes through the apex AND is parallel to the opposite edge",
        "derived", ANGLE_FACTS)
def _alternate_angle_facts(ex: Example) -> Content | None:
    scene = ex.inp.meta.get("scene")
    if not scene or scene.get("kind") != "geometry":
        return None
    through, parallel, _ = _geometry_state(scene)
    fact = ANGLE_FACTS[0] if (through and parallel) else ANGLE_FACTS[1]
    return Content.abstract(fact, label=fact)


def oracle_classes(name: str) -> list[str]:
    """Class list for a choice oracle, so a rule can be declared to match it."""
    return list(ORACLES[name].classes)


def oracle_kind(name: str) -> str:
    return ORACLES[name].kind


@oracle(
    "band_exponent_by_quadrature",
    doc="The frequency exponent of a band estimate, measured rather than "
        "computed. Builds a concrete bump concentrated at width 1/M, "
        "integrates its norms numerically at several M, and fits the slope "
        "in log M. This is a genuinely independent route to the same "
        "answer: the rule does exponent arithmetic on a written-down "
        "estimate, the oracle integrates a function on a grid and reads the "
        "exponent off the measurement.",
    kind="measured")
def oracle_band_exponent_by_quadrature(ex: Example) -> Content | None:
    from fractions import Fraction

    from .normcalc import est

    d = int(ex.meta["d"])
    dv = int(ex.meta["dv"])
    ip = Fraction(ex.meta["ip"])
    ip_to = Fraction(ex.meta["ip_to"])
    fr = Fraction(ex.meta["fr"])
    sc = Fraction(ex.meta["sc"])

    # u_M(x) = phi(M x), the extremiser for a band estimate: a bump whose
    # width is the reciprocal of its frequency. Its L^p norms are what
    # carry the exponent, and nothing about them is assumed here.
    span, points = 6.0, 4097
    x = np.linspace(-span, span, points)
    h = x[1] - x[0]
    phi = np.exp(-x ** 2)

    def norm(values: np.ndarray, index: Fraction) -> float:
        """||v||_{L^p} in ONE dimension, with 1/p = index. The d-dimensional
        bump is the product of d copies, so its norm is the d-th power of
        this and the exponent picks up the factor d on its own."""
        if index == 0:                       # the sup norm
            return float(np.max(np.abs(values)))
        p = 1.0 / float(index)
        return float((np.sum(np.abs(values) ** p) * h) ** (1.0 / p))

    ratios, scales = [], [8.0, 16.0, 32.0, 64.0]
    for M in scales:
        # phi(Mx) sampled on the same grid, and its dv-th derivative taken
        # by finite differences rather than by the chain rule, so the M^dv
        # is measured too.
        u = np.exp(-(M * x) ** 2)
        for _ in range(dv):
            u = np.gradient(u, h)
        lo, hi = norm(u, ip), norm(u, ip_to)
        if lo <= 0 or hi <= 0:
            return None
        ratios.append(math.log(hi) - math.log(lo))
    logs = [math.log(M) for M in scales]
    mean_l = sum(logs) / len(logs)
    mean_r = sum(ratios) / len(ratios)
    cov = sum((a - mean_l) * (b - mean_r) for a, b in zip(logs, ratios))
    var = sum((a - mean_l) ** 2 for a in logs)
    if var == 0:
        return None
    slope = cov / var                    # measured, in one dimension

    # d independent factors, so the measured one-dimensional exponent is
    # multiplied by d. The derivative count does NOT appear: it multiplies
    # both norms in the ratio by the same M^dv and cancels. Subtracting a
    # term for it was the first version of this line and it was wrong,
    # which the measurement itself caught.
    measured = slope * d
    snapped = Fraction(round(measured * 4), 4)
    if abs(float(snapped) - measured) > 0.05:
        # A measurement that does not land near an exact rational is not
        # labelled. Snapping anything at all would let the snap supply the
        # answer the oracle is supposed to supply.
        return None
    return Content.abstract(est(d=d, dv=dv, ip=ip_to, fr=fr + snapped, sc=sc))
