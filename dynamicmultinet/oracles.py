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

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .dataset import Example, ExampleSet
from .prior import eval_int_expression
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


@oracle("read_back",
        "the text the renderer was asked to draw -- supervision for a reader rule",
        "constructed")
def _read_back(ex: Example) -> Content | None:
    truth = ex.meta.get("truth") or ex.inp.text
    return Content.abstract(truth) if truth else None


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
    elif parallel:
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
