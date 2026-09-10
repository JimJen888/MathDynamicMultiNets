"""
The derivation itself, as rules that chain.

The other two modules treat the paper's results one at a time: which of them
decide an instance, and what method stands behind the ones that do not. This
one is about the shape of the argument. The construction is not a list of
propositions, it is a sequence of moves on a state -- add an increment to a
field and the residual changes in a known way; run the four corrections and
the residual's decay order goes up -- and moves on a state are exactly what a
rule is here.

Two things become possible once the derivation is written that way.

  A chain can be SEARCHED FOR rather than asserted. One correction cycle is
  four rules, and the machine finds the path from a stage to the next by
  proof search over them, so the derivation is a proof in the machine's own
  sense and not a comment.

  A chain can be KEPT. `keep_proof` turns a found path into one composite
  rule, and the next cycle is then a single move. That is the architecture's
  form of transfer: a derivation done once becomes a rule, and the composite
  inherits trust from its members, so a cycle assembled from unverified steps
  is itself unverified and cannot launder them.

Two things are added here.

  The increment identity of Section 3.3,

      R(u+w, p+pi) = R(u, p) + L_u(w, pi) + div(w tensor w),

  which is what lets the wave construction separate the linear evolution of
  the pulses from the quadratic flux that cancels the background stress. It
  holds because w is divergence-free, and the oracle checks it by building
  fields and differentiating them -- including fields that are not
  divergence-free, where the identity picks up the w div(w) term and fails.

  The correction cycle of Proposition 9.6 as four moves on the tuple of
  decay orders, one per numbered step of its proof. Chaining them is the
  derivation of sigma_{j+1} = sigma_j + 1/10, and the run checks the chain
  against the closed form the recursion is supposed to have.

What this does not do is make the estimates true. Every exponent the cycle
moves through is one, and they are carried here as they are written in the
paper. What is gained is that the argument's shape is now something the
machine holds rather than something a comment describes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

from .dataset import Example
from .generators import generator
from .linarith import Lin, Region, least, obligations, short, solve
from .oracles import oracle
from .provers import Judgement, prover
from .symalg import (Poly, advect, divergence, jacobian, tensor_divergence,
                     vector)
from .rules import PythonRule, Rule
from .tapes import ABSTRACT, Content

DERIVATION_RULES: dict[str, Callable[[], Rule]] = {}


def derivation_rule(name: str):
    def wrap(factory: Callable[[], Rule]) -> Callable[[], Rule]:
        DERIVATION_RULES[name] = factory
        return factory

    return wrap


_NUM = r"-?\d+(?:/\d+)?"


def sf(x: Fraction) -> str:
    return str(Fraction(x))


# ---------------------------------------------------------------------------
# Section 3.3: what adding an increment does to the residual
# ---------------------------------------------------------------------------
@derivation_rule("increment_identity")
def make_increment_identity() -> PythonRule:
    """`increment(divfree=1)` -> does the residual split exactly?

    Everything in Sections 7 and 9 rests on one line:

        R(u+w, p+pi) = R(u, p) + L_u(w, pi) + div(w tensor w),

    with L_u(w, pi) = d_t w + (u.grad)w + (w.grad)u - lap w + grad pi. The
    linear term is what the pulses evolve under and the quadratic term is
    what cancels the background stress, and the whole method is that those
    two can be treated separately because nothing else is left over.

    Nothing else is left over precisely when w is divergence-free. Expanding
    the advection gives (w.grad)w, and turning that into div(w tensor w)
    costs w div(w). The rule says so; the oracle differentiates actual
    fields, with and without the property.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*increment\(divfree=([01]),field=(\d+)\)\s*",
                         c.text.replace(" ", ""))
        if not m:
            return None
        # The field index is the oracle's business: which background and
        # which increment it builds. The identity does not depend on it,
        # and saying so is the claim being tested.
        return Content.abstract(
            "residual_splits_exactly" if m.group(1) == "1"
            else "extra_divergence_term",
            derivation="div(w tensor w) = (w.grad)w + w div(w)")

    return PythonRule("increment_identity", fn, ABSTRACT, ABSTRACT,
                      description="does adding an increment split the residual "
                                  "into linear and quadratic parts",
                      source="increment(divfree,field)->splits|extra_term",
                      exact=False, trusted=False)


def _curl(field, x: np.ndarray, t: float, step: float) -> np.ndarray:
    """Curl by central differences, so the result is divergence-free to the
    accuracy of the stencil and not by construction of the formula."""
    out = np.zeros(3)
    d = []
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        d.append((field(x + shift, t) - field(x - shift, t)) / (2 * step))
    out[0] = d[1][2] - d[2][1]
    out[1] = d[2][0] - d[0][2]
    out[2] = d[0][1] - d[1][0]
    return out


def _residual(velocity, pressure, x: np.ndarray, t: float,
              step: float) -> np.ndarray:
    """d_t v + (v.grad)v - lap v + grad q, all by central differences."""
    v = velocity(x, t)
    d_t = (velocity(x, t + step) - velocity(x, t - step)) / (2 * step)
    advect = np.zeros(3)
    laplace = np.zeros(3)
    grad_q = np.zeros(3)
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        plus, minus = velocity(x + shift, t), velocity(x - shift, t)
        advect += v[axis] * (plus - minus) / (2 * step)
        laplace += (plus - 2 * v + minus) / (step * step)
        grad_q[axis] = (pressure(x + shift, t)
                        - pressure(x - shift, t)) / (2 * step)
    return d_t + advect - laplace + grad_q


@oracle("increment_identity_by_finite_differences",
        "whether the residual of a field plus an increment splits into the "
        "linear and quadratic parts the construction uses, measured on "
        "actual fields",
        "definitional",
        ("residual_splits_exactly", "extra_divergence_term"))
def _increment_identity_by_finite_differences(ex: Example) -> Content | None:
    """The identity of Section 3.3, checked rather than quoted.

    Builds a background velocity as the curl of a smooth potential, an
    increment that is either another curl or the potential itself, and two
    smooth pressures. It then forms all four objects in the identity by
    finite differences and reports the largest disagreement relative to the
    size of the terms. An increment that is not divergence-free leaves
    w div(w) behind, and that is what the second class of instances is for.
    """
    m = re.fullmatch(r"\s*increment\(divfree=([01]),field=(\d+)\)\s*",
                     ex.inp.text.replace(" ", ""))
    if not m:
        return None
    divfree = m.group(1) == "1"
    step = 1e-4

    # A fixed pair of potentials would make every instance the same instance,
    # and a hundred copies of one fact is one fact. The index picks the
    # frequencies, phases and time dependence, so each cell is a different
    # background, a different increment and a different pressure. Frequencies
    # stay in [0.3, 1.8]: below that the fields are nearly constant and the
    # identity holds for want of anything happening, above it the second
    # difference in the Laplacian loses accuracy at this step size and the
    # check would measure the stencil rather than the algebra.
    fr = np.random.default_rng(int(m.group(2)))
    w_a, w_b = fr.uniform(0.3, 1.8, size=(2, 3, 3))
    ph_a, ph_b = fr.uniform(0.0, 2 * math.pi, size=(2, 3, 3))
    ta, tb = fr.uniform(-0.4, 0.4, size=(2, 3))
    wp, php = fr.uniform(0.3, 1.8, size=2), fr.uniform(0.0, 2 * math.pi, size=2)
    wq, phq = fr.uniform(0.3, 1.8, size=2), fr.uniform(0.0, 2 * math.pi, size=2)

    def _potential(w, ph, tt):
        """Every component depends on all three coordinates.

        A component that skips its own coordinate has zero divergence
        whatever its frequencies, which would make the not-divergence-free
        half of the instances divergence-free after all and delete the only
        class of case that can fail. Written out because the first version
        of this function had that bug and passed.
        """
        def f(x, t):
            return np.array([
                math.sin(w[i][0] * x[0] + ph[i][0])
                * math.cos(w[i][1] * x[1] + ph[i][1])
                * math.cos(w[i][2] * x[2] + ph[i][2])
                * (1.0 + tt[i] * t)
                for i in range(3)])

        return f

    potential_a = _potential(w_a, ph_a, ta)
    potential_b = _potential(w_b, ph_b, tb)

    def background(x, t):
        return _curl(potential_a, x, t, step)

    def increment(x, t):
        return _curl(potential_b, x, t, step) if divfree else potential_b(x, t)

    def pressure(x, t):
        return (math.sin(wp[0] * x[0] + php[0])
                * math.cos(wp[1] * x[1] + php[1]) * math.exp(-t / 4))

    def increment_pressure(x, t):
        return math.cos(wq[0] * x[2] + phq[0]) * math.sin(wq[1] * x[1] + phq[1])

    def total(x, t):
        return background(x, t) + increment(x, t)

    def total_pressure(x, t):
        return pressure(x, t) + increment_pressure(x, t)

    worst = 0.0
    for point in ([0.3, -0.4, 0.9], [1.1, 0.6, -0.7], [-0.8, 1.3, 0.2]):
        x, t = np.array(point), 0.35
        left = _residual(total, total_pressure, x, t, step)
        base = _residual(background, pressure, x, t, step)

        # L_u(w, pi): the linear part the pulses evolve under.
        w = increment(x, t)
        linear = (increment(x, t + step) - increment(x, t - step)) / (2 * step)
        linear = linear + np.array([
            (increment_pressure(x + np.eye(3)[a] * step, t)
             - increment_pressure(x - np.eye(3)[a] * step, t)) / (2 * step)
            for a in range(3)])
        u = background(x, t)
        for axis in range(3):
            shift = np.eye(3)[axis] * step
            w_plus, w_minus = increment(x + shift, t), increment(x - shift, t)
            u_plus, u_minus = background(x + shift, t), background(x - shift, t)
            linear += u[axis] * (w_plus - w_minus) / (2 * step)
            linear += w[axis] * (u_plus - u_minus) / (2 * step)
            linear -= (w_plus - 2 * w + w_minus) / (step * step)

        # div(w tensor w), the quadratic flux that carries the stress.
        quadratic = np.zeros(3)
        for axis in range(3):
            shift = np.eye(3)[axis] * step
            w_plus, w_minus = increment(x + shift, t), increment(x - shift, t)
            quadratic += (w_plus * w_plus[axis]
                          - w_minus * w_minus[axis]) / (2 * step)

        gap = left - base - linear - quadratic
        scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(base))),
                    float(np.max(np.abs(linear))), float(np.max(np.abs(quadratic))),
                    1e-12)
        worst = max(worst, float(np.max(np.abs(gap))) / scale)
    return Content.abstract("residual_splits_exactly" if worst < 1e-3
                            else "extra_divergence_term")


@generator("ns_increments",
           "Velocity increments to add to a background field, half of them "
           "divergence-free and half not, for the identity of Section 3.3. "
           "Each instance names a different pair of potentials, so asking "
           "for more instances asks about more fields.",
           {})
def ns_increments(n: int, rng: np.random.Generator):
    return [Example(inp=Content.abstract(
        f"increment(divfree={1 if i % 2 == 0 else 0},field={i})"))
        for i in range(n)]


# ---------------------------------------------------------------------------
# Proposition 9.6 as four moves on the decay orders
# ---------------------------------------------------------------------------
_STATE = ("step", "B0", "C0", "B", "C", "S", "k", "h")


def _read_state(text: str) -> dict[str, Fraction] | None:
    body = text.replace(" ", "")
    if not (body.startswith("state(") and body.endswith(")")):
        return None
    pattern = ",".join(rf"{n}=({_NUM})" for n in _STATE)
    m = re.fullmatch(pattern, body[6:-1])
    if not m:
        return None
    return {n: Fraction(v) for n, v in zip(_STATE, m.groups())}


def _write_state(f: dict[str, Fraction]) -> str:
    return "state(" + ",".join(f"{n}={sf(f[n])}" for n in _STATE) + ")"


def _move(name: str, at_step: int, description: str,
          update: Callable[[dict], dict | None]) -> PythonRule:
    """One numbered step of the cycle, as a move on the tuple of orders."""

    def fn(c: Content) -> Content | None:
        f = _read_state(c.text)
        if f is None or f["step"] != at_step:
            return None
        out = update(dict(f))
        return None if out is None else Content.abstract(
            _write_state(out), derivation=description)

    rule = PythonRule(name, fn, ABSTRACT, ABSTRACT, description=description,
                      source=f"state(step={at_step})->state",
                      exact=False, trusted=False)
    # The prover runs THIS function, not a transcription of it. Proving a
    # copy of a rule proves nothing about the rule.
    rule.update = update
    rule.at_step = at_step
    return rule


@derivation_rule("cycle_op1_harmonics")
def make_cycle_op1() -> PythonRule:
    """Step 1: solve the amplitude equation for every supported nonzero
    harmonic. The wave order improves to the smallest of the four errors
    that leaves -- the linear one, the cross terms with the waves already
    there and with the mean, and the new increment's self-interaction --
    while the mean and the three defects fall back to one radial derivative
    below the stage's mean order."""

    def update(f):
        k, B0, C0 = f["k"], f["B0"], f["C0"]
        # The stage's wave residual is cancelled outright, so the order that
        # replaces it is the smallest of the errors that cancelling leaves.
        # Taking a minimum with the old order would keep a term that is no
        # longer there and the cycle would never gain anything.
        f["B"] = least(B0 + Fraction(1, 2) - 3 * k, B0 + Fraction(1, 2) - k,
                       2 * B0 - k, B0 + Fraction(2, 5))
        f["C"] = C0 - k
        f["S"] = C0 - k
        f["step"] = Fraction(1)
        return f

    return _move("cycle_op1_harmonics", 0,
                 "Proposition 9.6 step 1: cancel the supported harmonics",
                 update)


@derivation_rule("cycle_op2_stress")
def make_cycle_op2() -> PythonRule:
    """Step 2: change the wave amplitudes so their averaged flux corrects the
    auxiliary-averaged tangential residual. The signed increments are built
    on the fixed positive leading amplitudes, so they may point either way;
    what they cost is another pass of the same four error types, and what
    they leave in the mean is the covariance expansion of (9.13)."""

    def update(f):
        k, B0, C0 = f["k"], f["B0"], f["C0"]
        f["B"] = least(f["B"], B0 + Fraction(1, 2) - 4 * k,
                       B0 + Fraction(2, 5) - k, B0 + Fraction(1, 2) - 2 * k,
                       2 * B0 - 3 * k)
        f["C"] = C0 - 2 * k
        f["S"] = C0 - 2 * k
        f["step"] = Fraction(2)
        return f

    return _move("cycle_op2_stress", 1,
                 "Proposition 9.6 step 2: correct the averaged stress",
                 update)


@derivation_rule("cycle_op3_mean")
def make_cycle_op3() -> PythonRule:
    """Step 3: remove the part of the mean residual with zero auxiliary
    average by inverting the fast time derivative. What survives is the
    averaged part, already improved in step 2, against the slow-time and
    viscous terms the increment itself creates."""

    def update(f):
        k, C0 = f["k"], f["C0"]
        f["C"] = least(C0 + Fraction(17, 100), (C0 - 2 * k) + 1 - 2 * k)
        f["B"] = least(f["B"], C0 - 2 * k)
        f["step"] = Fraction(3)
        return f

    return _move("cycle_op3_mean", 2,
                 "Proposition 9.6 step 3: remove the nonconstant auxiliary mean",
                 update)


@derivation_rule("cycle_op4_moments")
def make_cycle_op4() -> PythonRule:
    """Step 4: solve the five radial moment equations, which cancels the
    linear part of the three compatibility defects while preserving the two
    velocity moments, and closes the cycle.

    This is the move that declines. The next stage is only entitled to the
    orders B+1/10 and C+1/10 if all three of the cycle's gains reached them,
    and with a large enough radial derivative loss they do not. A search for
    the derivation then finds no path, which is the honest outcome: the
    construction has a budget and the machine can be shown running out of
    it.
    """

    def update(f):
        k, B0, C0 = f["k"], f["B0"], f["C0"]
        f["S"] = (C0 - 2 * k) + Fraction(9, 10) - 2 * k
        step_up = Fraction(1, 10)
        # `short` is `<` on numbers. Handed linear forms in the stage index
        # it records the guard and answers False, so the prover gets the
        # cycle's conclusion and the hypothesis it rests on separately --
        # see linarith.
        if (short(f["B"], B0 + step_up, "wave order gains 1/10")
                or short(f["C"], C0 + step_up, "mean order gains 1/10")
                or short(f["S"], C0 + step_up, "swirl order gains 1/10")):
            return None
        f["B0"] = B0 + step_up
        f["C0"] = C0 + step_up
        f["B"], f["C"], f["S"] = f["B0"], f["C0"], f["C0"]
        f["step"] = Fraction(0)
        return f

    return _move("cycle_op4_moments", 3,
                 "Proposition 9.6 step 4: restore the compatibility defects",
                 update)


def initial_state(kappa: Fraction = Fraction(1, 100_000),
                  h: Fraction = Fraction(1, 200)) -> str:
    """Stage zero, as Proposition 9.5 leaves it: B = 7/10, C* = 6/5."""
    return _write_state({"step": Fraction(0), "B0": Fraction(7, 10),
                         "C0": Fraction(6, 5), "B": Fraction(7, 10),
                         "C": Fraction(6, 5), "S": Fraction(6, 5),
                         "k": kappa, "h": h})


def state_after(cycles: int, kappa: Fraction = Fraction(1, 100_000),
                h: Fraction = Fraction(1, 200)) -> str:
    """Where the four moves land after a whole number of cycles."""
    gain = Fraction(cycles, 10)
    return _write_state({"step": Fraction(0), "B0": Fraction(7, 10) + gain,
                         "C0": Fraction(6, 5) + gain,
                         "B": Fraction(7, 10) + gain,
                         "C": Fraction(6, 5) + gain,
                         "S": Fraction(6, 5) + gain, "k": kappa, "h": h})


CYCLE_MOVES = ("cycle_op1_harmonics", "cycle_op2_stress",
               "cycle_op3_mean", "cycle_op4_moments")


@oracle("cycle_state_by_closed_form",
        "where a whole cycle should leave the decay orders, from the closed "
        "form of the recursion rather than from the four steps",
        "derived")
def _cycle_state_by_closed_form(ex: Example) -> Content | None:
    """The four moves, checked against sigma_j = 1/5 + j/10.

    Proposition 9.6 gains 1/10 per cycle and (9.8) states the closed form
    that follows. This oracle reads a state, works out which stage it is at
    from B = 1/2 + sigma, and returns the state one stage on according to
    the closed form. Chaining the four moves has to land on the same cell.

    It is a consistency check between two encodings of the same recursion,
    not evidence that either is right: every exponent the moves pass through
    is an estimate the paper proves elsewhere and this machine imports.
    """
    f = _read_state(ex.inp.text)
    if f is None or f["step"] != 0:
        return None
    sigma = f["B"] - Fraction(1, 2)
    if sigma < 0 or f["C"] != 1 + sigma or f["S"] != 1 + sigma:
        return None
    # Below this loss the four gains all clear 1/10; above it the cycle does
    # not close and the fourth move declines, so there is no next state.
    if f["k"] > Fraction(1, 30):
        return None
    nxt = Fraction(1, 10)
    return Content.abstract(_write_state({
        "step": Fraction(0), "B0": f["B0"] + nxt, "C0": f["C0"] + nxt,
        "B": f["B"] + nxt, "C": f["C"] + nxt, "S": f["S"] + nxt,
        "k": f["k"], "h": f["h"]}))


@generator("ns_cycle_states",
           "Correction-cycle states at a range of stages and radial "
           "derivative losses, for chaining the four moves of Proposition "
           "9.6.",
           {})
def ns_cycle_states(n: int, rng: np.random.Generator):
    out: list[Example] = []
    for _ in range(n):
        k = Fraction(1, int(rng.integers(31, 200_000)))
        j = int(rng.integers(0, 30))
        h = Fraction(1, int(rng.integers(101, 900)))
        out.append(Example(inp=Content.abstract(state_after(j, k, h))))
    return out


def _tagged(text: str, tag: str) -> Fraction | None:
    m = re.fullmatch(rf"\s*{re.escape(tag)},h=({_NUM})\s*", text.replace(" ", ""))
    return Fraction(m.group(1)) if m else None


@derivation_rule("apply_wave_increment")
def make_apply_wave_increment() -> PythonRule:
    """`background_annular_stress,h=H` -> `increment_split,h=H`.

    The move Section 3.3 opens with: add the divergence-free wave increment
    to the background and read the residual as three separate things -- what
    was there, what the pulses evolve under, and the quadratic flux. It is
    bookkeeping, and it is trusted here for one reason: the identity it
    encodes is `increment_identity`, which is checked against fields built
    and differentiated, on increments that are divergence-free and on
    increments that are not.
    """

    def fn(c: Content) -> Content | None:
        h = _tagged(c.text, "background_annular_stress")
        if h is None:
            return None
        return Content.abstract(f"increment_split,h={sf(h)}",
                                derivation="R(u+w) = R(u) + L(w) + div(w x w)")

    return PythonRule("apply_wave_increment", fn, ABSTRACT, ABSTRACT,
                      description="add the wave increment and split the residual",
                      source="background_annular_stress->increment_split")


@derivation_rule("prop_9_5_initialize")
def make_prop_9_5_initialize() -> PythonRule:
    """`stress_realized_by_waves,h=H` -> the stage-zero decay orders.

    Proposition 9.5: after the primary pulses, the pressure reconstruction
    and the first two mean corrections, the state satisfies the bounds the
    correction cycle needs, at B = 7/10 and C* = 6/5. Imported: those are
    estimates, and what this machine can do with them starts one line later.
    """

    def fn(c: Content) -> Content | None:
        h = _tagged(c.text, "stress_realized_by_waves")
        if h is None:
            return None
        return Content.abstract(initial_state(h=h),
                                derivation="imported: Proposition 9.5")

    return PythonRule("prop_9_5_initialize", fn, ABSTRACT, ABSTRACT,
                      description="Proposition 9.5: the state the correction "
                                  "cycle starts from",
                      source="stress_realized_by_waves->state",
                      exact=False, trusted=False)


@derivation_rule("prop_9_6_induction")
def make_prop_9_6_induction() -> PythonRule:
    """A state at any stage -> `residual_flat_at_singularity,h=H`.

    The cycle can be run, and the machine can run it: that is what the four
    moves and the composite above are. What cannot be run is the induction
    -- every finite stage, with the supports, the velocity bounds and the
    two moment constraints preserved at each -- and it is the induction that
    turns a gain per cycle into a residual flat at the singular time.

    Declines below stage one, so that a derivation has to go through at
    least one cycle rather than around it.
    """

    def fn(c: Content) -> Content | None:
        f = _read_state(c.text)
        if f is None or f["step"] != 0 or f["B0"] < Fraction(8, 10):
            return None
        return Content.abstract(f"residual_flat_at_singularity,h={sf(f['h'])}",
                                derivation="imported: the induction over stages")

    return PythonRule("prop_9_6_induction", fn, ABSTRACT, ABSTRACT,
                      description="every finite stage, hence a flat residual",
                      source="state->residual_flat_at_singularity",
                      exact=False, trusted=False)


# ---------------------------------------------------------------------------
# The induction over stages, proved rather than sampled
# ---------------------------------------------------------------------------
@dataclass
class StageProof:
    """What running the cycle on a symbolic stage index established.

    `budget` is the set of radial derivative losses for which the cycle
    closes at every stage. It is derived, not quoted: the loss is left as a
    variable alongside the stage index, and the interval falls out of the
    same inequalities.
    """

    budget: Region
    landed: bool                        # did the four moves land on stage n+1
    base_ok: bool                       # is stage 0 the state Prop 9.5 leaves

    @property
    def sound(self) -> bool:
        return self.landed and self.base_ok and self.budget.feasible

    def closes_at(self, kappa: Fraction) -> bool:
        return self.sound and self.budget.contains(kappa)

    def summary(self) -> str:
        if not self.base_ok:
            return "the symbolic stage 0 is not the state Proposition 9.5 leaves"
        if not self.landed:
            return "the four moves do not land on the closed form of stage n+1"
        if not self.budget.feasible:
            return ("no radial derivative loss lets the cycle close at every "
                    "stage: " + "; ".join(self.budget.blocking[:2]))
        return (f"PROVED for every stage n >= 0 and every kappa with "
                f"{self.budget.describe()}: {self.budget.decided} "
                f"inequalities, all decided by inspection")


def _symbolic_stage(kappa: Fraction, h: Fraction) -> dict:
    """Stage n of the recursion, with n left as a variable.

    B and C start each cycle at the orders stage n is entitled to, which is
    (9.8): sigma_n = 1/5 + n/10, B = 1/2 + sigma_n, C = S = 1 + sigma_n.
    Written as linear forms, those are the only place n enters.
    """
    stage = Lin.var("n", Fraction(1, 10))
    b0 = Fraction(7, 10) + stage
    c0 = Fraction(6, 5) + stage
    return {"step": Fraction(0), "B0": b0, "C0": c0, "B": b0, "C": c0,
            "S": c0, "k": kappa, "h": h}


def prove_cycle_closes_at_every_stage(
        library, h: Fraction = Fraction(1, 200)) -> StageProof:
    """Run the four moves once, on a stage index that is a variable.

    This is the one quantifier in the construction that a machine like this
    can discharge. "Every correction stage n" ranges over the non-negative
    integers, every order the cycle touches is linear in n with rational
    coefficients, and a universally quantified linear inequality over one
    integer variable is decidable. So the cycle is run ONCE, symbolically,
    and what comes out is a statement about all stages rather than about the
    stages someone thought to sample.

    The radial derivative loss is left as a variable too, so what comes back
    is not a yes for one loss but the interval of losses that work. The
    construction's budget is then something the machine derives rather than
    a constant copied out of the paper.

    What it establishes, exactly: given the per-stage estimates the four
    moves encode, chaining them carries stage n to stage n+1 for every
    n >= 0 and every loss in that interval. What it does not establish:
    those estimates. They are Theorem 4.6, Propositions 5.5 and 7.5 and the
    rest, they are imported, and no amount of arithmetic over the stage
    index touches them.
    """
    moves = [library.get(name) for name in CYCLE_MOVES]
    if any(not hasattr(m, "update") for m in moves):
        raise ValueError("the cycle moves do not expose the function the "
                         "rules run; the proof would be about a copy")

    kappa = Lin.var("kappa")
    state = _symbolic_stage(kappa, h)
    ground = {k: (v.at(n=0, kappa=0).value() if isinstance(v, Lin) else v)
              for k, v in state.items()}
    base_ok = _write_state(ground) == initial_state(Fraction(0), h)

    with obligations() as collected:
        for move in moves:
            state = move.update(dict(state))
            if state is None:                       # pragma: no cover
                raise AssertionError("a move declined a symbolic stage; guards "
                                     "are supposed to be recorded, not decided")

    # Where the four moves left the orders, against stage n+1 of (9.8).
    want = _symbolic_stage(kappa, h)
    landed = all(state[key] == want[key] + Fraction(1, 10)
                 for key in ("B0", "C0", "B", "C", "S"))
    return StageProof(budget=solve(collected, parameter="kappa"),
                      landed=landed, base_ok=base_ok)


@prover("increment_identity_by_polynomial_algebra",
        "decides the residual splitting of Section 3.3 by expanding both "
        "sides as polynomials in the one-jet, which settles it at every "
        "point of every smooth field at once")
def _prove_increment_identity(library, rule) -> Judgement:
    """Section 3.3, done as the derivation rather than measured on fields.

    The identity is

        R(u+w, p+pi) = R(u, p) + L_u(w, pi) + div(w tensor w),

    and its whole content is what happens to the advection term, because
    d_t and the Laplacian and the pressure gradient are linear and pass
    through the sum without comment. Expanding the advection,

        (u+w).grad(u+w) = u.grad u + u.grad w + w.grad u + w.grad w,

    and the last of those is the quadratic flux only up to one term:

        div(w tensor w)_i = w_j d_j w_i + w_i div w.

    So the two sides differ by exactly -w_i div w, which vanishes precisely
    when the increment is divergence-free. That is the derivation, it is
    four lines, and every step of it is algebra among the values of the
    fields and their first derivatives at a point.

    Which is why it can be settled here rather than sampled. Take those
    values as independent symbols -- nothing in a pointwise identity knows
    that `dw0_1` came from differentiating anything -- expand both sides as
    polynomials, and subtract. The oracle that measured this on 1200
    randomly generated fields was doing the right thing and getting a
    weaker answer.
    """
    dims = 3
    u, w = vector("u"), vector("w")
    du, dw = jacobian("u"), jacobian("w")
    # The linear pieces enter as symbols because they ARE linear: the
    # identity does not turn on them and assuming d_t(u+w) = d_t u + d_t w
    # assumes nothing that is in question.
    dtu, dtw = vector("dtu"), vector("dtw")
    lapu, lapw = vector("lapu"), vector("lapw")
    gp, gpi = vector("gp"), vector("gpi")

    total = [u[i] + w[i] for i in range(dims)]
    dtotal = [[du[i][j] + dw[i][j] for j in range(dims)] for i in range(dims)]

    left = [dtu[i] + dtw[i] + advect(total, dtotal)[i]
            - lapu[i] - lapw[i] + gp[i] + gpi[i] for i in range(dims)]
    base = [dtu[i] + advect(u, du)[i] - lapu[i] + gp[i] for i in range(dims)]
    linear = [dtw[i] + advect(u, dw)[i] + advect(w, du)[i]
              - lapw[i] + gpi[i] for i in range(dims)]
    quadratic = tensor_divergence(w, dw)

    defect = [left[i] - base[i] - linear[i] - quadratic[i] for i in range(dims)]
    want = [Poly.constant(-1) * w[i] * divergence(dw) for i in range(dims)]

    if any(not (defect[i] - want[i]).is_zero for i in range(dims)):
        return Judgement(False, obstruction=(
            "the two sides do not differ by w div(w); the expansion leaves "
            + str(defect[0] - want[0])))
    if all(d.is_zero for d in defect):
        return Judgement(False, obstruction=(
            "the expansion makes the identity unconditional, which would "
            "mean the divergence-free hypothesis is doing no work and the "
            "algebra has been set up wrong"))

    # The rule is a predicate on the flag, so deciding it is two questions.
    for flag, expected in ((1, "residual_splits_exactly"),
                           (0, "extra_divergence_term")):
        answer = rule.apply(Content.abstract(
            f"increment(divfree={flag},field=0)"))
        if answer is None or answer.text != expected:
            return Judgement(False, obstruction=(
                f"the rule answers {answer.text if answer else 'nothing'} for "
                f"divfree={flag}, and the algebra says {expected}"))

    return Judgement(
        True,
        statement=("R(u+w, p+pi) - R(u, p) - L_u(w, pi) - div(w tensor w) "
                   "= -w div(w) identically, so the residual splits exactly "
                   "when the increment is divergence-free and not otherwise"),
        covers=("every point of every smooth field, both values of the "
                "divergence-free flag, which is the rule's whole domain"),
        whole_domain=True,
        detail=[
            "expanded in the one-jet: u_i, w_i and d_j u_i, d_j w_i taken as "
            "independent symbols, so no field is ever chosen",
            "the defect is exactly -w_i div(w), in closed form, rather than "
            "a residual that happened to be small",
            "d_t, the Laplacian and the pressure gradient enter as symbols "
            "because they are linear and the identity does not turn on them",
            "the finite-difference oracle still runs, on 1200 fields; it now "
            "confirms a theorem rather than standing in for one",
        ])


@prover("stage_induction_by_linear_arithmetic",
        "decides the correction cycle's induction over stages by running its "
        "four moves once with the stage index and the radial derivative loss "
        "both left as variables")
def _prove_stage_induction(library, rule) -> Judgement:
    """Settle `prop_9_6_all_stages` everywhere at once.

    The rule answers a question about every stage for a given loss. This
    runs the cycle with BOTH left symbolic, so what comes back is the exact
    interval of losses for which the recursion never stops closing -- and
    comparing that interval with the rule's own threshold decides the rule
    on every input it accepts, not on a sample of them.
    """
    proof = prove_cycle_closes_at_every_stage(library)
    if not proof.base_ok:
        return Judgement(False, obstruction=(
            "the symbolic stage 0 is not the state Proposition 9.5 leaves, so "
            "the induction is not about this construction"))
    if not proof.landed:
        return Judgement(False, obstruction=(
            "the four moves do not land on the closed form of stage n+1, so "
            "there is no recursion to induct over"))
    if not proof.budget.feasible:
        return Judgement(False, obstruction=(
            "no radial derivative loss lets the cycle close at every stage: "
            + "; ".join(proof.budget.blocking[:2])))

    # Does the rule's own threshold agree with the derived interval? The
    # rule is a predicate on kappa, and the prover has the true predicate,
    # so this is a comparison of two sets rather than a spot check.
    derived_upper = proof.budget.upper
    disagreements = []
    for kappa in _threshold_probes(derived_upper):
        answer = rule.apply(Content.abstract(
            f"cycle_from(stage=0,kappa={sf(kappa)})"))
        want = ("cycle_closes_at_every_stage" if proof.closes_at(kappa)
                else "cycle_runs_out_of_budget")
        if answer is None or answer.text != want:
            disagreements.append(f"kappa={sf(kappa)}: rule says "
                                 f"{answer.text if answer else 'nothing'}, "
                                 f"the procedure says {want}")
    if disagreements:
        return Judgement(False, obstruction=(
            "the rule's threshold is not the one the cycle actually has: "
            + "; ".join(disagreements[:3])))

    return Judgement(
        True,
        statement=("the correction cycle of Proposition 9.6 carries stage n "
                   "to stage n+1 for every integer n >= 0, exactly when the "
                   f"radial derivative loss satisfies {proof.budget.describe()}"),
        covers=("every non-negative integer stage and every positive rational "
                "loss, which is the rule's whole domain"),
        whole_domain=True,
        detail=[
            f"{proof.budget.decided} linear inequalities, each decided by "
            f"reading its coefficients rather than by trying values",
            f"the binding one is step 2's stress correction, which costs four "
            f"radial derivatives against the 1/10 the stage must gain",
            f"derived budget: {proof.budget.describe()}; the construction "
            f"uses kappa = 1/100000, smaller by four orders of magnitude",
            "the estimates the four moves encode are NOT in this: they are "
            "Theorem 4.6, Propositions 5.5 and 7.5 and the rest, and they "
            "stay imported",
        ])


def _threshold_probes(upper: Fraction | None) -> list[Fraction]:
    """Losses that pin down a threshold: on it, either side, and far away.

    A predicate on the rationals cannot be compared to another predicate by
    enumeration, but two predicates that are each a half-line agree
    everywhere exactly when they agree at the boundary and on both sides of
    it. These are those points, plus a few decades out to catch a rule
    whose threshold is right but whose shape is not.
    """
    if upper is None:
        return [Fraction(1, 10 ** k) for k in range(0, 7)]
    out = [upper, upper + Fraction(1, 10 ** 9), upper - Fraction(1, 10 ** 9)]
    out += [upper / 10 ** k for k in range(1, 6)]
    out += [upper * 10 ** k for k in range(1, 6)]
    return [q for q in out if q > 0]


@derivation_rule("prop_9_6_all_stages")
def make_prop_9_6_all_stages() -> PythonRule:
    """`cycle_from(stage=0,kappa=1/100000)` -> does the recursion close forever?

    Proposition 9.6 is an induction: one cycle gains 1/10 in the decay
    order, so sigma_j = 1/5 + j/10 for every j. The run checks one cycle on
    instances and can check ten, and neither reaches "every j" -- which is
    why `prop_9_6_induction` sits in the library as an imported step.

    This rule is the induction itself, and unlike the estimates inside the
    cycle it is a claim about rational arithmetic over an integer index,
    which is decidable. The rule states the verdict; the prover in this
    module decides it by running the four moves once with the stage index
    left as a variable.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(rf"\s*cycle_from\(stage=(\d+),kappa=({_NUM})\)\s*",
                         c.text.replace(" ", ""))
        if not m:
            return None
        # The claim, and where its threshold comes from. Every gain in the
        # cycle clears 1/10 with room to spare except one: step 2 corrects
        # the averaged stress at a cost of four radial derivatives, leaving
        # the wave order at B + 1/2 - 4*kappa, and the stage is only
        # entitled to B + 1/10 if 4*kappa <= 2/5. So the cycle runs forever
        # exactly while kappa <= 1/10, and the construction's kappa is
        # smaller than that by four orders of magnitude.
        kappa = Fraction(m.group(2))
        if kappa <= 0:
            return None
        closes = kappa <= Fraction(1, 10)
        return Content.abstract(
            "cycle_closes_at_every_stage" if closes
            else "cycle_runs_out_of_budget",
            derivation="sigma_{j+1} = sigma_j + 1/10 for every j")

    return PythonRule("prop_9_6_all_stages", fn, ABSTRACT, ABSTRACT,
                      description="does the correction cycle keep gaining "
                                  "1/10 at every stage without end",
                      source="cycle_from(stage,kappa)->closes|runs_out",
                      exact=False, trusted=False)


#: The imported steps that live in this module.
DERIVATION_IMPORTED = ("prop_9_5_initialize", "prop_9_6_induction")


#: The derivation, as the cells it passes through. The example run proves
#: each leg by search rather than asserting the sequence.
DERIVATION_LEGS = (
    ("h=1/200", "bounded_energy_unbounded_velocity,h=1/200",
     "the exponents and the energy budget"),
    ("bounded_energy_unbounded_velocity,h=1/200", "increment_split,h=1/200",
     "profiles, background, and the wave increment"),
    ("increment_split,h=1/200", "residual_flat_at_singularity,h=1/200",
     "the stress carried by the pulses, then one correction cycle"),
    ("residual_flat_at_singularity,h=1/200", "theorem_1_1_forced_blowup",
     "summation, localization, force, energy and comparison"),
)


def install_derivation_rules(library) -> None:
    for factory in DERIVATION_RULES.values():
        library.add(factory(), replace=True)


def _mutant_increment(c: Content) -> Content | None:
    """The identity asserted whatever the increment does.

    Section 3.3 is only true because w is divergence-free; a rule that
    dropped that condition would still be right on half the instances,
    which is exactly what the oracle has to notice.
    """
    if not c.text.replace(" ", "").startswith("increment("):
        return None
    return Content.abstract("residual_splits_exactly")


#: The mechanisms these moves and this identity speak for.
DERIVATION_CHECKS = {
    "increment_identity": ("increment_identity_by_finite_differences",
                           "ns_increments"),
}

DERIVATION_MUTATIONS = {
    "increment_identity": ("divergence-free condition dropped",
                           _mutant_increment),
}
