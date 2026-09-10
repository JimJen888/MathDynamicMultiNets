"""
The imported estimates, with their methods executed.

`navierstokes.py` splits the construction into results that decide an instance
and results that are bounds on function spaces, and leaves the second group
untrusted. This module attacks that group. It does not prove any of them --
each asserts something uniform in the concentration scale, the dyadic band and
the correction stage, and no computation reaches a statement of that shape.
What it does is take the METHOD the paper uses for each and run it, so that
the step stops being a citation and becomes a mechanism the machine has
executed and measured.

  paper                     mechanism executed here
  ------------------------  -------------------------------------------------
  Proposition 9.6           the exponent table of one correction cycle,
                            enumerated term by term, against the four
                            closing inequalities the proof states
  Lemma 5.4, and with it    the cutoff scales chosen by the paper's own
  Propositions 5.5 and 9.9  recursion, and the resulting tail summed and
                            measured for flatness
  Lemmas 10.2 and 10.3      the Borel-type extension of the force past t=1
                            built from its derivative limits, with the
                            shrinking time cutoffs, and differentiated
  Lemma 10.4                the energy identity and its bound, on a
                            finite-dimensional model with the same structure:
                            a dissipative part, a skew nonlinearity that
                            cancels, and a forcing term
  Proposition 10.1          the localization, done to the vector potential
                            before the curl, with the divergence measured
  Proposition 7.5           the covariance of two pulse families, integrated
                            from the amplitude equation rather than from the
                            frozen leading form, and solved for positive
                            squared amplitudes

Each mechanism is a rule that states the cheap prediction and an oracle that
carries the construction out. Every generator here emits instances on both
sides of the question -- a cutoff schedule that works and one that does not, a
nonlinearity that cancels in the energy identity and one that does not, a
localization applied before the curl and one applied after -- because a check
whose answer never changes measures nothing.

What is still missing after all of this is stated in `REMAINING` at the end of
the module, and printed by the example run.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Callable

import numpy as np

from .dataset import Example
from .generators import generator
from .oracles import oracle
from .rules import PythonRule, Rule
from .tapes import ABSTRACT, Content

MECHANISM_RULES: dict[str, Callable[[], Rule]] = {}


def mechanism(name: str):
    def wrap(factory: Callable[[], Rule]) -> Callable[[], Rule]:
        MECHANISM_RULES[name] = factory
        return factory

    return wrap


_NUM = r"-?\d+(?:/\d+)?"
_WORD = r"[a-z_]+"


def _parse(text: str, head: str, spec: str) -> dict[str, object] | None:
    """Parse `head(a=1/2,policy=recursive)` against a field spec.

    `spec` is a comma-separated list of `name:kind`, kind being `q` for a
    rational or `w` for a word. Order is fixed, which keeps the cells on the
    tape canonical and the rules cheap to match.
    """
    body = text.replace(" ", "")
    if not (body.startswith(head + "(") and body.endswith(")")):
        return None
    fields = [item.split(":") for item in spec.split(",")]
    pattern = ",".join(
        rf"{re.escape(name)}=({_NUM if kind == 'q' else _WORD})"
        for name, kind in fields)
    m = re.fullmatch(pattern, body[len(head) + 1:-1])
    if not m:
        return None
    return {name: (Fraction(value) if kind == "q" else value)
            for (name, kind), value in zip(fields, m.groups())}


def sf(x: Fraction) -> str:
    return str(Fraction(x))


# ---------------------------------------------------------------------------
# Proposition 9.6: the exponent table of one correction cycle
# ---------------------------------------------------------------------------
def _cycle_terms(kappa: float, sigma: float) -> dict[str, float]:
    """Every exponent the proof of Proposition 9.6 produces in one cycle.

    Transcribed from the four numbered steps: the harmonic errors left by
    cancelling the supported harmonics and by the signed amplitude change,
    the tensor terms of the covariance expansion (9.13) after their radial
    divergence, and the defect order left by the five-equation correction.
    """
    B = 0.5 + sigma                       # current wave order
    C = 1.0 + sigma                       # current mean order
    k = kappa
    waves = {
        "step1 linear error": B + 0.5 - 3 * k,
        "step1 cross with old waves": B + 0.5 - k,
        "step1 self-interaction": 2 * B - k,
        "step1 cross with the mean": B + 0.4,
        "step2 linear error": B + 0.5 - 4 * k,
        "step2 mean interaction": B + 0.4 - k,
        "step2 cross with old waves": B + 0.5 - 2 * k,
        "step2 self-interaction": 2 * B - 3 * k,
    }
    # (9.13), each losing one kappa to the radial divergence, plus the
    # retained moment term and the complementary part corrected in step 3.
    H = C - 2 * k
    means = {
        "transverse correction x old remainder": C + 0.18 - 2 * k,
        "signed curl x old wave": C + 0.5 - 3 * k,
        "signed self-interaction": C + sigma - 3 * k,
        "retained moment term": C + 1 - k,
        "complementary part after step 3": H + 1 - 2 * k,
    }
    defects = {"five-equation remainder": H + 0.9 - 2 * k}
    return {"B": B, "C": C,
            "wave": min(waves.values()),
            "mean": min(means.values()),
            "defect": min(defects.values())}


@mechanism("prop_9_6_table")
def make_prop_9_6_table() -> PythonRule:
    """`cycle(kappa=1/100000,sigma=1/5)` -> does one cycle gain 1/10?

    The rule is the paper's own summary: the cycle closes if the four groups
    of inequalities at the end of Proposition 9.6 hold,

        min{1/2-3k, 1/2-k, B-k, 2/5}     >= 2/5
        min{1/2-4k, 2/5-k, 1/2-2k, B-3k} >= 2/5 - k
        1/2 - 2k > 1/10  and  17/100 > 1/10
        9/10 - 4k > 1/10

    That is four comparisons. What it summarises is a table of thirteen
    exponents spread over four pages, and the oracle rebuilds that table and
    takes its minima. Agreement means the summary and the body of the proof
    describe the same cycle, which is the kind of thing that goes wrong in a
    long bookkeeping argument and cannot go wrong twice the same way.

    The two are not equivalent, and the generator respects that: the closing
    inequalities are sufficient, not necessary, so they fail before the table
    does. Instances are drawn where both are in their intended regime,
    kappa <= 1/50, and the gap is stated rather than tuned away.
    """

    def fn(c: Content) -> Content | None:
        f = _parse(c.text, "cycle", "kappa:q,sigma:q")
        if f is None:
            return None
        k, sigma = f["kappa"], f["sigma"]
        if k < 0 or sigma <= 0:
            return None
        B = Fraction(1, 2) + sigma
        half, two5 = Fraction(1, 2), Fraction(2, 5)
        groups = (
            min(half - 3 * k, half - k, B - k, two5) >= two5,
            min(half - 4 * k, two5 - k, half - 2 * k, B - 3 * k) >= two5 - k,
            half - 2 * k > Fraction(1, 10)
            and min(Fraction(17, 100), 1 - 4 * k) > Fraction(1, 10),
            Fraction(9, 10) - 4 * k > Fraction(1, 10),
        )
        return Content.abstract(
            "wave_and_mean_gain_at_least_1/10" if all(groups)
            else "cycle_gain_short",
            derivation="the four closing inequalities of Proposition 9.6")

    return PythonRule("prop_9_6_table", fn, ABSTRACT, ABSTRACT,
                      description="one correction cycle gains 1/10, by the "
                                  "closing inequalities",
                      source="cycle(kappa,sigma)->gain",
                      exact=False, trusted=False)


@oracle("cycle_table_by_enumeration",
        "the gain of one correction cycle, taken as the minimum over every "
        "exponent the proof of Proposition 9.6 produces",
        "derived")
def _cycle_table_by_enumeration(ex: Example) -> Content | None:
    f = _parse(ex.inp.text, "cycle", "kappa:q,sigma:q")
    if f is None:
        return None
    k, sigma = float(f["kappa"]), float(f["sigma"])
    if k < 0 or sigma <= 0:
        return None
    t = _cycle_terms(k, sigma)
    gained = min(t["wave"] - t["B"], t["mean"] - t["C"], t["defect"] - t["C"])
    return Content.abstract("wave_and_mean_gain_at_least_1/10"
                            if gained >= 0.1 - 1e-12 else "cycle_gain_short")


@generator("ns_cycle_budgets",
           "Radial derivative losses and stage parameters for one correction "
           "cycle, on both sides of where the cycle stops closing.",
           {"skip_lo": "start of the excluded window, as a reciprocal",
            "skip_hi": "end of the excluded window, as a reciprocal"})
def ns_cycle_budgets(n: int, rng: np.random.Generator,
                     skip_lo: int = 25, skip_hi: int = 29):
    """Half the instances have a loss small enough for the cycle to close and
    half do not, so the check has something to be wrong about.

    Reciprocals from 25 to 29 are left out, and the reason is worth recording:
    on that window the paper's closing summary fails while its own term table
    still gives the 1/10 gain. The summary is sufficient, not necessary, and
    sampling there would score a real difference between two correct
    statements as a disagreement.
    """
    out: list[Example] = []
    for i in range(n):
        if i % 2 == 0:
            k = Fraction(1, int(rng.integers(skip_hi + 1, 200_000)))
        else:
            k = Fraction(1, int(rng.integers(3, skip_lo)))
        j = int(rng.integers(0, 400))
        out.append(Example(inp=Content.abstract(
            f"cycle(kappa={sf(k)},sigma={sf(Fraction(1, 5) + Fraction(j, 10))})")))
    return out


# ---------------------------------------------------------------------------
# Lemma 5.4: summing a divergent expansion behind shrinking cutoffs
# ---------------------------------------------------------------------------
def _log_cutoff_scales(log_c, gain: float, terms: int, policy: str) -> list[float]:
    """log of the scale a_j at which the jth correction is switched off.

    The paper's recursion asks for a_j increasing, at least doubling, and
    large enough that the jth term with all its constants is below 2^-j
    wherever its cutoff is not already zero. Solving that at the worst point
    q = 1/a_j gives a_j >= (c_j 2^j)^(2/g_j), which is what `recursive`
    computes. `fixed` is the naive alternative -- one geometric schedule,
    chosen without looking at the coefficients.

    Everything is kept in logs. The scales the recursion produces are past
    the largest float by the twentieth term, which is not a numerical
    nuisance but the point of the lemma: the cutoffs have to shrink as fast
    as the coefficients grow, and the coefficients are allowed to grow
    however they like.
    """
    scales: list[float] = []
    previous = math.log(2.0)
    for j in range(1, terms + 1):
        g_j = gain * j
        if policy == "recursive":
            wanted = 2.0 * (log_c(j) + j * math.log(2.0)) / g_j
        else:
            wanted = j * math.log(2.0)
        value = max(previous + math.log(2.0), wanted)
        scales.append(value)
        previous = value
    return scales


@mechanism("lemma_5_4_summation")
def make_lemma_5_4_summation() -> PythonRule:
    """`summation(gain=1/5,growth=1/2,policy=recursive)` -> is the tail flat?

    Propositions 5.5 and 9.9 both sum an expansion whose coefficients need not
    converge, by multiplying the jth term by a cutoff that switches it off
    outside a shrinking neighbourhood of the singularity. Lemma 5.4 is where
    that is done, and its content is that the cutoff scales can be chosen
    from the coefficients themselves.

    The rule is the cheap reading of that: with the scales chosen by the
    recursion the tail is flat whatever the coefficients do, and with a
    schedule fixed in advance it is flat only if the coefficients grow slower
    than the schedule can absorb, which for coefficients exp(beta j^2) and a
    geometric schedule means beta below gain*log 2.
    """

    def fn(c: Content) -> Content | None:
        f = _parse(c.text, "summation", "gain:q,growth:q,policy:w")
        if f is None or f["policy"] not in ("recursive", "fixed"):
            return None
        gain, beta = float(f["gain"]), float(f["growth"])
        if gain <= 0 or beta <= 0:
            return None
        flat = f["policy"] == "recursive"
        return Content.abstract(
            "tail_flat_at_every_order" if flat else "tail_not_flat",
            derivation="cutoff schedule against coefficient growth")

    return PythonRule("lemma_5_4_summation", fn, ABSTRACT, ABSTRACT,
                      description="does the cutoff schedule of Lemma 5.4 make "
                                  "the summed tail flat",
                      source="summation(gain,growth,policy)->flat",
                      exact=False, trusted=False)


@oracle("summation_by_cutoffs",
        "flatness of the summed tail, measured by building the sum with the "
        "cutoff scales the schedule prescribes",
        "definitional")
def _summation_by_cutoffs(ex: Example) -> Content | None:
    """Lemma 5.4 carried out.

    Builds the series sum_j c_j chi(a_j q) q^(g_j) with c_j = exp(beta j^2),
    g_j = j*gain and the scales the policy gives, then measures the local
    exponent log S(q)/log q as q decreases. Flat means that exponent keeps
    climbing: the sum is smaller than every fixed power of q near zero. All
    of it is done in logs, since the coefficients here overflow a float long
    before the cutoffs switch them on.
    """
    f = _parse(ex.inp.text, "summation", "gain:q,growth:q,policy:w")
    if f is None or f["policy"] not in ("recursive", "fixed"):
        return None
    gain, beta = float(f["gain"]), float(f["growth"])
    if gain <= 0 or beta <= 0:
        return None
    terms = 80
    log_scales = _log_cutoff_scales(lambda j: beta * j * j, gain, terms,
                                    str(f["policy"]))

    def log_tail(log_q: float, after: int) -> float:
        """Everything past stage `after`, in logs. The coefficients overflow
        a float long before their cutoffs switch them on."""
        pieces = [beta * j * j + gain * j * log_q
                  for j in range(after + 1, terms + 1)
                  if log_scales[j - 1] + log_q < 0.0]
        if not pieces:
            return -math.inf
        top = max(pieces)
        return top + math.log(sum(math.exp(p - top) for p in pieces))

    # The lemma's own conclusion (5.35): past stage J the remainder is below
    # 2^-J q^(g_{J+1}/2), wherever the first J cutoffs are still one. That is
    # what is tested, at the largest such q and then well inside it.
    for J in range(1, 25):
        ceiling = -math.log(2.0) - log_scales[J - 1]
        for below in (0.0, math.log(1e-2), math.log(1e-5)):
            log_q = ceiling + below
            claimed = -J * math.log(2.0) + gain * (J + 1) / 2 * log_q
            if log_tail(log_q, J) > claimed:
                return Content.abstract("tail_not_flat")
    return Content.abstract("tail_flat_at_every_order")


@generator("ns_summation_schedules",
           "Coefficient growth and cutoff policy for the summation of Lemma "
           "5.4. Half the instances use the recursion the paper prescribes "
           "and half a geometric schedule chosen blind, the latter only at "
           "growth rates where it is known to fail.",
           {})
def ns_summation_schedules(n: int, rng: np.random.Generator):
    out: list[Example] = []
    for i in range(n):
        gain = Fraction(int(rng.integers(1, 40)), 100)
        threshold = float(gain) * math.log(2.0) / 2
        policy = "recursive" if i % 2 == 0 else "fixed"
        # The prescribed recursion is asked to cope with everything,
        # including growth a geometric schedule could also have handled;
        # the geometric one is asked only where it cannot. The multiplier
        # is drawn from the interval rather than from four points on it:
        # four points and five gains was 35 questions no matter how many
        # instances were requested, so the count stopped being evidence.
        lo, hi = (0.6, 25.0) if policy == "recursive" else (2.5, 25.0)
        beta = Fraction(max(1, round(threshold * float(rng.uniform(lo, hi)) * 1000)),
                        1000)
        out.append(Example(inp=Content.abstract(
            f"summation(gain={sf(gain)},growth={sf(beta)},policy={policy})")))
    return out


# ---------------------------------------------------------------------------
# Lemmas 10.2 and 10.3: extending the force past the singular time
# ---------------------------------------------------------------------------
def _smooth_step(x: np.ndarray) -> np.ndarray:
    """A cutoff that is one below 1/2, zero above 1, and C^5 in between.

    Polynomial rather than the usual exp(-1/x) so that its derivatives are
    exact and cheap; five continuous derivatives is more than the orders the
    oracle differentiates to, and the limitation is stated rather than
    hidden.
    """
    t = np.clip((x - 0.5) / 0.5, 0.0, 1.0)
    s = 462 * t ** 6 - 1980 * t ** 7 + 3465 * t ** 8 - 3080 * t ** 9 \
        + 1386 * t ** 10 - 252 * t ** 11
    return 1.0 - s


@mechanism("lemma_10_3_borel")
def make_lemma_10_3_borel() -> PythonRule:
    """`borel(growth=2,policy=shrinking)` -> does the force extend smoothly?

    The residual is defined only before t = 1. Lemma 10.2 gives every one of
    its derivatives a limit there, and Lemma 10.3 builds a force past that
    time which takes those limits as its Taylor data, as a series whose jth
    term carries a cutoff of width 1/b_j. The b_j are chosen from the size of
    the limits, which is what makes the differentiated series converge; the
    cheap reading is that the choice is what matters and the growth rate of
    the data does not.
    """

    def fn(c: Content) -> Content | None:
        f = _parse(c.text, "borel", "growth:q,policy:w")
        if f is None or f["policy"] not in ("shrinking", "fixed"):
            return None
        if f["growth"] <= 0:
            return None
        return Content.abstract(
            "force_extends_smoothly" if f["policy"] == "shrinking"
            else "extension_diverges",
            derivation="cutoff widths chosen from the derivative limits")

    return PythonRule("lemma_10_3_borel", fn, ABSTRACT, ABSTRACT,
                      description="does the Borel-type extension of the force "
                                  "converge after differentiation",
                      source="borel(growth,policy)->extends|diverges",
                      exact=False, trusted=False)


@oracle("force_extension_by_construction",
        "the extended force built term by term and differentiated, to see "
        "whether the series survives it",
        "definitional")
def _force_extension_by_construction(ex: Example) -> Content | None:
    """Lemma 10.3 carried out on scalar data.

    Takes derivative limits of size F_j = (j!)^growth, builds

        f(1+s) = sum_j chi(b_j s) s^j / j! * F_j,

    and differentiates it numerically at several points. `shrinking` picks
    b_j the way the lemma does, from the size of F_j; `fixed` uses one
    geometric schedule. The test is whether the partial sums of the
    differentiated series stop changing, and whether f keeps the derivative
    limits it was built from at s = 0.
    """
    f = _parse(ex.inp.text, "borel", "growth:q,policy:w")
    if f is None or f["policy"] not in ("shrinking", "fixed"):
        return None
    growth = float(f["growth"])
    terms = 40
    logs = [growth * math.lgamma(j + 1) for j in range(terms + 1)]   # log F_j

    widths = []
    previous = 1
    for j in range(1, terms + 1):
        if f["policy"] == "shrinking":
            # b_j from (10.12): b^(m-j) F_j <= 2^-j for every m <= j/2, worst
            # at m = floor(j/2), so b >= (F_j 2^j)^(1/ceil(j/2)).
            need = math.exp((logs[j] + j * math.log(2.0)) / math.ceil(j / 2))
        else:
            # Linear, not geometric. A geometric schedule chosen blind still
            # absorbs factorial data, so it would make the fixed policy look
            # as good as the prescribed one and measure nothing.
            need = float(j + 1)
        b = max(previous + 1, need)
        widths.append(b)
        previous = b

    def partial(s: np.ndarray, upto: int, order: int) -> np.ndarray:
        total = np.zeros_like(s)
        for j in range(1, upto + 1):
            b = widths[j - 1]
            if order == 0:
                term = _smooth_step(b * s) * s ** j * math.exp(
                    logs[j] - math.lgamma(j + 1))
            else:
                step = max(1e-6 / b, 1e-12)
                plus = _smooth_step(b * (s + step)) * (s + step) ** j
                minus = _smooth_step(b * (s - step)) * (s - step) ** j
                middle = _smooth_step(b * s) * s ** j
                if order == 1:
                    term = (plus - minus) / (2 * step)
                else:
                    term = (plus - 2 * middle + minus) / (step * step)
                term = term * math.exp(logs[j] - math.lgamma(j + 1))
            total = total + term
        return total

    grid = np.linspace(1e-4, 0.9, 41)
    for order in (0, 1, 2):
        near = partial(grid, terms // 2, order)
        far = partial(grid, terms, order)
        scale = max(float(np.max(np.abs(near))), 1.0)
        if not np.all(np.isfinite(far)) or float(np.max(np.abs(far - near))) > 1e-6 * scale:
            return Content.abstract("extension_diverges")
    return Content.abstract("force_extends_smoothly")


@generator("ns_force_extensions",
           "Derivative limits of factorial growth and a cutoff policy for the "
           "extension of the force past the singular time. The prescribed "
           "widths meet every growth rate; the schedule chosen blind is "
           "sampled only where it is known to fail.",
           {})
def ns_force_extensions(n: int, rng: np.random.Generator):
    out: list[Example] = []
    for i in range(n):
        # Tenths gave 35 shrinking and 15 fixed growth rates and nothing
        # more, so the thousandth is the unit here and the instance count
        # buys distinct questions again.
        if i % 2 == 0:
            policy = "shrinking"
            growth = Fraction(int(rng.integers(500, 4000)), 1000)
        else:
            # A linear schedule copes with derivative limits up to about
            # (j!)^2; past that it does not, and that is where it is asked.
            policy = "fixed"
            growth = Fraction(int(rng.integers(2500, 4000)), 1000)
        out.append(Example(inp=Content.abstract(
            f"borel(growth={sf(growth)},policy={policy})")))
    return out


# ---------------------------------------------------------------------------
# Lemma 10.4: the energy bound, from the equation
# ---------------------------------------------------------------------------
@mechanism("lemma_10_4_energy_bound")
def make_lemma_10_4_energy_bound() -> PythonRule:
    """`energy(dim=8,skew=1,seed=3)` -> is the energy bounded by the force?

    Lemma 10.4 gets its bound from one property of the nonlinearity: paired
    against the velocity it contributes nothing, so the energy identity has
    only dissipation and the force in it, and dividing by the norm gives
    ||u(t)|| <= integral ||f||. The rule reads that property off the cell.
    Everything else about the nonlinearity is irrelevant to the bound, which
    is the whole point of the argument and is exactly what the oracle tests
    by running the same argument with the property switched off.
    """

    def fn(c: Content) -> Content | None:
        f = _parse(c.text, "energy", "dim:q,skew:q,seed:q")
        if f is None:
            return None
        return Content.abstract(
            "energy_bounded_by_force_integral" if f["skew"] == 1
            else "energy_bound_fails",
            derivation="the nonlinearity drops out of the energy identity")

    return PythonRule("lemma_10_4_energy_bound", fn, ABSTRACT, ABSTRACT,
                      description="is the kinetic energy bounded by the "
                                  "integral of the force",
                      source="energy(dim,skew,seed)->bounded|fails",
                      exact=False, trusted=False)


@oracle("energy_identity_by_integration",
        "the energy bound measured by integrating a model with the same "
        "structure as the equation",
        "definitional")
def _energy_identity_by_integration(ex: Example) -> Content | None:
    """Lemma 10.4 on a finite-dimensional model of the same shape.

        u' = -A u + N(u) + f(t),   u(0) = 0,

    with A symmetric positive definite standing for the viscous term and N
    quadratic standing for the advection. When N is built so that <N(u),u> is
    zero -- the property the divergence-free advection has -- the argument of
    the lemma gives ||u(t)|| <= integral ||f||. When it is not, the same
    argument has nothing to stand on, and the oracle reports whether the
    bound actually survives.
    """
    f = _parse(ex.inp.text, "energy", "dim:q,skew:q,seed:q")
    if f is None:
        return None
    dim, skew, seed = int(f["dim"]), int(f["skew"]), int(f["seed"])
    if dim < 2:
        return None
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(dim, dim))
    A = root @ root.T / dim + np.eye(dim)                   # dissipative
    tensor = rng.normal(size=(dim, dim, dim)) / dim

    def nonlinear(u: np.ndarray) -> np.ndarray:
        raw = np.einsum("ijk,j,k->i", tensor, u, u)
        norm2 = float(u @ u)
        if norm2 <= 0:
            return raw
        if skew == 1:
            # Strip the component along u. That is exactly <N(u),u> = 0, the
            # one property of the advection term the energy identity uses.
            return raw - u * (float(raw @ u) / norm2)
        # The counterfactual: a quadratic term that pushes along the velocity
        # instead of across it, so the identity has a source the bound does
        # not account for. Anything weaker would be swamped by dissipation
        # and the check would see no difference.
        return raw + 20.0 * math.sqrt(norm2) * u

    amplitude = rng.normal(size=dim)

    def force(t: float) -> np.ndarray:
        return amplitude * math.sin(3.0 * t) * math.exp(-t)

    steps, T = 20_000, 4.0
    dt = T / steps
    u = np.zeros(dim)
    force_integral = 0.0
    worst = 0.0
    for i in range(steps):
        t = i * dt

        def rhs(tt: float, v: np.ndarray) -> np.ndarray:
            return -A @ v + nonlinear(v) + force(tt)

        k1 = rhs(t, u)
        k2 = rhs(t + dt / 2, u + dt / 2 * k1)
        k3 = rhs(t + dt / 2, u + dt / 2 * k2)
        k4 = rhs(t + dt, u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        force_integral += dt * float(np.linalg.norm(force(t + dt / 2)))
        if not np.all(np.isfinite(u)) or float(np.linalg.norm(u)) > 1e6:
            return Content.abstract("energy_bound_fails")
        worst = max(worst, float(np.linalg.norm(u)) - force_integral)
    return Content.abstract("energy_bounded_by_force_integral" if worst <= 1e-8
                            else "energy_bound_fails")


@generator("ns_energy_models",
           "Finite-dimensional models of the momentum equation: a dissipative "
           "part, a quadratic term that either cancels against the velocity "
           "or does not, and a forcing term.",
           {})
def ns_energy_models(n: int, rng: np.random.Generator):
    out: list[Example] = []
    for i in range(n):
        dim = int(rng.integers(3, 10))
        skew = 1 if i % 2 == 0 else 0
        seed = int(rng.integers(1, 10_000))
        out.append(Example(inp=Content.abstract(
            f"energy(dim={dim},skew={skew},seed={seed})")))
    return out


# ---------------------------------------------------------------------------
# Proposition 10.1: cutting the potential, not the velocity
# ---------------------------------------------------------------------------
@mechanism("prop_10_1_curl_order")
def make_prop_10_1_curl_order() -> PythonRule:
    """`localize(order=potential_first,seed=7)` -> is the cut field still
    divergence-free?

    The local solution has to be made compactly supported before it can be a
    solution on the whole space, and the paper does that by multiplying the
    VECTOR POTENTIAL by the cutoff and taking the curl afterwards. The order
    is the whole content: a curl is divergence-free whatever it is taken of,
    so cutting first costs nothing, while cutting the velocity leaves
    grad c . u behind and the field stops being incompressible.
    """

    def fn(c: Content) -> Content | None:
        f = _parse(c.text, "localize", "order:w,seed:q")
        if f is None or f["order"] not in ("potential_first", "velocity_first"):
            return None
        return Content.abstract(
            "divergence_free_compact_support" if f["order"] == "potential_first"
            else "divergence_leaks",
            derivation="the curl is taken after the cutoff, or before it")

    return PythonRule("prop_10_1_curl_order", fn, ABSTRACT, ABSTRACT,
                      description="does the localized field stay "
                                  "divergence-free",
                      source="localize(order,seed)->free|leaks",
                      exact=False, trusted=False)


@oracle("localization_by_finite_differences",
        "the divergence of the localized field, measured on a grid",
        "definitional")
def _localization_by_finite_differences(ex: Example) -> Content | None:
    """Proposition 10.1 carried out on an axisymmetric field.

    Builds a Stokes potential A = S(r,z)/r e_theta, forms the velocity by
    curl in cylindrical coordinates, applies a smooth cutoff in r and z
    either to the potential or to the velocity, and measures the largest
    divergence on the transition region against the size of the velocity
    gradient there. Nothing about the specific potential matters, which is
    why it is drawn at random from the seed.
    """
    f = _parse(ex.inp.text, "localize", "order:w,seed:q")
    if f is None or f["order"] not in ("potential_first", "velocity_first"):
        return None
    rng = np.random.default_rng(int(f["seed"]))
    c1, c2, c3 = rng.uniform(0.5, 2.0, size=3)

    def potential(r, z):
        """A_theta, smooth and vanishing at the axis like r."""
        return r * np.exp(-c1 * r * r) * np.cos(c2 * z) * (1.0 + c3 * r)

    def cut(r, z):
        return _smooth_step(np.sqrt(r * r + z * z) / 2.2)

    step = 1e-4

    def curl(r, z, amplitude):
        """(u_r, u_z) = (-d_z A, (d_r + 1/r) A) for an azimuthal potential."""
        d_z = (amplitude(r, z + step) - amplitude(r, z - step)) / (2 * step)
        d_r = (amplitude(r + step, z) - amplitude(r - step, z)) / (2 * step)
        return -d_z, d_r + amplitude(r, z) / r

    if f["order"] == "potential_first":
        def field(r, z):
            return curl(r, z, lambda rr, zz: cut(rr, zz) * potential(rr, zz))
    else:
        def field(r, z):
            u_r, u_z = curl(r, z, potential)
            return cut(r, z) * u_r, cut(r, z) * u_z

    worst = 0.0
    support = 0.0
    for r in np.linspace(0.4, 3.0, 25):
        for z in np.linspace(-2.0, 2.0, 25):
            u_r, u_z = field(r, z)
            rp, _ = field(r + step, z)
            _, zp = field(r, z + step)
            rm, _ = field(r - step, z)
            _, zm = field(r, z - step)
            divergence = (rp - rm) / (2 * step) + u_r / r + (zp - zm) / (2 * step)
            scale = max(abs(rp - rm) / (2 * step), abs(zp - zm) / (2 * step),
                        abs(u_r / r), 1e-12)
            worst = max(worst, abs(divergence) / scale)
            if math.hypot(r, z) > 2.4:
                support = max(support, abs(u_r) + abs(u_z))
    if support > 1e-9:
        return Content.abstract("divergence_leaks")
    return Content.abstract("divergence_free_compact_support" if worst < 1e-4
                            else "divergence_leaks")


@generator("ns_localizations",
           "Axisymmetric potentials and the order in which the cutoff and "
           "the curl are applied, half each way.",
           {})
def ns_localizations(n: int, rng: np.random.Generator):
    return [Example(inp=Content.abstract(
        f"localize(order={'potential_first' if i % 2 == 0 else 'velocity_first'},"
        f"seed={int(rng.integers(1, 10_000))})")) for i in range(n)]


# ---------------------------------------------------------------------------
# Proposition 7.5: the covariance, integrated rather than frozen
# ---------------------------------------------------------------------------
@oracle("covariance_from_integrated_pulses",
        "the admissible cone decided by integrating both pulses across their "
        "slot, forming their covariance as an integral, and solving for the "
        "squared amplitudes",
        "definitional", ("admissible", "not_admissible"))
def _covariance_from_integrated_pulses(ex: Example) -> Content | None:
    """Proposition 7.5 with the pulses actually solved for.

    The cone's other oracle in `navierstokes.py` builds the covariance from
    the frozen leading form (7.28), where the pulse has been replaced by its
    value at the middle of the slot and the decaying coordinate dropped.
    This one integrates the amplitude equation of Lemma 7.4 across the slot
    for both signs, keeps both frame coordinates, and forms the covariance
    as the integral (7.27) that defines it, envelope and all. The weights
    then come from solving the 2x2 system, and admissible means both are
    positive.

    What is still left out is the frame error E of (7.10): the evolution
    used here is the reference one, so the check covers the mechanism at
    leading order and not the estimate that its errors do not spoil the
    positivity.
    """
    f = _parse(ex.inp.text.replace("cone(", "cone(").replace(")", ")"),
               "cone", "a:q,bs:q,p1:q,p2:q")
    if f is None or f["a"] <= 0:
        return None
    a, bs = float(f["a"]), float(f["bs"])
    g0 = np.array([-a, bs])
    norm = float(np.linalg.norm(g0))
    if norm == 0:
        return None
    N = g0 / norm
    K = np.array([-N[1], N[0]])
    disc = -2 * N[0] * (2 * N[0] + norm)
    if disc <= 0:
        return Content.abstract("not_admissible")
    lam0 = math.sqrt(disc)
    c0 = lam0 / (2 * N[0])
    target = np.array([float(f["p1"]) - a, float(f["p2"]) + bs])
    if float(target @ N) >= 0:
        return Content.abstract("not_admissible")

    slot, steps = 40.0, 240
    dt = slot / steps
    for ustar in np.linspace(0.1, 40.0, 90):
        columns = []
        for sign in (+1.0, -1.0):
            z = np.array([1.0, 0.0])
            column = np.zeros(2)
            for i in range(steps):
                v = i * dt
                phase = sign * (ustar / 2 + ustar * v / slot)
                lam = lam0 / math.sqrt(1 + phase * phase)
                damp = lam0 * (1 + phase * phase) / (1 + ustar * ustar) ** 1.5
                rate = np.array([lam - damp, -lam - damp])
                z = z * np.exp(rate * dt)               # exact on the diagonal
                x = z[0] + z[1]
                y = c0 * math.sqrt(1 + phase * phase) * (z[0] - z[1])
                envelope = _smooth_step(abs(2 * v / slot - 1.0) * 1.5) ** 2
                column += envelope * x * (-phase * x * K + y * N) * dt
            columns.append(column)
        matrix = np.stack(columns, axis=1)
        if abs(float(np.linalg.det(matrix))) < 1e-14:
            continue
        weights = np.linalg.solve(matrix, target)
        if weights[0] > 0 and weights[1] > 0:
            return Content.abstract("admissible")
    return Content.abstract("not_admissible")


# ---------------------------------------------------------------------------
# Proposition 4.2 and (4.1): the exponents, against an actual field
# ---------------------------------------------------------------------------
def _solve_q(z: float, tau: float, D: float) -> float:
    """The concentration scale at (z, tau), from tau = q(1 - eta^2).

    With eta = z/q^D that reads q - z^2 q^(1-2D) = tau, which is increasing
    in q above |z|^(1/D) and so has one root there. Bisection, because the
    exponent is arbitrary here and Newton would need a sign condition the
    generator is allowed to violate.
    """
    lower = abs(z) ** (1.0 / D) if z != 0.0 else 0.0
    low, high = lower + 1e-15, max(lower * 2.0 + 1.0, tau * 4.0 + 1.0)
    for _ in range(200):
        mid = 0.5 * (low + high)
        if mid - z * z * mid ** (1.0 - 2.0 * D) < tau:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _swirl(r: float, z: float, t: float, A: float, D: float) -> float:
    """u_theta = q^-A E(X, eta) for a fixed smooth profile E."""
    tau = 1.0 - t
    if tau <= 0.0 or r <= 0.0:
        return 0.0
    q = _solve_q(z, tau, D)
    X = r * r / (2.0 * q)
    eta = z / q ** D
    E = math.sqrt(2.0 * X) * math.exp(-X) / (1.0 + eta * eta)
    return q ** (-A) * E


def _axial(r: float, z: float, t: float, A: float, D: float) -> float:
    """u_z = q^-A U(X, eta), also fixed and smooth."""
    tau = 1.0 - t
    if tau <= 0.0:
        return 0.0
    q = _solve_q(z, tau, D)
    X = r * r / (2.0 * q)
    eta = z / q ** D
    # Deliberately comparable in size to the swirl. Axial transport is the
    # only term whose exponent depends on D, so a profile with a faint axial
    # component would leave the check almost nothing to measure.
    U = 4.0 * math.exp(-X) * (0.4 + 0.6 * eta) * (1.0 + 0.3 * X)
    return q ** (-A) * U


def _radial(r: float, z: float, t: float, A: float, D: float) -> float:
    """u_r from incompressibility, integrated out from the axis.

    r u_r = -integral_0^r s d_z u_z ds. Imposing it this way rather than
    from the profile identity (4.7) keeps the field exactly divergence-free
    whatever the exponents are, which matters here: the point of the check
    is to try exponents the construction forbids.
    """
    if r <= 0.0:
        return 0.0
    step = 1e-5
    nodes = 80
    grid = np.linspace(0.0, r, nodes + 1)[1:]
    values = np.array([(_axial(float(sv), z + step, t, A, D)
                        - _axial(float(sv), z - step, t, A, D)) / (2 * step) * sv
                       for sv in grid])
    integral = float(_trapezoid(values, grid))
    return -integral / r


_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _angular_residual(r: float, z: float, t: float, A: float, D: float) -> float:
    """The leading angular residual: transport and radial viscosity, with
    axial viscosity removed as in (4.12)."""
    hr, hz, ht = 1e-4 * r, 1e-4 * max(abs(z), 1e-2), 1e-6
    u = _swirl(r, z, t, A, D)
    d_r = (_swirl(r + hr, z, t, A, D) - _swirl(r - hr, z, t, A, D)) / (2 * hr)
    d_rr = (_swirl(r + hr, z, t, A, D) - 2 * u
            + _swirl(r - hr, z, t, A, D)) / (hr * hr)
    d_z = (_swirl(r, z + hz, t, A, D) - _swirl(r, z - hz, t, A, D)) / (2 * hz)
    d_t = (_swirl(r, z, t + ht, A, D) - _swirl(r, z, t - ht, A, D)) / (2 * ht)
    u_r = _radial(r, z, t, A, D)
    u_z = _axial(r, z, t, A, D)
    transport = d_t + u_r * d_r + u_z * d_z + u_r * u / r
    viscous = d_rr + d_r / r - u / (r * r)
    return transport - viscous


@mechanism("eq_4_1_exponents")
def make_eq_4_1_exponents() -> PythonRule:
    """`similarity(A=101/200,D=99/200,X=1,eta=1/5)` -> is the residual
    self-similar?

    Everything downstream in the construction assumes the leading residual
    is a function of (X, eta) times one power of the concentration scale,
    because that is what lets a single profile equation stand for every
    time. Whether it is depends on the exponents: transport by the radial
    and swirl velocities scales as q^(-A-1) on its own, and axial transport
    scales as q^(-2A-D), so the two agree only when A + D = 1. That identity
    is what the rule checks, in one line of rational arithmetic.

    Until now the exponents in `navierstokes.py` were written down and never
    weighed against anything. Here they are: the oracle builds an exactly
    divergence-free field from the given exponents, differentiates it, and
    reports whether the residual really does collapse.
    """

    def fn(c: Content) -> Content | None:
        f = _parse(c.text, "similarity", "A:q,D:q,X:q,eta:q")
        if f is None or f["A"] <= 0 or f["D"] <= 0 or f["X"] <= 0:
            return None
        return Content.abstract(
            "residual_self_similar" if f["A"] + f["D"] == 1
            else "residual_not_self_similar",
            derivation="A + D = 1 matches axial to radial transport")

    return PythonRule("eq_4_1_exponents", fn, ABSTRACT, ABSTRACT,
                      description="do the similarity exponents collapse the "
                                  "leading residual",
                      source="similarity(A,D,X,eta)->self_similar|not",
                      exact=False, trusted=False)


@oracle("residual_collapse_by_finite_differences",
        "whether the leading angular residual collapses to one profile, "
        "measured by building the field at two scales and comparing",
        "definitional",
        ("residual_self_similar", "residual_not_self_similar"))
def _residual_collapse_by_finite_differences(ex: Example) -> Content | None:
    """The exponents of (4.1), weighed against a field rather than asserted.

    Two points are chosen with the same similarity coordinates (X, eta) and
    concentration scales a decade apart. At each, the field is built from
    the given exponents, made divergence-free by integrating the radial
    component out from the axis, and differentiated to form the leading
    angular residual. If the similarity claim holds, q^(A+1) times that
    residual is the same number at both points.
    """
    f = _parse(ex.inp.text, "similarity", "A:q,D:q,X:q,eta:q")
    if f is None or f["A"] <= 0 or f["D"] <= 0 or f["X"] <= 0:
        return None
    A, D = float(f["A"]), float(f["D"])
    X, eta = float(f["X"]), float(f["eta"])
    if abs(eta) >= 1.0:
        return None

    def collapse(x_value: float, eta_value: float) -> float | None:
        """Relative disagreement between two scales at one similarity point."""
        normalized = []
        for q in (2e-2, 2e-4):
            z = eta_value * q ** D
            tau = q * (1.0 - eta_value * eta_value)
            r = math.sqrt(2.0 * q * x_value)
            residual = _angular_residual(r, z, 1.0 - tau, A, D)
            if not math.isfinite(residual):
                return None
            normalized.append(q ** (A + 1.0) * residual)
        scale = max(abs(v) for v in normalized)
        return None if scale == 0.0 else abs(normalized[0] - normalized[1]) / scale

    # Several points, worst case. The terms whose exponent depends on D can
    # cancel against each other on a curve -- for this profile that happens
    # near eta = 1/10, where a mismatch of a twentieth moves the residual by
    # less than a matched pair's own numerical noise -- and a check made at
    # one point would report the collapse it happened to land on.
    places = [(X, eta), (X * 1.7, eta), (X, min(abs(eta) + 0.35, 0.8)),
              (X * 0.6, -min(abs(eta) + 0.2, 0.8))]
    spreads = [value for value in
               (collapse(x, e) for x, e in places) if value is not None]
    if not spreads:
        return None
    spread = max(spreads)
    # Two decades apart. A matched pair agrees to five digits there and a
    # mismatch of a twentieth in the exponent moves it by several percent,
    # so the threshold sits in a gap of two orders of magnitude rather than
    # on a slope.
    return Content.abstract("residual_self_similar" if spread < 3e-3
                            else "residual_not_self_similar")


@generator("ns_similarity_points",
           "Similarity exponents and a point to test them at. Half the "
           "instances use the pairing the construction requires and half "
           "break it, so the check has to notice.",
           {})
def ns_similarity_points(n: int, rng: np.random.Generator):
    out: list[Example] = []
    for i in range(n):
        h = Fraction(1, int(rng.integers(101, 400)))
        A = Fraction(1, 2) + h
        D = Fraction(1, 2) - h if i % 2 == 0 else \
            Fraction(1, 2) - h + Fraction(int(rng.choice([-2, -1, 1, 2])), 20)
        X = Fraction(int(rng.integers(5, 30)), 10)
        eta = Fraction(int(rng.integers(-6, 7)), 10)
        out.append(Example(inp=Content.abstract(
            f"similarity(A={sf(A)},D={sf(D)},X={sf(X)},eta={sf(eta)})")))
    return out


#: Each mechanism, the oracle that carries it out, and its generator.
MECHANISM_CHECKS = {
    "eq_4_1_exponents": ("residual_collapse_by_finite_differences",
                         "ns_similarity_points"),
    "prop_9_6_table": ("cycle_table_by_enumeration", "ns_cycle_budgets"),
    "lemma_5_4_summation": ("summation_by_cutoffs", "ns_summation_schedules"),
    "lemma_10_3_borel": ("force_extension_by_construction",
                         "ns_force_extensions"),
    "lemma_10_4_energy_bound": ("energy_identity_by_integration",
                                "ns_energy_models"),
    "prop_10_1_curl_order": ("localization_by_finite_differences",
                             "ns_localizations"),
}

#: Which imported step each mechanism speaks for, and what it leaves behind.
SUPPORTS = {
    "thm_4_6_profiles": (
        "eq_4_1_exponents",
        "the exponents collapse the residual; the profiles themselves, their "
        "cone margin and their moment identities are not built here"),
    "prop_5_5_background": (
        "lemma_5_4_summation",
        "the cutoff schedule tames any coefficient growth; the coefficient "
        "bounds it is given are the estimate"),
    "prop_7_5_stress": (
        "lemma_4_5_cone",
        "the covariance is integrated from the pulses and solved for positive "
        "weights; the frame error of (7.10) is still dropped"),
    "prop_9_5_initialize": (
        None,
        "not attempted: the orders the cycle starts from are the estimates of "
        "Proposition 9.5"),
    "prop_9_6_induction": (
        "prop_9_6_table",
        "the induction over stages is now PROVED here rather than sampled "
        "(prop_9_6_all_stages), so what is still imported is only the "
        "per-stage estimates the four moves encode"),
    "prop_9_9_summation": (
        "lemma_5_4_summation",
        "same schedule, same gap: flatness of the tail given the bounds, not "
        "the bounds"),
    "prop_10_1_localize": (
        "prop_10_1_curl_order",
        "cutting the potential before the curl keeps the field "
        "divergence-free and compactly supported"),
    "lemma_10_3_force": (
        "lemma_10_3_borel",
        "the extension converges after differentiation; that the residual's "
        "derivative limits exist at all is Lemma 10.2"),
    "lemma_10_4_energy": (
        "lemma_10_4_energy_bound",
        "the bound follows from the nonlinearity dropping out, on a model "
        "with that structure rather than on the equation"),
    "lemma_10_5_unique": (
        None,
        "not attempted: the comparison argument turns on a pressure flux "
        "controlled by Riesz transforms on the whole space, and nothing here "
        "models that"),
    "thm_1_1_blowup": (
        None,
        "not attempted: it is the conjunction of every step above, uniformly"),
}


def _mutant_similarity(c: Content) -> "Content | None":
    """A + D = 1 relaxed to within a tenth."""
    f = _parse(c.text, "similarity", "A:q,D:q,X:q,eta:q")
    if f is None or f["A"] <= 0 or f["D"] <= 0 or f["X"] <= 0:
        return None
    ok = abs(f["A"] + f["D"] - 1) <= Fraction(1, 10)
    return Content.abstract("residual_self_similar" if ok
                            else "residual_not_self_similar")


#: Mutations for the mechanisms whose criterion is more than a flag. For the
#: rest the base rate is already the mutation test: a generator that emits
#: both answers equally often makes "always say yes" score one half.
MECHANISM_MUTATIONS = {
    "eq_4_1_exponents": ("A + D = 1 relaxed to within 1/10", _mutant_similarity),
}


def install_mechanism_rules(library) -> None:
    """Add the executed mechanisms to a library that already has the
    construction's rules."""
    for factory in MECHANISM_RULES.values():
        library.add(factory(), replace=True)


#: What running these mechanisms still does not give.
REMAINING = (
    "Each mechanism above was executed on instances, and every one of the "
    "results it stands for is uniform: in the concentration scale q as it "
    "goes to zero, in the dyadic band, in the slow label, in the correction "
    "stage. An instance cannot reach a statement of that shape.",
    "The models are models. The energy bound is checked on a "
    "finite-dimensional system with the same algebraic structure, not on the "
    "Navier-Stokes equation; the summation is checked on a coefficient "
    "sequence, not on the actual expansion; the force extension is checked on "
    "scalar derivative limits, not on the residual's.",
    "Where a rule and its oracle both come from the same page of the paper, "
    "agreement measures transcription and internal consistency, not truth. "
    "The cycle table is the clearest case: it establishes that the proof's "
    "summary and its body describe the same bookkeeping.",
)
