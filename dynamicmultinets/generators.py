"""
Generators: the machine writing its own experiments.

"New rules are summarized or learned from thought or tool experiments generated
from prior knowledge and existing rules that have already been learned so far."
That sentence is this module. A generator produces INPUTS only -- unlabeled
cells the machine has decided are worth looking at. Turning them into (data,
label) pairs is the oracle's job (oracles.py), and it is kept separate because
the two answer different questions: what is worth trying, versus what is true.

Generators are a closed registry, not arbitrary code. The controller is an LLM;
letting it emit executable Python to make training data would put model output
straight into the interpreter. It chooses a generator by name and supplies
parameters, which is enough expressiveness for everything in the paper and
leaves nothing to sanitize.

Every generator takes (n, rng, **params) and returns a list of Examples in a
deterministic order for a given seed, so an ExampleSet's provenance really does
reproduce it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .dataset import Example, ExampleSet
# The geometry decision boundary and step sizes, imported rather than restated:
# a generator that disagrees with the oracle about where a construction stops
# trains rules on configurations nobody scores them on.
from .oracles import _ANGLE_STEP, _OFFSET_TOL
from .tapes import ABSTRACT, SPECIFIC, Content

GeneratorFn = Callable[..., "list[Example]"]


@dataclass
class GeneratorSpec:
    name: str
    fn: GeneratorFn
    doc: str
    params: dict[str, str]          # name -> human description, shown to the LLM


GENERATORS: dict[str, GeneratorSpec] = {}


def generator(name: str, doc: str, params: dict[str, str]):
    def wrap(fn: GeneratorFn) -> GeneratorFn:
        GENERATORS[name] = GeneratorSpec(name, fn, doc, params)
        return fn

    return wrap


def generate(name: str, n: int, seed: int = 0, **params: Any) -> ExampleSet:
    """Run a registered generator and wrap the result with its provenance."""
    if name not in GENERATORS:
        raise KeyError(f"no generator {name!r}; known: {sorted(GENERATORS)}")
    spec = GENERATORS[name]
    rng = np.random.default_rng(seed)
    examples = spec.fn(n, rng, **params)
    return ExampleSet(name=f"{name}_{n}_{seed}", examples=examples,
                      generator=name, generator_params=dict(params), seed=seed)


def catalogue() -> str:
    rows = []
    for spec in sorted(GENERATORS.values(), key=lambda s: s.name):
        args = ", ".join(f"{k}: {v}" for k, v in spec.params.items()) or "none"
        rows.append(f"{spec.name}: {spec.doc}\n    params: {args}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
@generator(
    "mul_pairs",
    "products a*b written on the abstract tape",
    {"a_digits": "digits in the left factor (default 2)",
     "b_digits": "digits in the right factor (default 2)",
     "round_b": "if true the right factor is a multiple of 10 (default false)",
     "tail_digits": "digits in the left factor for the last 20% (generalisation tail)",
     "domain": "'abstract' (symbols) or 'specific' (drawn on the screen, default)"},
)
def _mul_pairs(n: int, rng: np.random.Generator, a_digits: int = 2, b_digits: int = 2,
               round_b: bool = False, tail_digits: int | None = None,
               domain: str = "specific") -> list[Example]:
    # Say so rather than treating everything that is not "specific" as
    # abstract. A controller that asks for domain="integers" means something,
    # and silently drawing the opposite tape gives it a rule that reads symbols
    # when it asked for one that reads the screen -- a mistake that survives
    # training and verification, because both are measured on the wrong data.
    if domain not in (ABSTRACT, SPECIFIC):
        raise ValueError(f"domain must be {ABSTRACT!r} or {SPECIFIC!r}, "
                         f"got {domain!r}")
    if a_digits < 1 or b_digits < 1 or (tail_digits is not None and tail_digits < 1):
        raise ValueError("digit counts must be at least 1; got "
                         f"a_digits={a_digits}, b_digits={b_digits}, "
                         f"tail_digits={tail_digits}")

    def draw(digits: int) -> int:
        lo, hi = 10 ** (digits - 1), 10 ** digits - 1
        return int(rng.integers(max(lo, 1), hi + 1))

    out, cut = [], int(n * 0.8)
    for i in range(n):
        da = a_digits if (tail_digits is None or i < cut) else tail_digits
        a, b = draw(da), draw(b_digits)
        if round_b:
            b = (b // 10) * 10 or 10
        text = f"{a}*{b}"
        # A rule that is supposed to be learned FROM PICTURES has to be trained
        # on pictures; the default therefore writes the experiment to the
        # specific tape. The caption is kept as provenance so an oracle can
        # still say what the correct answer is.
        out.append(Example(inp=(Content.specific_text(text) if domain == "specific"
                                else Content.abstract(text))))
    return out


@generator(
    "sum_expressions",
    "sums of products, e.g. 10*30+2*30, on the abstract tape",
    {"terms": "how many product terms (default 2)", "digits": "digits per factor (default 2)"},
)
def _sum_expressions(n: int, rng: np.random.Generator, terms: int = 2,
                     digits: int = 2) -> list[Example]:
    out = []
    for _ in range(n):
        parts = []
        for _ in range(terms):
            a = int(rng.integers(1, 10 ** digits))
            b = int(rng.integers(1, 10 ** digits))
            parts.append(f"{a}*{b}")
        out.append(Example(inp=Content.abstract("+".join(parts))))
    return out


@generator(
    "rendered_expressions",
    "expressions already drawn on the specific tape -- training data for a reader "
    "(specific -> abstract) rule",
    {"max_terms": "product terms per expression (default 2)",
     "digits": "digits per factor (default 2)"},
)
def _rendered_expressions(n: int, rng: np.random.Generator, max_terms: int = 2,
                          digits: int = 2) -> list[Example]:
    out = []
    for _ in range(n):
        k = int(rng.integers(1, max_terms + 1))
        parts = [f"{int(rng.integers(1, 10 ** digits))}*{int(rng.integers(1, 10 ** digits))}"
                 for _ in range(k)]
        text = "+".join(parts)
        out.append(Example(inp=Content.specific_text(text),
                           meta={"truth": text}))
    return out


@generator(
    "rendered_integers",
    "single integers drawn on the specific tape -- training data for a reader "
    "whose output is fed to the divisor-sum rules",
    {"min_digits": "shortest integer to draw (default 1)",
     "max_digits": "longest integer to draw (default 5)",
     "above": "draw only integers strictly greater than this (default 0), which "
              "is how the Robin range n>5040 is asked for"},
)
def _rendered_integers(n: int, rng: np.random.Generator, min_digits: int = 1,
                       max_digits: int = 5, above: int = 0) -> list[Example]:
    # Digit LENGTH is sampled first, then a value of that length. Sampling
    # uniformly from [10^min, 10^max] instead would put ~90% of the mass on the
    # longest length, and a reader trained that way is worst at exactly the
    # short inputs whose divisor sums are cheapest to check by hand.
    out = []
    while len(out) < n:
        k = int(rng.integers(min_digits, max_digits + 1))
        lo, hi = 10 ** (k - 1), 10 ** k - 1
        v = int(rng.integers(max(lo, 1), hi + 1))
        if v <= above:
            continue
        text = str(v)
        out.append(Example(inp=Content.specific_text(text), meta={"truth": text}))
    return out


@generator(
    "robin_cases",
    "integers whose Robin verdict actually VARIES -- the exceptional integers "
    "that fail the inequality, balanced against integers above 5040 that pass",
    {"fail_fraction": "share of cases drawn from the exceptional set (default 0.5)",
     "max_digits": "longest passing integer to draw (default 5)"},
)
def _robin_cases(n: int, rng: np.random.Generator, fail_fraction: float = 0.5,
                 max_digits: int = 5) -> list[Example]:
    """Why this generator has to exist, and it is not a detail.

    `rendered_integers(above=5040)` labelled by `robin_verdict` produces 300
    cases and 300 identical labels, because Robin's inequality is believed to
    hold at every n>5040 and certainly does over any range we can enumerate. A
    rule verified on that family scores 1.000 by answering "robin_holds" and
    never reading anything -- measured, not hypothesised: a reader at 0.133
    character accuracy drove the composite to a perfect score on exactly that
    set. The dataset was vacuous, and the accuracy number was reporting the base
    rate of its own labels.

    The only integers that make the verdict informative are the ones that FAIL,
    and Robin's theorem is precisely the statement that there are finitely many:
    26 of them at n>=3 (n=2 is excluded because ln ln 2 is negative and the
    ratio is not defined in the intended direction). So the discriminating half
    of any Robin test set is capped at 26 distinct cases, and this generator
    emits each at most once rather than resampling to a requested size -- the
    same drawing repeated does not add evidence, it only inflates a denominator.

    The exceptional set is FOUND here, by evaluating the criterion, not pasted
    in from the literature. If the machine's arithmetic disagreed with Robin,
    that disagreement should surface as a different test set rather than be
    hidden by a hard-coded list that papers over it.
    """
    from .prior import EXP_GAMMA, sigma

    exceptional = [k for k in range(3, 5041)
                   if sigma(k) / (k * math.log(math.log(k))) >= EXP_GAMMA]
    n_fail = min(len(exceptional), int(round(n * fail_fraction)))
    fails = list(rng.permutation(exceptional))[:n_fail]

    holds: list[int] = []
    hi = 10 ** max_digits - 1
    while len(holds) < n - n_fail:
        holds.append(int(rng.integers(5041, hi + 1)))

    out = []
    for v in fails + holds:
        text = str(v)
        out.append(Example(inp=Content.specific_text(text), meta={"truth": text}))
    return [out[i] for i in rng.permutation(len(out))]


# ---------------------------------------------------------------------------
# Level spacings: random matrices, and the zeta zeros
# ---------------------------------------------------------------------------
# Both generators below produce cells of the SAME shape by the SAME reduction,
# and that is the only reason a comparison between them means anything. A rule
# trained on 500-spacing histograms and shown a 2000-spacing one could separate
# the two on sampling noise alone and would look like it had discovered
# something. `n_spacings` is therefore a parameter of both, and the experiment
# passes the same value to each.
_ZERO_CACHE: dict[int, Any] = {}


@generator(
    "spacing_histograms",
    "nearest-neighbour level-spacing histograms drawn from a known random-matrix "
    "ensemble -- training data for a rule that identifies an ensemble by eye",
    {"n_spacings": "spacings summarised by each drawing (default 500)",
     "dim": "matrix dimension the eigenvalues come from (default 60)",
     "ensembles": "which ensembles to draw from (default all three)"},
)
def _spacing_histograms(n: int, rng: np.random.Generator, n_spacings: int = 500,
                        dim: int = 60, ensembles: Any = None) -> list[Example]:
    from .render import spacing_scene
    from .zeta import ENSEMBLES, ensemble_spacings

    kinds = list(ensembles) if ensembles else list(ENSEMBLES)
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]          # balanced by construction
        sp = ensemble_spacings(kind, n_spacings, rng, dim=dim)
        scene = spacing_scene(sp, kind)
        out.append(Example(
            # The caption does NOT name the ensemble. The net only ever sees
            # pixels so it could not read it either way, but `transcribe_unsafe`
            # copies captions, and a cell carrying its own answer in its text is
            # a cheat waiting for the one rule that takes it.
            inp=Content.specific_sketch(scene, caption=f"spacing_{i}", ensemble=kind),
            meta={"ensemble": kind}))
    return [out[i] for i in rng.permutation(len(out))]


@generator(
    "zeta_spacings",
    "the same histogram, built from the actual nontrivial zeros of the Riemann "
    "zeta function -- data from outside, with NO ensemble label",
    {"n_spacings": "spacings summarised by each drawing (default 500)",
     "skip": "zeros to discard before the first block (default 0)"},
)
def _zeta_spacings(n: int, rng: np.random.Generator, n_spacings: int = 500,
                   skip: int = 0) -> list[Example]:
    """Cells the machine did not invent, and deliberately cannot label.

    Every other generator here is paired with an oracle that knows the answer,
    because the machine either drew the data or can compute the truth. Not this
    one: what ensemble the zeta zeros "are" is the open question, so these
    examples carry no label, `spacing_ensemble` declines them, and any attempt
    to VERIFY a rule on them reports that nothing was labelled rather than a
    number. Applying an already-verified rule to them is the only thing that can
    be done, and what comes back is a conjecture.
    """
    from .render import spacing_scene
    from .zeta import unfolded_spacings, zeta_zeros

    need = skip + n * n_spacings + 1
    if need not in _ZERO_CACHE:
        _ZERO_CACHE[need] = zeta_zeros(need)
    zeros = _ZERO_CACHE[need]
    spacings = unfolded_spacings(zeros)[skip:]

    out = []
    for i in range(n):
        block = spacings[i * n_spacings:(i + 1) * n_spacings]
        lo = zeros[skip + i * n_spacings]
        hi = zeros[skip + (i + 1) * n_spacings]
        scene = spacing_scene(block, f"zeta_{i}")
        out.append(Example(
            inp=Content.specific_sketch(scene, caption=f"zeta_t{lo:.0f}-{hi:.0f}",
                                        source="zeta"),
            meta={"source": "zeta", "t_lo": float(lo), "t_hi": float(hi)}))
    return out


# ---------------------------------------------------------------------------
# Robotics (Appendix A)
# ---------------------------------------------------------------------------
@generator(
    "robot_scenes",
    "sketches of a robot, obstacles and a goal in the simplified 3D space -- "
    "training data for the sketch-to-direction rule",
    {"n_obstacles": "obstacles per scene (default 2)",
     "blocked_fraction": "fraction of scenes where the straight path is blocked (default 0.5)"},
)
def _robot_scenes(n: int, rng: np.random.Generator, n_obstacles: int = 2,
                  blocked_fraction: float = 0.5) -> list[Example]:
    from .oracles import segment_hits_sphere

    out = []
    for i in range(n):
        want_blocked = bool(rng.random() < blocked_fraction)   # see triangle_scenes
        for _ in range(80):                     # rejection-sample the wanted case
            eef = rng.uniform(-0.7, 0.7, 3)
            goal = rng.uniform(-0.7, 0.7, 3)
            if np.linalg.norm(goal - eef) < 0.4:
                continue
            obstacles = []
            for _ in range(n_obstacles):
                t = rng.uniform(0.25, 0.75) if want_blocked else rng.uniform(0.0, 1.0)
                jitter = rng.normal(0, 0.05 if want_blocked else 0.45, 3)
                c = eef + t * (goal - eef) + jitter
                obstacles.append(tuple(np.clip(c, -0.9, 0.9)) + (float(rng.uniform(0.10, 0.22)),))
            blocked = any(segment_hits_sphere(eef, goal, o[:3], o[3]) for o in obstacles)
            if blocked == want_blocked:
                break
        scene = {"kind": "robot", "base": (0.0, 0.0, -0.9),
                 "eef": tuple(map(float, eef)), "goal": tuple(map(float, goal)),
                 "obstacles": [tuple(map(float, o)) for o in obstacles]}
        out.append(Example(inp=Content.specific_sketch(scene, caption=f"scene{i}")))
    return out


# ---------------------------------------------------------------------------
# Geometry (Figure 3)
# ---------------------------------------------------------------------------
@generator(
    "triangle_scenes",
    "a triangle plus an auxiliary line at a random offset and angle -- training "
    "data for deciding the next construction step, and for reading the angle facts",
    {"solved_fraction": "fraction already in the proof configuration, i.e. the line "
                        "passes through the apex and is parallel to the base (default 0.25)"},
)
def _triangle_scenes(n: int, rng: np.random.Generator,
                     solved_fraction: float = 0.25) -> list[Example]:
    out = []
    for i in range(n):
        apex = (float(rng.uniform(-0.35, 0.35)), float(rng.uniform(0.30, 0.55)))
        left = (float(rng.uniform(-0.85, -0.45)), float(rng.uniform(-0.65, -0.35)))
        right = (float(rng.uniform(0.45, 0.85)), left[1] + float(rng.uniform(-0.10, 0.10)))
        base_angle = float(np.arctan2(right[1] - left[1], right[0] - left[0]))

        # Drawn per example, NOT from i % 100. A periodic pattern lines up with
        # the held-out tail and hands the holdout a single class, which reads as
        # perfect accuracy and means nothing.
        solved = bool(rng.random() < solved_fraction)
        if solved:
            # Where the CONSTRUCTION stops, not the idealised configuration.
            # The loop moves in steps of 0.12 and rotates in steps of 0.15, and
            # `next_construction_step` stops it once another move would not get
            # closer -- so it finishes somewhere in |offset| <= 0.06 and
            # |err| <= 0.075, essentially never on zero. Drawing only the exact
            # configuration produced a reader measured at 1.000 on an exactly
            # parallel line and 0.525 on the one the construction actually
            # makes: a rule that verifies at 0.995 and then cannot recognise a
            # finished proof, which is what left `triangle_180` unproved.
            #
            # These bounds stop short of the 0.12 the reader is scored against,
            # and unsolved scenes below start at 0.12, so no two drawings in
            # this set are visually alike with opposite labels. Filling the
            # whole tolerance band instead does restore the proof, but triples
            # the rate of proofs licensed on unfinished constructions.
            offset = float(rng.uniform(-_OFFSET_TOL, _OFFSET_TOL))
            angle = base_angle + float(rng.uniform(-_ANGLE_STEP / 2.0,
                                                   _ANGLE_STEP / 2.0))
        else:
            # Below the apex (negative), so the corrective action is "up", as
            # in Figure 3. Half the unsolved scenes already pass through the
            # apex and only need rotating.
            offset = float(rng.choice([0.0, -rng.uniform(0.12, 0.36)]))
            angle = base_angle + float(rng.uniform(-0.9, 0.9))
            if abs(offset) < 1e-6 and abs(angle - base_angle) < 0.12:
                angle = base_angle + 0.4      # do not accidentally emit a solved one
        scene = {"kind": "geometry", "tri": [apex, left, right],
                 "line_offset": offset, "line_angle": angle,
                 "base_angle": base_angle, "annotated": True}
        out.append(Example(inp=Content.specific_sketch(scene, caption=f"triangle{i}")))
    return out


@generator(
    "norm_band_estimates",
    doc="Instances that are STATEMENTS about function space norms, not "
        "numbers. Each is a band estimate ||grad^dv P_M u||_{L^p} <= C M^fr "
        "q^sc together with an integrability index to move it to, which is "
        "the question `bernstein_uniform` and its siblings answer. The "
        "point of the generator is that the instance space is a space of "
        "written-down estimates: discovery applies wherever concepts can be "
        "written down, and a norm space is one of those.",
    params={"min_dim": "smallest spatial dimension to draw",
            "max_dim": "largest spatial dimension to draw",
            "max_derivatives": "largest derivative count to draw",
            "ip_to": "the integrability index the rule under test moves to, "
                     "as a string like '0' or '1/2'; instances are drawn "
                     "only where that rule's own guard admits them"})
def gen_norm_band_estimates(n: int, rng, max_dim: int = 2,
                            max_derivatives: int = 2,
                            ip_to: str = "0", min_dim: int = 1) -> list[Example]:
    from fractions import Fraction

    from .normcalc import est

    # Quarters, so the forced exponent d*(ip - ip_to) lands on a multiple
    # of 1/4 for d <= 2 and the oracle's measured slope can be snapped to
    # an exact rational without the snap doing the work.
    target = Fraction(ip_to)
    # Only indices the rule's guard admits: `bernstein_uniform` requires
    # 1/p > 0 and `bernstein_to_energy` requires 1/p > 1/2, so drawing
    # below the target would fill the set with instances the rule declines
    # and measure nothing at all.
    grid = [Fraction(k, 4) for k in range(0, 5) if Fraction(k, 4) > target]
    if not grid:
        raise ValueError(f"no integrability index above {target}")
    out: list[Example] = []
    for _ in range(n):
        d = int(rng.integers(min_dim, max_dim + 1))
        dv = int(rng.integers(0, max_derivatives + 1))
        ip = grid[int(rng.integers(0, len(grid)))]
        fr = Fraction(int(rng.integers(-4, 1)))
        sc = Fraction(int(rng.integers(0, 3)))
        cell = est(d=d, dv=dv, ip=ip, fr=fr, sc=sc)
        out.append(Example(
            inp=Content.abstract(cell),
            meta={"d": d, "dv": dv, "ip": ip, "ip_to": target,
                  "fr": fr, "sc": sc}))
    return out
