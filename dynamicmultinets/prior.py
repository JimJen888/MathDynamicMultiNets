"""
Prior knowledge: the rules the machine starts with.

These are the "known" arrows in Figure 2 -- rendering, exact evaluation, the
decimal split, the substitution step -- plus the sketch-space update used by the
robotic architecture. Everything here is exact and symbolic; nothing here is
learned. They matter for two separate reasons:

  1. A learned rule is only useful in a chain with rules that can compute. The
     distributive rewrite gets you from `12*30` to `10*30+2*30`; something has
     to actually add.
  2. They are the ground the machine verifies against. `eval_arith` gives a
     second, independent way to compute what a learned rule claims, which is
     what "facts reasoned by other existing rules" means in practice.

Arithmetic evaluation goes through `ast`, restricted to integer + - * and
parentheses. Not `eval`: expressions on the tape are produced by a NETWORK, and
a mis-decoded slot must fail as "not applicable", never execute.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Callable

from .rules import PythonRule, Rule, RuleLibrary, TableRule
from .tapes import ABSTRACT, SPECIFIC, Content

PRIOR_RULES: dict[str, Callable[[], Rule]] = {}


def prior_rule(name: str):
    """Register a factory so a saved library can rebuild built-ins by name."""

    def wrap(factory: Callable[[], Rule]) -> Callable[[], Rule]:
        PRIOR_RULES[name] = factory
        return factory

    return wrap


# ---------------------------------------------------------------------------
# Exact integer arithmetic
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult)


def eval_int_expression(text: str) -> int:
    """Evaluate an integer expression over + - * and parentheses, or raise.

    Deliberately narrow: this is the machine's definition of calculation in the
    abstract domain, and widening it (division, power, names) would let a
    mis-read tape cell mean something surprising.
    """
    tree = ast.parse(text.replace("x", "*").replace("X", "*"), mode="eval")

    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        raise ValueError(f"not an integer arithmetic expression: {text!r}")

    return ev(tree)


@prior_rule("eval_arith")
def make_eval_arith() -> PythonRule:
    def fn(c: Content) -> Content | None:
        return Content.abstract(str(eval_int_expression(c.text)))

    return PythonRule(
        "eval_arith", fn, ABSTRACT, ABSTRACT,
        description="evaluate an integer expression over + - * exactly",
        source="eval_int(+,-,*,parens)",
    )


@prior_rule("mul_by_definition")
def make_mul_by_definition() -> PythonRule:
    """`a * b` computed as b added to itself a times -- multiplication BY ITS
    DEFINITION, not by a rule. This is the bedrock the distributive rule is
    discovered against and verified against: it is slow, it only works for
    small operands, and it presupposes nothing."""

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*(\d+)\s*[*x]\s*(\d+)\s*", c.text)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if a > 400:                     # repeated addition has to stay honest
            return None
        total = 0
        for _ in range(a):
            total += b
        return Content.abstract(str(total), derivation="repeated addition")

    return PythonRule(
        "mul_by_definition", fn, ABSTRACT, ABSTRACT,
        description="a*b as b added a times (the definition of multiplication)",
        source="repeated_addition(a,b)",
    )


# ---------------------------------------------------------------------------
# Structural rewrites known in advance
# ---------------------------------------------------------------------------
@prior_rule("decimal_split")
def make_decimal_split() -> PythonRule:
    """`12 * 30` -> `(10+2) * 30`: write the left operand in place-value form.

    The known abstract-domain rule of Figure 2's second row. It creates the
    structure the distributive rule then acts on, and it is the reason the
    machine can attack an arbitrarily large product with a 9x9 table.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", c.text)
        if not m:
            return None
        a, b = m.group(1), m.group(2)
        if len(a) < 2:
            return None
        terms = [f"{int(d) * 10 ** (len(a) - 1 - i)}" for i, d in enumerate(a)]
        terms = [t for t in terms if t != "0"]
        if len(terms) < 2:
            return None
        return Content.abstract(f"({'+'.join(terms)})*{b}")

    return PythonRule(
        "decimal_split", fn, ABSTRACT, ABSTRACT,
        description="write a multi-digit factor as a sum of place values",
        source="a*b -> (a1+a2+..)*b",
    )


@prior_rule("distribute_symbolic")
def make_distribute_symbolic() -> PythonRule:
    """`(a+b)*c` -> `a*c+b*c`, as an identity the machine already holds.

    Kept alongside the LEARNED distributive rule on purpose. The learned one is
    what the paper demonstrates the machine acquiring from experiments in the
    specific domain; this one is the independent check that says whether what
    it acquired is right, and the conciseness pass later gets to notice that
    two rules with identical behaviour is one rule too many.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*\(([^()]+)\)\s*\*\s*(\d+)\s*", c.text)
        if not m:
            return None
        terms = [t for t in m.group(1).split("+") if t.strip()]
        if len(terms) < 2:
            return None
        return Content.abstract("+".join(f"{t.strip()}*{m.group(2)}" for t in terms))

    return PythonRule(
        "distribute_symbolic", fn, ABSTRACT, ABSTRACT,
        description="(a+b)*c = a*c+b*c",
        source="(a+b)*c -> a*c+b*c",
    )


@prior_rule("decimal_split_right")
def make_decimal_split_right() -> PythonRule:
    """`40 * 19` -> `40 * (10+9)`: the same move on the OTHER factor.

    The mirror of `decimal_split`, and it is here because its absence was a gap
    rather than a decision. Splitting the left factor turns `46*19` into
    `40*19+6*19`, and then neither part can be taken further: `40*19` has a
    single non-zero place, so the left-hand rule declines it and the
    decomposition stops one level above the times table. Multiplication is
    commutative and nothing about place value prefers one side, so a machine
    that can only split on the left cannot finish a decomposition it started.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", c.text)
        if not m:
            return None
        a, b = m.group(1), m.group(2)
        if len(b) < 2:
            return None
        terms = [f"{int(d) * 10 ** (len(b) - 1 - i)}" for i, d in enumerate(b)]
        terms = [t for t in terms if t != "0"]
        if len(terms) < 2:
            return None
        return Content.abstract(f"{a}*({'+'.join(terms)})")

    return PythonRule(
        "decimal_split_right", fn, ABSTRACT, ABSTRACT,
        description="write the RIGHT factor as a sum of place values",
        source="a*b -> a*(b1+b2+..)",
    )


@prior_rule("distribute_symbolic_right")
def make_distribute_symbolic_right() -> PythonRule:
    """`a*(b+c)` -> `a*b+a*c`. The mirror identity, for the same reason."""

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*(\d+)\s*\*\s*\(([^()]+)\)\s*", c.text)
        if not m:
            return None
        terms = [t for t in m.group(2).split("+") if t.strip()]
        if len(terms) < 2:
            return None
        return Content.abstract("+".join(f"{m.group(1)}*{t.strip()}" for t in terms))

    return PythonRule(
        "distribute_symbolic_right", fn, ABSTRACT, ABSTRACT,
        description="a*(b+c) = a*b+a*c",
        source="a*(b+c) -> a*b+a*c",
    )


@prior_rule("substitute_equalities")
def make_substitute_equalities() -> PythonRule:
    """Apply `x=y` facts to the last statement on the cell.

    The final step of the geometry proof: given `A1=B1, A2=B2,
    A1+A3+A2=180`, rewrite the sum into `B1+A3+B2=180`. Substitution is a rigid
    formal-logic operation, so it belongs in the abstract domain and is exact.
    """

    def fn(c: Content) -> Content | None:
        parts = [p.strip() for p in c.text.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        goal, subs = parts[-1], {}
        for p in parts[:-1]:
            if p.count("=") == 1:
                lhs, rhs = (s.strip() for s in p.split("="))
                if lhs.isidentifier() and rhs.isidentifier():
                    subs[lhs] = rhs
        if not subs:
            return None
        out = re.sub(r"[A-Za-z]\w*", lambda m: subs.get(m.group(0), m.group(0)), goal)
        return Content.abstract(out, derivation="substitution")

    return PythonRule(
        "substitute_equalities", fn, ABSTRACT, ABSTRACT,
        description="substitute proven equalities into the goal statement",
        source="x=y, P(x) -> P(y)",
    )


# ---------------------------------------------------------------------------
# Divisors, and the two RH criteria that are decidable one integer at a time
# ---------------------------------------------------------------------------
# Robin (1984) and Lagarias (2002) are each EQUIVALENT to the Riemann
# hypothesis, and that equivalence is prior knowledge here in exactly the sense
# the distributive identity is: a result the machine is given, not one it
# discovers. What the equivalence buys is the thing this architecture needs and
# the zeta function does not otherwise offer -- a statement about zeros of an
# analytic function, re-expressed as a predicate over the integers that is
# EXACTLY DECIDABLE at each n. One integer at a time, RH becomes a cell the
# machine can write down, and a chain of these rules is a proof about that cell.
#
# What no rule below does, and what nothing in this package can do, is discharge
# the quantifier. Robin's criterion says the inequality holds for EVERY n>5040;
# these rules settle it for whichever n is on the tape. The gap between those
# two is not a missing rule, it is the whole of the problem, and `run_riemann.py`
# puts it where the machine states it as a transfer claim rather than hiding it
# inside a chain that reports a confidence.
EULER_GAMMA = 0.5772156649015328606
EXP_GAMMA = 1.7810724179901979852     # e^gamma, the constant in Robin's bound


def divisors_of(n: int) -> list[int]:
    """Every divisor of n, found by trial division.

    BY DEFINITION, in the sense `mul_by_definition` means it: a divisor is a
    number that divides n, and this checks exactly that. It deliberately does
    not factor n and multiply out (p^(a+1)-1)/(p-1) over the prime powers --
    that formula is a theorem about unique factorization, and using it here
    would make the machine's arithmetic bedrock depend on a result it has never
    been given. The sqrt bound is not such a result: it only says divisors come
    in pairs d, n/d, which is immediate from the definition.
    """
    if n < 1:
        raise ValueError(f"divisors are defined for positive integers, not {n}")
    out: list[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            if d != n // d:
                out.append(n // d)
        d += 1
    return sorted(out)


def sigma(n: int) -> int:
    """sigma(n): the sum of the divisors of n."""
    return sum(divisors_of(n))


def harmonic(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


@prior_rule("divisor_sum")
def make_divisor_sum() -> PythonRule:
    """`5040` -> `sigma(5040)=19344`. Exact, and the only arithmetic in the
    chain that actually touches the integer.

    The output carries n as well as sigma(n) because every criterion below
    needs both, and a cell that dropped n would force the next rule to invert
    the sum -- which is not merely expensive, it is not a function.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*(\d+)\s*", c.text)
        if not m:
            return None
        n = int(m.group(1))
        if not 1 <= n <= 10 ** 7:       # trial division has to stay honest
            return None
        return Content.abstract(f"sigma({n})={sigma(n)}",
                                derivation="sum of divisors by trial division")

    return PythonRule(
        "divisor_sum", fn, ABSTRACT, ABSTRACT,
        description="sum the divisors of n, by definition",
        source="n -> sigma(n)=S",
    )


@prior_rule("robin_ratio")
def make_robin_ratio() -> PythonRule:
    """`sigma(5040)=19344` -> `robin_ratio(5040)=1.790096`.

    The quantity Robin's criterion is about: sigma(n) / (n * ln ln n), which the
    hypothesis asserts stays below e^gamma for every n>5040. Splitting the ratio
    off from the comparison is not ceremony -- the ratio is the interesting
    number, it is what a reader wants to see next to the verdict, and keeping it
    in its own cell means a chain that ends in the wrong verdict can be
    diagnosed one step before the end.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*sigma\((\d+)\)=(\d+)\s*", c.text)
        if not m:
            return None
        n, s = int(m.group(1)), int(m.group(2))
        if n < 3:                       # ln ln n <= 0; the criterion says nothing
            return None
        ratio = s / (n * math.log(math.log(n)))
        return Content.abstract(f"robin_ratio({n})={ratio:.6f}",
                                derivation="sigma(n)/(n*lnln n)")

    return PythonRule(
        "robin_ratio", fn, ABSTRACT, ABSTRACT,
        description="the Robin quotient sigma(n)/(n ln ln n)",
        source="sigma(n)=S -> robin_ratio(n)=r",
    )


@prior_rule("robin_decide")
def make_robin_decide() -> PythonRule:
    """`robin_ratio(5040)=1.790096` -> `robin_fails`.

    5040 really does fail, and it is the largest integer known to: Robin proved
    that RH holds if and only if 5040 is the LAST exception. So a machine that
    reports `robin_holds` for every n it is given above 5040 has reproduced the
    evidence for RH, and a machine that ever reported `robin_fails` above 5040
    would have refuted it. Both outcomes are one cell wide, which is the point
    of coming through this criterion at all.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*robin_ratio\((\d+)\)=([\d.]+)\s*", c.text)
        if not m:
            return None
        verdict = "robin_holds" if float(m.group(2)) < EXP_GAMMA else "robin_fails"
        return Content.abstract(verdict, derivation=f"ratio vs e^gamma={EXP_GAMMA:.6f}")

    return PythonRule(
        "robin_decide", fn, ABSTRACT, ABSTRACT,
        description="compare the Robin quotient against e^gamma",
        source="robin_ratio(n)=r -> robin_holds|robin_fails",
    )


@prior_rule("lagarias_decide")
def make_lagarias_decide() -> PythonRule:
    """`sigma(5040)=19344` -> `lagarias_holds`.

    Lagarias's criterion -- sigma(n) <= H_n + exp(H_n) ln(H_n) for every n>=1 --
    is equivalent to RH as well, and it is here as an INDEPENDENT second route
    to the same conclusion rather than a spare -- the "facts reasoned by other
    existing rules" the oracle documentation asks for.

    Where the two agree is itself informative and was measured, not assumed.
    Above 5040 they agree on every integer checked (3..40000: zero
    disagreements). At or below 5040 they disagree on exactly 26 values -- 3, 4,
    5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 30, 36, 48, 60, 72, 84, 120, 180, 240,
    360, 720, 840, 2520, 5040 -- which is Robin's exceptional set with n=2
    omitted, because ln ln 2 is negative and `robin_ratio` declines it rather
    than emit a ratio whose sign inverts the comparison. That divergence is the
    expected one and it is what makes the pair a real cross-check: Lagarias has
    NO exceptional set, so agreement above 5040 is a conclusion the two routes
    reach separately, while a disagreement up there would mean one of these
    rules is wrong.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*sigma\((\d+)\)=(\d+)\s*", c.text)
        if not m:
            return None
        n, s = int(m.group(1)), int(m.group(2))
        if n < 2:                       # H_1 = 1, ln H_1 = 0; the bound is tight
            return None
        h = harmonic(n)
        verdict = "lagarias_holds" if s <= h + math.exp(h) * math.log(h) else "lagarias_fails"
        return Content.abstract(verdict, derivation="sigma(n) vs H_n+exp(H_n)ln H_n")

    return PythonRule(
        "lagarias_decide", fn, ABSTRACT, ABSTRACT,
        description="compare sigma(n) against Lagarias's harmonic bound",
        source="sigma(n)=S -> lagarias_holds|lagarias_fails",
    )


# ---------------------------------------------------------------------------
# Crossing between the domains
# ---------------------------------------------------------------------------
@prior_rule("render")
def make_render() -> PythonRule:
    """abstract -> specific. The built-in write function of Figure 1: put the
    symbols on the screen. No learning involved -- this direction is easy, and
    that asymmetry (easy to draw, hard to read) is why the reader is a net."""

    def fn(c: Content) -> Content | None:
        return Content.specific_text(c.text, source_rule="render")

    return PythonRule("render", fn, ABSTRACT, SPECIFIC,
                      description="draw symbols onto the specific tape",
                      source="render(text)->image")


@prior_rule("transcribe_unsafe")
def make_transcribe_unsafe() -> PythonRule:
    """specific -> abstract WITHOUT reading the pixels: it copies the caption
    the renderer stored.

    This is a deliberate cheat, provided only as a baseline/fallback so a
    workflow can run end to end before the reader net is trained. It is NOT
    trusted, it is charged as if it were exact, and any chain that uses it is
    marked in `library_report` -- because a machine that can only "read" what
    it already wrote has not crossed from the specific domain to the abstract
    one at all, which is the entire claim of the architecture.
    """

    def fn(c: Content) -> Content | None:
        if c.meta.get("observed"):      # a camera frame has no caption to cheat with
            return None
        return Content.abstract(c.text, derivation="transcribed caption")

    r = PythonRule("transcribe_unsafe", fn, SPECIFIC, ABSTRACT,
                   description="BASELINE ONLY: copy the renderer's caption, no perception",
                   source="caption(image)->text", exact=False)
    r.trusted = False
    return r


@prior_rule("sketch_action")
def make_sketch_action() -> PythonRule:
    """specific -> specific: update the sketch's parameters and redraw.

    The "sketch space updating module" of Appendix A. The decision of WHICH
    update to make is a perception problem and belongs to a learned rule; the
    update itself is arithmetic on scene parameters and belongs here. Chaining
    the two is what a specific->specific rule actually is in this machine.

    The action to apply is read from `content.meta["action"]`, which is where
    the deciding rule writes it.
    """

    def fn(c: Content) -> Content | None:
        scene = c.meta.get("scene")
        action = c.meta.get("action")
        if not scene or not action or action == "done":
            return None
        scene = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                 for k, v in scene.items()}
        step = 0.12
        if action == "move_up":
            scene["line_offset"] = scene.get("line_offset", 0.0) + step
        elif action == "rotate_cw":
            scene["line_angle"] = scene.get("line_angle", 0.0) - 0.15
        elif action == "rotate_ccw":
            scene["line_angle"] = scene.get("line_angle", 0.0) + 0.15
        else:
            return None
        out = Content.specific_sketch(scene, caption=f"{c.text}|{action}")
        out.meta["action_applied"] = action
        return out

    return PythonRule("sketch_action", fn, SPECIFIC, SPECIFIC,
                      description="apply a geometric action to the sketch and redraw",
                      source="scene,action->scene'")


# ---------------------------------------------------------------------------
# Memorized tables
# ---------------------------------------------------------------------------
def make_times_table(limit: int = 9) -> TableRule:
    """The 9x9 multiplication table -- "learned or memorized" in Figure 2.

    Kept as a TableRule rather than folded into `eval_arith` so the library can
    see what memorization costs (81 entries of bits) next to what a rule costs.
    """
    table = {f"{a}*{b}": str(a * b)
             for a in range(limit + 1) for b in range(limit + 1)}
    return TableRule(f"times_table_{limit}", table, ABSTRACT,
                     description=f"memorized products up to {limit}x{limit}")


PRIOR_RULES["times_table_9"] = lambda: make_times_table(9)


def install_prior_rules(library: RuleLibrary, include: list[str] | None = None) -> RuleLibrary:
    """Give a fresh machine what it already knows."""
    names = include or list(PRIOR_RULES)
    for name in names:
        rule = PRIOR_RULES[name]()
        library.add(rule, replace=True)
    return library
