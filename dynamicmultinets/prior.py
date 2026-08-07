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
