"""
Proof search: finding a chain of mapping rules from what is written to what is
to be shown.

"Logical reasoning thus means the transfer of mapping rules." Once that is the
definition, proving something is a graph search whose nodes are tape cells and
whose edges are rules -- and, crucially, the search is allowed to cross domains,
so a chain may render symbols into a picture, decide something by looking, and
read the conclusion back out. That is the part a purely symbolic prover cannot
do, and it is why `Proof.crosses_domains` is reported.

Search order is best-first on

    g(node) = steps so far + (bits of the rules used) / BITS_PER_STEP

so short chains of cheap, already-known rules are tried before long ones or
ones that lean on an expensive learned rule. That is the same currency the
conciseness objective in compose.py minimises, which is deliberate: the proof
the machine finds first is the one it would most like to keep.

Confidence multiplies along the chain. A ten-step proof whose links each verify
at 0.99 is a 0.90 proof, and `Proof.confidence` says so rather than reporting
that every step "worked".
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .rules import CompositeRule, Rule, RuleLibrary
from .tapes import Content

BITS_PER_STEP = 1000.0     # how many description bits one rule application is worth


def normalize(text: str) -> str:
    return text.replace(" ", "").strip()


@dataclass
class ProofStep:
    rule: str
    before: str
    after: str
    domain_in: str
    domain_out: str

    def line(self) -> str:
        arrow = f"{self.domain_in[:4]}->{self.domain_out[:4]}"
        return f"  --{self.rule} [{arrow}]-->  {self.after!r}"


@dataclass
class Proof:
    found: bool
    start: str
    target: str
    steps: list[ProofStep] = field(default_factory=list)
    final: Content | None = None
    nodes_expanded: int = 0
    confidence: float = 1.0
    cost_bits: float = 0.0
    note: str = ""

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def crosses_domains(self) -> bool:
        return any(s.domain_in != s.domain_out for s in self.steps)

    def rule_names(self) -> list[str]:
        return [s.rule for s in self.steps]

    def as_text(self) -> str:
        head = (f"{'PROVED' if self.found else 'NOT PROVED'}: {self.start!r} => "
                f"{self.target!r}  ({self.length} steps, confidence "
                f"{self.confidence:.4f}, {self.nodes_expanded} nodes expanded)")
        body = "\n".join([f"  {self.start!r}"] + [s.line() for s in self.steps])
        tail = f"\n  note: {self.note}" if self.note else ""
        return head + "\n" + body + tail


def search(
    library: RuleLibrary,
    start: Content,
    target: str | Callable[[Content], bool],
    max_depth: int = 6,
    max_nodes: int = 4000,
    rules: Sequence[str] | None = None,
    trusted_only: bool = True,
    min_confidence: float = 0.0,
    target_domain: str = "abstract",
) -> Proof:
    """Best-first search for a rule chain from `start` to `target`.

    `target` is a string to reach (compared modulo whitespace) or a predicate
    over the resulting cell -- the latter is what you want when the goal is a
    shape ("any cell whose text is a bare integer") rather than a literal.

    `trusted_only` defaults to True: an unverified rule may not appear in a
    proof. Turn it off to explore what WOULD be provable if a candidate rule
    held, which is a reasonable thing for the controller to ask before deciding
    whether that rule is worth verifying.
    """
    pool: list[Rule] = [library.get(n) for n in rules] if rules else list(library)
    if trusted_only:
        pool = [r for r in pool if r.trusted]
    if not pool:
        return Proof(False, start.text, str(target), note="no usable rules in the library")

    # A string target must be reached IN A DOMAIN. Without that, a picture of
    # "47*83" would count as having proved the symbols "47*83" -- the cell's
    # caption would be doing the work that reading it is supposed to do, and
    # every perception task would trivially succeed in zero steps.
    goal: Callable[[Content], bool] = (
        target if callable(target)
        else (lambda c, t=normalize(target), d=target_domain:
              normalize(c.text) == t and c.domain == d)
    )
    target_text = "<predicate>" if callable(target) else target

    if goal(start):
        return Proof(True, start.text, target_text, note="already proved")

    counter = itertools.count()
    # (priority, tiebreak, content, path, confidence, bits)
    frontier: list[tuple[float, int, Content, list[ProofStep], float, float]] = [
        (0.0, next(counter), start, [], 1.0, 0.0)
    ]
    seen: set[tuple[str, str]] = {(start.domain, normalize(start.text))}
    expanded = 0

    while frontier and expanded < max_nodes:
        _, _, cur, path, conf, bits = heapq.heappop(frontier)
        expanded += 1
        if len(path) >= max_depth:
            continue

        for rule in pool:
            if rule.domain_in != cur.domain:
                continue
            nxt = rule.apply(cur)
            if nxt is None:
                continue
            key = (nxt.domain, normalize(nxt.text))
            if key in seen:
                continue
            seen.add(key)

            step = ProofStep(rule.name, cur.text, nxt.text, rule.domain_in, rule.domain_out)
            new_path = path + [step]
            new_conf = conf * rule.confidence()
            new_bits = bits + rule.cost_bits()
            if new_conf < min_confidence:
                continue
            if goal(nxt):
                return Proof(True, start.text, target_text, new_path, nxt,
                             expanded, new_conf, new_bits)
            priority = len(new_path) + new_bits / BITS_PER_STEP
            heapq.heappush(frontier, (priority, next(counter), nxt, new_path,
                                      new_conf, new_bits))

    return Proof(False, start.text, target_text, nodes_expanded=expanded,
                 note=("node budget exhausted" if expanded >= max_nodes
                       else "search space exhausted"))


def proof_to_rule(library: RuleLibrary, proof: Proof, name: str,
                  description: str = "") -> CompositeRule:
    """Keep a found proof as a single named rule.

    This is the machine learning something in the ordinary sense of the word:
    it searched once, and from now on the derivation is one move. The composite
    is trusted only if every member was -- a shortcut may not launder an
    unverified step into a trusted one.
    """
    if not proof.found:
        raise ValueError("cannot make a rule out of a failed proof")
    members = [library.get(s.rule) for s in proof.steps]
    rule = CompositeRule(name, members,
                         description or f"{proof.start!r} => {proof.target!r} in "
                                        f"{proof.length} steps")
    rule.trusted = all(m.trusted for m in members)
    library.add(rule, replace=True)
    return rule
