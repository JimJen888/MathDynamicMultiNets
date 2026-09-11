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
    #: Confidence over the steps that have evidence behind them: exact rules
    #: contribute their 1.0, checked rules their measured accuracy, and
    #: unmeasured rules contribute nothing rather than the prior 1/2.
    measured_confidence: float = 1.0
    #: How many steps used a rule nothing has checked. `confidence` charges
    #: each of those 1/2, which compounds ignorance into a number that looks
    #: like a probability; this is the count that number is standing in for.
    unmeasured: int = 0
    #: How many steps used a rule that leans on something it does not
    #: establish -- a classical inequality, or a hypothesis someone
    #: asserted. Counted separately from `unmeasured` because such a rule
    #: may be perfectly reliable AS A REWRITE and still carry a premise
    #: nobody here checked. A chain with any of these is a proof modulo
    #: its assumptions, and the number is what stops it reading otherwise.
    assumed: int = 0

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def crosses_domains(self) -> bool:
        return any(s.domain_in != s.domain_out for s in self.steps)

    def rule_names(self) -> list[str]:
        return [s.rule for s in self.steps]

    def evidence(self) -> str:
        """The confidence, said in a way that does not overstate it.

        A chain through steps nobody has checked has no confidence to
        report, and saying 0.0005 invites the reader to treat compounded
        ignorance as near-certain failure. What can be said is how reliable
        the measured part is and how many steps are not measured at all.
        """
        if not self.unmeasured and not self.assumed:
            return f"confidence {self.confidence:.4f}"
        if not self.unmeasured:
            return (f"confidence {self.confidence:.4f}, resting on "
                    f"{self.assumed} assumed step"
                    f"{'' if self.assumed == 1 else 's'}")
        measured = self.length - self.unmeasured
        if measured <= 0:
            # Saying "1.0000 over the measured steps" when there are none is
            # worse than saying nothing, so say nothing.
            return f"nothing measured, {self.unmeasured} unmeasured steps"
        plural = "" if measured == 1 else "s"
        tail = f", {self.assumed} assumed" if self.assumed else ""
        return (f"confidence {self.measured_confidence:.4f} over "
                f"{measured} measured step{plural}, "
                f"{self.unmeasured} unmeasured{tail}")

    def as_text(self) -> str:
        head = (f"{'PROVED' if self.found else 'NOT PROVED'}: {self.start!r} => "
                f"{self.target!r}  ({self.length} steps, {self.evidence()}, "
                f"{self.nodes_expanded} nodes expanded)")
        body = "\n".join([f"  {self.start!r}"] + [s.line() for s in self.steps])
        tail = f"\n  note: {self.note}" if self.note else ""
        return head + "\n" + body + tail


# ---------------------------------------------------------------------------
# Saturation: the search a conjunction needs
# ---------------------------------------------------------------------------
def saturate(
    library,
    givens: Sequence[str],
    target: str,
    max_rounds: int = 8,
    max_known: int = 600,
    trusted_only: bool = True,
) -> Proof:
    """Derive a SET of cells until the target appears, rather than a path.

    `search` walks one cell to the next, which is the right shape for a
    calculation and the wrong one for a proof. A proof establishes several
    things and then collects them, and there is nowhere in a path to put
    the collecting step. So this keeps everything derived so far, applies
    every rule that fires, and repeats.

    The cost of that generality is real and is bounded here rather than
    hidden: every unary rule is tried against every known cell each round,
    so the work grows with the square of what has been derived. `max_known`
    stops it, `max_rounds` stops it, and a run that hits either says so in
    the note instead of reporting a clean failure.

    What comes back is a `Proof` whose steps are the sub-graph that
    actually reached the target, in an order where every premise precedes
    its use. Accounting is the same as anywhere else: measured, unmeasured
    and assumed are counted over the rules the sub-graph used, so a
    conclusion collected from assumed premises still says so.
    """
    known: dict[str, Content] = {}
    origin: dict[str, tuple[str, tuple[str, ...]]] = {}
    for text in givens:
        cell = Content.abstract(text.strip())
        known[cell.text] = cell

    rules = [library.get(n) for n in library.rules]
    if trusted_only:
        rules = [r for r in rules if r.trusted]
    joins = [r for r in rules if hasattr(r, "fires")]
    singles = [r for r in rules if not hasattr(r, "fires")]

    target_text = normalize(target)
    note = ""
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        grew = False
        for rule in singles:
            for text in list(known):
                out = rule.apply(known[text])
                if out is None or out.text in known:
                    continue
                known[out.text] = out
                origin[out.text] = (rule.name, (text,))
                grew = True
                if len(known) >= max_known:
                    note = (f"stopped at {max_known} derived cells; the target "
                            f"may be reachable with a larger budget")
                    break
            if note:
                break
        for rule in joins:
            if note:
                break
            out = rule.fires(known)
            if out is None or out.text in known:
                continue
            known[out.text] = out
            origin[out.text] = (rule.name, rule.premises)
            grew = True
        if any(normalize(t) == target_text for t in known):
            break
        if not grew or note:
            if not grew and not note:
                note = "no rule fired on anything derived; the set is closed"
            break

    reached = next((t for t in known if normalize(t) == target_text), None)
    if reached is None:
        return Proof(False, " + ".join(givens), target, nodes_expanded=len(known),
                     note=note or f"not derived in {rounds} rounds")

    # Walk back from the target, keeping only what it actually used.
    needed: list[str] = []

    def collect(text: str) -> None:
        if text in needed or text not in origin:
            return
        for premise in origin[text][1]:
            collect(premise)
        needed.append(text)

    collect(reached)

    steps: list[ProofStep] = []
    conf = measured = 1.0
    unmeasured = assumed = 0
    for text in needed:
        rule_name, premises = origin[text]
        rule = library.get(rule_name)
        steps.append(ProofStep(rule_name, " + ".join(premises), text,
                               rule.domain_in, rule.domain_out))
        conf *= rule.confidence()
        if rule.measured():
            measured *= rule.confidence()
        else:
            unmeasured += 1
        if getattr(rule, "assumes", ()):
            assumed += 1

    return Proof(True, " + ".join(givens), target, steps, known[reached],
                 len(known), conf, sum(library.get(origin[t][0]).cost_bits()
                                       for t in needed),
                 measured_confidence=measured, unmeasured=unmeasured,
                 assumed=assumed,
                 note=f"derived in {rounds} rounds from {len(givens)} givens")


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
    beam: int = 1,
) -> Proof:
    """Best-first search for a rule chain from `start` to `target`.

    `target` is a string to reach (compared modulo whitespace) or a predicate
    over the resulting cell -- the latter is what you want when the goal is a
    shape ("any cell whose text is a bare integer") rather than a literal.

    `trusted_only` defaults to True: an unverified rule may not appear in a
    proof. Turn it off to explore what WOULD be provable if a candidate rule
    held, which is a reasonable thing for the controller to ask before deciding
    whether that rule is worth verifying.

    `beam` is how many of a CHOOSING rule's ranked answers get expanded, and it
    DEFAULTS TO 1 because widening it was measured and does not work. The idea
    is tempting: a construction loop is a single path, so one wrong perception
    ends the proof with no branch to recover through, and the right action is
    usually the runner-up. But replaying each proof and asking the oracle
    whether the drawing it ends on genuinely licenses the conclusion tells a
    different story -- over 120 triangles, beam 1 proved 111 of which 94 were
    genuine, while beam 2 proved all 120 and beam 3 all 120, with the genuine
    count FALLING to 93 and 89 as the bogus count climbed from 17 to 27 to 31.

    A wider beam does not find more proofs. It finds more drawings, and among
    more drawings there are more that the reader misjudges, so the extra
    "successes" are the search hunting down its own perception's mistakes.
    That is worth stating plainly because the failure is invisible from the
    outside: those proofs terminate, cross domains, and report a confidence.
    Only replaying them against the oracle shows what they are.

    Keep this at 1 unless you have an independent check on the final drawing.
    Only rules that PROPOSE would ever get alternatives anyway -- see
    `NeuralRule.offers_alternatives` for why a rule writing into the abstract
    domain must never have its runner-up expanded.
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
    # (priority, tiebreak, content, path, confidence, bits, measured,
    #  unmeasured, assumed)
    frontier: list[tuple[float, int, Content, list[ProofStep], float, float,
                         float, int, int]] = [
        (0.0, next(counter), start, [], 1.0, 0.0, 1.0, 0, 0)
    ]
    seen: set[tuple[str, str]] = {(start.domain, normalize(start.text))}
    expanded = 0

    while frontier and expanded < max_nodes:
        (_, _, cur, path, conf, bits, measured, unmeasured,
         assumed) = heapq.heappop(frontier)
        expanded += 1
        if len(path) >= max_depth:
            continue

        for rule in pool:
            if rule.domain_in != cur.domain:
                continue
            for nxt, plausibility in rule.successors(cur, beam):
                key = (nxt.domain, normalize(nxt.text))
                if key in seen:
                    continue
                seen.add(key)

                step = ProofStep(rule.name, cur.text, nxt.text,
                                 rule.domain_in, rule.domain_out)
                new_path = path + [step]
                # The runner-up is a real possibility, not a free one: a chain
                # that leans on a second choice is less certain than one that
                # did not have to, and `plausibility` is 1.0 for the argmax so
                # nothing changes for a proof that never needed an alternative.
                new_conf = conf * rule.confidence() * plausibility
                new_bits = bits + rule.cost_bits()
                # A step nobody has checked contributes its count, not its
                # prior: compounding 1/2 several times says less than saying
                # how many steps are unmeasured.
                # A rule that names something it leans on but does not
                # establish is counted, however reliable its own rewrite is.
                new_assumed = assumed + (1 if getattr(rule, "assumes", ())
                                         else 0)
                if rule.measured():
                    new_measured = measured * rule.confidence() * plausibility
                    new_unmeasured = unmeasured
                else:
                    new_measured = measured * plausibility
                    new_unmeasured = unmeasured + 1
                if new_conf < min_confidence:
                    continue
                if goal(nxt):
                    return Proof(True, start.text, target_text, new_path, nxt,
                                 expanded, new_conf, new_bits,
                                 measured_confidence=new_measured,
                                 unmeasured=new_unmeasured,
                                 assumed=new_assumed)
                priority = len(new_path) + new_bits / BITS_PER_STEP
                heapq.heappush(frontier, (priority, next(counter), nxt, new_path,
                                          new_conf, new_bits, new_measured,
                                          new_unmeasured, new_assumed))

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
