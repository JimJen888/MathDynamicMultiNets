"""
Verification: deciding whether a rule may be used in a proof.

A learned rule starts untrusted. It becomes trusted by agreeing, on examples it
was not trained on, with something the machine already believes -- either a
definitional oracle (multiplication as repeated addition), or a chain of rules
already trusted (the paper's "facts reasoned by other existing rules"). The
report records WHICH, because the two are not equally strong and a proof built
on the second inherits everything the first would have caught.

Two numbers come out and both are needed:

    accuracy   how often the rule agreed
    grounding  what it agreed WITH

An accuracy of 1.000 against `read_back` means the reader reproduced captions
the renderer wrote; an accuracy of 0.997 against `product_by_definition` means
the rewrite rule survived contact with arithmetic. The second is worth more,
and `report.strength` says so rather than leaving it to whoever reads the log.

The threshold matters for the same reason as the 0.99999^1000 argument in the
conclusions: a chain's confidence is the product of its links', so a rule that
verifies at 0.98 is not "nearly right", it is a rule that breaks a fifty-step
derivation more often than not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .dataset import Example, ExampleSet
from .oracles import ORACLES, label
from .rules import CompositeRule, Rule, RuleLibrary
from .tapes import Content

# How much a grounding source is worth as evidence, worst to best.
GROUNDING_STRENGTH = {
    "constructed": 0.2,     # the machine checked its own handwriting
    "derived": 0.7,         # sound only as far as the rules it leaned on
    "measured": 0.9,        # came from outside
    "definitional": 1.0,    # bottoms out in a definition
    "rule_chain": 0.7,
    "unknown": 0.0,
}


def normalize(text: str) -> str:
    """Compare expressions modulo whitespace. Deliberately not modulo
    arithmetic: a rewrite rule that returns a numerically equal but
    structurally different expression has not done what it claimed."""
    return text.replace(" ", "").strip()


def answer_text(content: Content | None) -> str:
    """What a rule's output should be compared against.

    Usually the cell's text. But a rule that EDITS a drawing outputs another
    drawing, whose text is a caption -- comparing that with an oracle's label
    would score every such rule at zero however well it decided. Those rules
    record what they decided in `meta['decision']`, and that is what gets
    checked.
    """
    if content is None:
        return ""
    return str(content.meta.get("decision", content.text))


@dataclass
class VerificationReport:
    rule: str
    n_checked: int = 0
    n_correct: int = 0
    grounding: str = "unknown"
    source: str = ""
    counterexamples: list[str] = field(default_factory=list)
    threshold: float = 0.99
    became_trusted: bool = False

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_checked if self.n_checked else 0.0

    @property
    def strength(self) -> float:
        """Accuracy discounted by how good the evidence was."""
        return self.accuracy * GROUNDING_STRENGTH.get(self.grounding, 0.0)

    def summary(self) -> str:
        if not self.n_checked:
            return (f"{self.rule}: NO CHECKS RAN against {self.source} -- every "
                    "example was declined, so this says nothing. Usually the "
                    "reference or the rule expects a different input domain.")
        head = (f"{self.rule}: {self.accuracy:.4f} over {self.n_checked} checks "
                f"vs {self.source} [{self.grounding}], strength {self.strength:.3f}")
        if self.counterexamples:
            head += "\n  counterexamples: " + "; ".join(self.counterexamples[:5])
        head += f"\n  {'TRUSTED' if self.became_trusted else 'not trusted'} " \
                f"(threshold {self.threshold})"
        return head


def _check(rule: Rule, examples: Sequence[Example]) -> tuple[int, int, list[str]]:
    n = correct = 0
    bad: list[str] = []
    for ex in examples:
        if not ex.labeled:
            continue
        n += 1
        got = rule.apply(ex.inp)
        want = normalize(answer_text(ex.out))
        if got is not None and normalize(answer_text(got)) == want:
            correct += 1
        elif len(bad) < 32:
            bad.append(f"{ex.inp.text!r} -> {answer_text(got)!r} (want {want!r})")
    return n, correct, bad


def verify_rule(
    library: RuleLibrary,
    rule_name: str,
    example_set: ExampleSet,
    oracle_name: str | None = None,
    threshold: float = 0.99,
    relabel: bool = True,
) -> VerificationReport:
    """Check a rule against oracle-labeled examples it was not trained on.

    `relabel=True` re-runs the oracle over the set, so a verification set can be
    freshly generated (a different seed, a harder tail) and labeled in one call
    -- which is what "verified by more examples outside the training data"
    requires.
    """
    rule = library.get(rule_name)
    oracle_name = oracle_name or example_set.oracle
    if not oracle_name:
        raise ValueError("verification needs an oracle: pass one or label the set first")
    if relabel or not example_set.labeled:
        label(example_set, oracle_name)

    n, correct, bad = _check(rule, example_set.examples)
    report = VerificationReport(
        rule=rule_name, n_checked=n, n_correct=correct,
        grounding=ORACLES[oracle_name].kind, source=oracle_name,
        counterexamples=bad, threshold=threshold,
    )
    rule.stats.merge(n, correct, oracle_name, bad)
    if n and report.accuracy >= threshold:
        rule.trusted = True
        report.became_trusted = True
    return report


def verify_against_rules(
    library: RuleLibrary,
    rule_name: str,
    reference_chain: Sequence[str],
    example_set: ExampleSet,
    threshold: float = 0.99,
) -> VerificationReport:
    """Check a rule against a chain of rules the machine already trusts.

    This is how a learned rule gets checked with no oracle in sight: the
    distributive rewrite learned from pictures is compared with
    decimal_split -> distribute_symbolic applied to the same input. Agreement
    is evidence; it is weaker than a definitional check, and `grounding` says
    so. If any link in the reference chain is itself untrusted, the check is
    refused outright rather than quietly producing a number.
    """
    rule = library.get(rule_name)
    members = [library.get(n) for n in reference_chain]
    untrusted = [m.name for m in members if not m.trusted]
    if untrusted:
        raise ValueError(
            f"cannot verify against untrusted rules {untrusted}; verify those first"
        )
    reference = CompositeRule(f"ref[{'+'.join(reference_chain)}]", members)

    n = correct = 0
    bad: list[str] = []
    for ex in example_set.examples:
        want = reference.apply(ex.inp)
        if want is None:
            continue                       # the reference has nothing to say here
        n += 1
        got = rule.apply(ex.inp)
        if got is not None and normalize(answer_text(got)) == normalize(answer_text(want)):
            correct += 1
        elif len(bad) < 32:
            bad.append(f"{ex.inp.text!r} -> {answer_text(got)!r} "
                       f"(reference {answer_text(want)!r})")

    report = VerificationReport(
        rule=rule_name, n_checked=n, n_correct=correct, grounding="rule_chain",
        source=reference.name, counterexamples=bad, threshold=threshold,
    )
    rule.stats.merge(n, correct, reference.name, bad)
    if n and report.accuracy >= threshold:
        rule.trusted = True
        report.became_trusted = True
    return report


def topk_report(rule, example_set: ExampleSet, k: int = 3) -> dict:
    """Top-1 and top-k accuracy for a rule that makes one choice.

    Reported separately from `verify_rule` because they answer different
    questions. Verification asks "is this rule's answer right", which is what
    trust in a PROOF requires. A planner asks "is the right answer near the top
    of the list", because it will try them in order and something downstream
    vets each one -- detourNet reports top-3 for exactly this reason. A rule can
    be operationally excellent and still not belong in a proof.
    """
    hits1 = hitsk = n = 0
    per_class: dict[str, list[int]] = {}
    for ex in example_set.examples:
        if not ex.labeled:
            continue
        want = normalize(answer_text(ex.out))
        ranked = [name for name, _ in rule.rank_options(ex.inp, k)]
        n += 1
        hits1 += ranked[:1] == [want]
        hitsk += want in ranked
        bucket = per_class.setdefault(want, [0, 0])
        bucket[0] += 1
        bucket[1] += ranked[:1] == [want]
    return {
        "n": n,
        "top1": hits1 / max(n, 1),
        f"top{k}": hitsk / max(n, 1),
        "per_class": {c: (ok, tot) for c, (tot, ok) in sorted(per_class.items())},
    }


def equivalent(library: RuleLibrary, a: str, b: str, example_set: ExampleSet,
               tolerance: float = 0.0) -> tuple[bool, float, list[str]]:
    """Do two rules agree everywhere the first applies?

    The test the conciseness pass runs before proposing that one of them be
    dropped. Asymmetric on purpose: inputs where `a` declines are not evidence
    against `b`, so only cases `a` handles are counted.
    """
    ra, rb = library.get(a), library.get(b)
    n = agree = 0
    diffs: list[str] = []
    for ex in example_set.examples:
        out_a = ra.apply(ex.inp)
        if out_a is None:
            continue
        n += 1
        out_b = rb.apply(ex.inp)
        if out_b is not None and normalize(answer_text(out_a)) == normalize(answer_text(out_b)):
            agree += 1
        elif len(diffs) < 16:
            diffs.append(f"{ex.inp.text!r}: {a}->{answer_text(out_a)!r} vs "
                         f"{b}->{answer_text(out_b)!r}")
    rate = agree / n if n else 0.0
    return (n > 0 and rate >= 1.0 - tolerance), rate, diffs
