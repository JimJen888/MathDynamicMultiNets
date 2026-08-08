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

Grounding is not a property of the reference alone but of the PAIR. A chain
that shares no rule and no training oracle with what it checks, and whose
answers are spread widely enough that two routes cannot collide by luck, is
worth far more than its links suggest: unrelated routes cannot agree on the
same wrong answer, so every agreement is a confirmation and the accuracy is a
lower bound rather than an estimate. `independence` decides that on evidence
instead of assuming it -- the same argument is worthless for a rule choosing
one of four actions, where a quarter of agreements are coincidence. The
converse guard matters more: a reference that runs the rule under test scores
a perfect 1.000 and means nothing, so it is refused.

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
    # A chain that shares no rule and no training signal with what it checks.
    # Worth nearly as much as a definition, for the reason in `independence`.
    "independent_chain": 0.95,
    "unknown": 0.0,
}


def primitives(rule: Rule) -> set[str]:
    """Every rule that firing `rule` can end up invoking, itself included.

    A composite hides its members and an ensemble hides its specialists, so
    comparing top-level names would call `ref[distributive_learned+render]`
    independent of `distributive_learned`.
    """
    names = {rule.name}
    for sub in getattr(rule, "members", ()):
        names |= primitives(sub)
    base = getattr(rule, "base", None)
    if base is not None:
        names |= primitives(base)
    for _, spec in getattr(rule, "specialists", ()):
        names |= primitives(spec)
    return names


def _training_signal(rule: Rule) -> set[str]:
    """The oracles a rule (or anything inside it) was fitted to.

    Two nets trained by the same oracle inherit that oracle's mistakes, so they
    can agree perfectly and be wrong together. Different weights are not
    independence; a different teacher is.
    """
    out: set[str] = set()
    recipe = getattr(rule, "recipe", None)
    if recipe is not None and getattr(recipe, "oracle", ""):
        out.add(recipe.oracle)
    for sub in getattr(rule, "members", ()):
        out |= _training_signal(sub)
    base = getattr(rule, "base", None)
    if base is not None:
        out |= _training_signal(base)
    for _, spec in getattr(rule, "specialists", ()):
        out |= _training_signal(spec)
    return out


@dataclass
class Independence:
    """Whether a reference could have agreed with a rule for a shared reason.

    Agreement is only evidence to the extent that the two routes could have
    disagreed. Three things can destroy that, and they are separated here
    because they call for different responses:

    `shared_rules`    the reference runs the rule under test, or some FALLIBLE
                      rule the rule under test also runs. Agreement is then
                      partly the rule agreeing with itself.
    `shared_exact`    shared components that cannot be wrong -- exact symbolic
                      rules and memorised tables, at confidence 1.0. These are
                      recorded and do NOT disqualify anything, because what
                      makes a shared component fatal is that its errors are
                      shared, and these have none. Two routes that both finish
                      with `eval_arith` still disagree whenever their
                      perception differs, which is the whole of what the check
                      is measuring.
    `shared_oracles`  both were taught by the same oracle, so both learned its
                      errors and will reproduce them together.
    `chance`          the collision probability of the reference's own answers
                      on this probe set. Where a rule picks one of four
                      actions, two unrelated routes agree a quarter of the time
                      for nothing; where it writes a sixteen-character
                      expression, coincidental agreement is negligible, and
                      that is what makes agreement worth so much there.
    """

    shared_rules: list[str] = field(default_factory=list)
    shared_oracles: list[str] = field(default_factory=list)
    shared_exact: list[str] = field(default_factory=list)
    chance: float = 0.0
    max_chance: float = 0.05

    @property
    def independent(self) -> bool:
        return (not self.shared_rules and not self.shared_oracles
                and self.chance <= self.max_chance)

    def why_not(self) -> str:
        if self.shared_rules:
            return f"the reference runs {', '.join(sorted(self.shared_rules))}"
        if self.shared_oracles:
            return f"both were taught by {', '.join(sorted(self.shared_oracles))}"
        if self.chance > self.max_chance:
            return (f"answers collide by chance {self.chance:.4f} of the time, "
                    f"over the {self.max_chance} that makes agreement mean anything")
        return ""


def collision_probability(answers: Sequence[str]) -> float:
    """P(two draws from this answer distribution coincide).

    Estimated from the reference's own outputs rather than from the size of the
    codec's vocabulary: what matters is how concentrated the answers actually
    are on this probe set, not how many the rule could in principle emit.
    """
    if len(answers) < 2:
        return 1.0
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    n = len(answers)
    # Unbiased estimate of the repeat rate, so a set of all-distinct answers
    # scores 0 rather than 1/n.
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def fallible(rule: Rule) -> bool:
    """Can this rule be wrong?

    An exact symbolic rule or a memorised table cannot: it is trusted and its
    confidence is 1.0 by construction, not by measurement. The distinction
    matters because sharing a component is only fatal when its ERRORS are
    shared, and a component with no errors contributes none.
    """
    return not (rule.trusted and rule.confidence() >= 1.0)


def _split_shared(rule: Rule, reference: Rule) -> tuple[list[str], list[str]]:
    """Shared primitives, separated into the ones that can be wrong and the
    ones that cannot. Needs the rule objects, so it walks the reference."""
    shared = primitives(rule) & primitives(reference)
    by_name = {r.name: r for r in _walk(reference)} | {r.name: r for r in _walk(rule)}
    bad = sorted(n for n in shared if fallible(by_name[n])) if shared else []
    good = sorted(n for n in shared if n not in bad)
    return bad, good


def _walk(rule: Rule):
    yield rule
    for sub in getattr(rule, "members", ()):
        yield from _walk(sub)
    base = getattr(rule, "base", None)
    if base is not None:
        yield from _walk(base)
    for _, spec in getattr(rule, "specialists", ()):
        yield from _walk(spec)


def independence(rule: Rule, reference: Rule, answers: Sequence[str],
                 max_chance: float = 0.05) -> Independence:
    """Could this reference have agreed with this rule for a shared reason?"""
    bad, good = _split_shared(rule, reference)
    return Independence(
        shared_rules=bad,
        shared_exact=good,
        shared_oracles=sorted(_training_signal(rule) & _training_signal(reference)),
        chance=collision_probability(answers),
        max_chance=max_chance,
    )


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
    independence: "Independence | None" = None
    diagnosis: str = ""                     # why nothing was checked, measured

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_checked if self.n_checked else 0.0

    @property
    def confirmed(self) -> bool:
        """Did an independent route agree, so that agreement means correctness?

        When nothing is shared, the two routes can only agree on a wrong answer
        by landing on the SAME wrong answer, whose probability is the collision
        rate in `Independence.chance`. Below that threshold, every case they
        agree on is a case the rule got right -- which is a stronger statement
        than any accuracy figure, because it does not rest on the reference
        being right, only on it being unrelated.
        """
        return self.independence is not None and self.independence.independent

    @property
    def strength(self) -> float:
        """Accuracy discounted by how good the evidence was."""
        return self.accuracy * GROUNDING_STRENGTH.get(self.grounding, 0.0)

    def summary(self) -> str:
        if not self.n_checked:
            return (f"{self.rule}: NO CHECKS RAN against {self.source} -- every "
                    "example was declined, so this says nothing.\n  "
                    + (self.diagnosis or
                       "Usually the reference or the rule expects a different "
                       "input domain."))
        head = (f"{self.rule}: {self.accuracy:.4f} over {self.n_checked} checks "
                f"vs {self.source} [{self.grounding}], strength {self.strength:.3f}")
        if self.confirmed and self.independence.shared_exact:
            head += (f"\n  shares {', '.join(self.independence.shared_exact)} with "
                     f"the reference, which is exact and contributes no error, "
                     f"so the routes are still independent where it counts")
        if self.confirmed:
            head += (f"\n  INDEPENDENT: the reference shares no rule and no oracle "
                     f"with {self.rule}, and its answers collide by chance only "
                     f"{self.independence.chance:.4f} of the time, so the "
                     f"{self.n_correct} agreements are confirmations, not "
                     f"coincidences -- {self.accuracy:.4f} is a LOWER BOUND on "
                     f"accuracy and the {self.n_checked - self.n_correct} "
                     f"disagreements are unattributed (either side may be wrong)")
        elif self.independence is not None:
            head += f"\n  NOT independent: {self.independence.why_not()}"
        if self.counterexamples:
            head += "\n  counterexamples: " + "; ".join(self.counterexamples[:5])
        head += f"\n  {'TRUSTED' if self.became_trusted else 'not trusted'} " \
                f"(threshold {self.threshold})"
        return head


def why_nothing_ran(rule: Rule, examples: Sequence[Example],
                    chain: Sequence[Rule] = ()) -> str:
    """Which link stopped, and on what. Measured, not guessed.

    A check that examined nothing is the most confusing result the machine
    produces: it looks like a failure of the rule when it is almost always a
    mismatch several steps earlier. Walking the chain per example finds the
    first member that declines and reports it with a case, which turns "this
    says nothing" into the specific edit that would make it say something.
    """
    if not examples:
        return "the dataset is empty."

    sample = examples[0].inp
    rows: list[str] = []

    if chain:
        for i, member in enumerate(chain):
            cur = sample
            declined = 0
            for ex in examples:
                cell = ex.inp
                ok = True
                for earlier in chain[:i]:
                    cell = earlier.apply(cell) if cell is not None else None
                    if cell is None:
                        ok = False
                        break
                if ok and cell is not None and member.apply(cell) is None:
                    declined += 1
                    cur = cell
            if declined:
                rows.append(
                    f"the reference stops at {member.name!r}: it declined "
                    f"{declined}/{len(examples)} of the cells reaching it, "
                    f"such as {cur.summary()}. It accepts {member.expects()}.")
                break
        else:
            rows.append("every member of the reference chain applied, so the "
                        "reference produced an answer and the mismatch is "
                        "elsewhere.")

    if rule.apply(sample) is None:
        rows.append(f"the rule under test, {rule.name!r}, also declines "
                    f"{sample.summary()}; it accepts {rule.expects()}.")
    if not rows:
        rows.append("nothing declined the inputs, so the examples were probably "
                    "unlabelled -- label_data first.")
    return "\n  ".join(rows)


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
    diagnosis = ""
    if not n:
        labelled = sum(1 for ex in example_set.examples if ex.labeled)
        if not labelled:
            diagnosis = (f"none of the {len(example_set.examples)} examples "
                         f"carries a label: {oracle_name!r} declined every one "
                         f"of them, so it is the wrong oracle for this "
                         f"generator's output.")
        else:
            diagnosis = why_nothing_ran(rule, example_set.examples)
    report = VerificationReport(
        rule=rule_name, n_checked=n, n_correct=correct,
        grounding=ORACLES[oracle_name].kind, source=oracle_name,
        counterexamples=bad, threshold=threshold, diagnosis=diagnosis,
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
    independent_threshold: float | None = None,
    max_chance: float = 0.05,
) -> VerificationReport:
    """Check a rule against a chain of rules the machine already trusts.

    This is how a learned rule gets checked with no oracle in sight: the
    distributive rewrite learned from pictures is compared with
    decimal_split -> distribute_symbolic applied to the same input. If any link
    in the reference chain is itself untrusted, the check is refused outright
    rather than quietly producing a number.

    How much the agreement is worth depends on whether the reference COULD have
    disagreed, which `independence` decides on three grounds: no shared rule, no
    shared training oracle, and answers spread widely enough that coincidence is
    negligible. When all three hold, agreement is not merely evidence that the
    rule matches a reference -- it is evidence the rule is RIGHT, because two
    unrelated routes have no way to land on the same wrong answer. A rewrite
    into a sixteen-character expression makes that argument overwhelming; a rule
    choosing one of four actions does not, and `max_chance` is what separates
    the two rather than an assumption about which rules deserve it.

    So an independent check is held to `independent_threshold` (default: the
    same threshold, minus nothing -- pass a lower one deliberately), and the
    accuracy it reports is a LOWER BOUND: disagreements are unattributed,
    because an independent reference is exactly the kind that can be wrong on
    its own. Attributing every disagreement to the rule, as the plain accuracy
    does, understates a rule checked this way.
    """
    rule = library.get(rule_name)
    members = [library.get(n) for n in reference_chain]
    untrusted = [m.name for m in members if not m.trusted]
    if untrusted:
        raise ValueError(
            f"cannot verify against untrusted rules {untrusted}; verify those first"
        )
    reference = CompositeRule(f"ref[{'+'.join(reference_chain)}]", members)

    # A reference that runs the rule under test agrees with it for free. This is
    # the one failure that produces a perfect score, so it is refused rather
    # than discounted.
    # Refused on two grounds, and only these two. The reference literally runs
    # the rule under test, which scores a perfect 1.000 for nothing; or the two
    # share a component that CAN BE WRONG, whose mistakes therefore appear on
    # both sides and inflate the agreement.
    #
    # Sharing an exact rule is neither. A chain ending in `eval_arith` and a
    # rule ending in `eval_arith` still disagree wherever their perception
    # differs, because exact arithmetic has no errors to contribute -- refusing
    # that check rejects a sound one and leaves the rule unverifiable by any
    # route that has to finish by computing something.
    fallible_shared, exact_shared = _split_shared(rule, reference)
    if rule_name in primitives(reference):
        raise ValueError(
            f"{rule_name!r} cannot be verified against a reference that runs it: "
            f"agreement would be the rule agreeing with itself. Build the "
            f"reference from an independent route."
        )
    if fallible_shared:
        raise ValueError(
            f"{rule_name!r} and the reference both depend on {fallible_shared}, "
            f"which can be wrong -- shared mistakes would appear on both sides "
            f"and inflate the agreement. Replace that link, or check against an "
            f"oracle instead."
            + (f" (They also share {exact_shared}, which is exact and would "
               f"have been fine on its own.)" if exact_shared else "")
        )

    n = correct = 0
    bad: list[str] = []
    ref_answers: list[str] = []
    for ex in example_set.examples:
        want = reference.apply(ex.inp)
        if want is None:
            continue                       # the reference has nothing to say here
        n += 1
        ref_answers.append(normalize(answer_text(want)))
        got = rule.apply(ex.inp)
        if got is not None and normalize(answer_text(got)) == normalize(answer_text(want)):
            correct += 1
        elif len(bad) < 32:
            bad.append(f"{ex.inp.text!r} -> {answer_text(got)!r} "
                       f"(reference {answer_text(want)!r})")

    indep = independence(rule, reference, ref_answers, max_chance=max_chance)
    diagnosis = ("" if n else
                 why_nothing_ran(rule, example_set.examples, members))
    report = VerificationReport(
        rule=rule_name, n_checked=n, n_correct=correct,
        grounding="independent_chain" if indep.independent else "rule_chain",
        source=reference.name, counterexamples=bad,
        threshold=(independent_threshold if indep.independent
                   and independent_threshold is not None else threshold),
        independence=indep, diagnosis=diagnosis,
    )
    rule.stats.merge(n, correct, reference.name, bad)
    if n and report.accuracy >= report.threshold:
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
