"""
Conciseness: what the machine is actually optimising.

The goal is "rules as concise and useful as possible", and those two words pull
against each other unless they are priced in the same currency. They are, here,
by a two-part code:

    J(library) = bits to write the rules down
               + BITS_PER_STEP * rule applications needed to solve the benchmark

The first term punishes keeping things; the second punishes deriving things the
long way round. A rule earns its place exactly when it shortens more derivation
than it costs to state. Three consequences fall out, and all three are the
behaviours the paper describes:

  * A rule nothing uses is deleted -- it is pure first term.
  * A six-step chain that recurs is worth replacing with one composite or one
    distilled net, because six applications cost more than one recipe.
  * A learned net that duplicates a two-symbol algebraic identity is not worth
    keeping, even at equal accuracy, because the identity is cheaper to state.

`distill` is the interesting operation: it trains a single net to imitate a
whole chain, which is how a derivation the machine had to search for becomes a
rule it can apply in one move. Note what it does NOT do -- it does not inherit
the chain's trust. A distilled rule is a new empirical claim about a function
and has to be verified like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .codec import ChoiceCodec, Codec, TextSlotCodec
from .dataset import ExampleSet
from .proof import BITS_PER_STEP, Proof, search
from .rules import CompositeRule, NeuralRule, Rule, RuleLibrary
from .tapes import Content
from .verify import answer_text, equivalent, normalize

UNSOLVED_PENALTY_STEPS = 20.0     # what an unsolved benchmark task costs


@dataclass
class Task:
    """One benchmark problem: what is on the tape, and what must be reached."""

    name: str
    start: Content
    target: str
    max_depth: int = 8
    target_domain: str = "abstract"     # where the conclusion must be written


@dataclass
class LibraryReport:
    total_bits: float = 0.0
    per_rule_bits: dict[str, float] = field(default_factory=dict)
    solved: dict[str, bool] = field(default_factory=dict)
    steps: dict[str, float] = field(default_factory=dict)
    used_by: dict[str, list[str]] = field(default_factory=dict)
    unused: list[str] = field(default_factory=list)
    proofs: dict[str, Proof] = field(default_factory=dict)

    @property
    def total_steps(self) -> float:
        return sum(self.steps.values())

    @property
    def objective(self) -> float:
        """J, in bits. Lower is a more concise and more useful library."""
        return self.total_bits + BITS_PER_STEP * self.total_steps

    def summary(self) -> str:
        n_solved = sum(1 for v in self.solved.values() if v)
        rows = [f"library: {len(self.per_rule_bits)} rules, {self.total_bits:.0f} bits",
                f"benchmark: {n_solved}/{len(self.solved)} solved, "
                f"{self.total_steps:.0f} rule applications",
                f"objective J = {self.objective:.0f} bits"]
        if self.unused:
            waste = sum(self.per_rule_bits[n] for n in self.unused)
            rows.append(f"unused by the benchmark ({waste:.0f} bits): "
                        + ", ".join(sorted(self.unused)))
        for name, ok in sorted(self.solved.items()):
            p = self.proofs[name]
            rows.append(f"  {name}: {'ok' if ok else 'FAILED'} "
                        f"{p.length} steps via {', '.join(p.rule_names()) or '-'}")
        return "\n".join(rows)


def library_report(library: RuleLibrary, benchmark: Sequence[Task],
                   trusted_only: bool = True, max_nodes: int = 4000) -> LibraryReport:
    """Price the library: description bits plus derivation length on a benchmark."""
    rep = LibraryReport()
    rep.per_rule_bits = {r.name: r.cost_bits() for r in library}
    rep.total_bits = sum(rep.per_rule_bits.values())
    rep.used_by = {name: [] for name in rep.per_rule_bits}

    for task in benchmark:
        proof = search(library, task.start, task.target, max_depth=task.max_depth,
                       max_nodes=max_nodes, trusted_only=trusted_only,
                       target_domain=task.target_domain)
        rep.proofs[task.name] = proof
        rep.solved[task.name] = proof.found
        rep.steps[task.name] = (float(proof.length) if proof.found
                                else UNSOLVED_PENALTY_STEPS)
        for name in set(proof.rule_names()):
            rep.used_by.setdefault(name, []).append(task.name)

    rep.unused = [n for n, tasks in rep.used_by.items() if not tasks]
    return rep


# ---------------------------------------------------------------------------
# Building bigger rules
# ---------------------------------------------------------------------------
def compose(library: RuleLibrary, names: Sequence[str], new_name: str,
            description: str = "") -> CompositeRule:
    """Name a chain of rules so it can be applied as one."""
    members = [library.get(n) for n in names]
    rule = CompositeRule(new_name, members,
                         description or "chain: " + " -> ".join(names))
    rule.trusted = all(m.trusted for m in members)
    return library.add(rule, replace=True)


def _codec_for(outputs: Sequence[Content], out_domain: str,
               classes: Sequence[str] | None = None) -> Codec:
    """Pick the smallest codec that can express the observed outputs.

    Few distinct outputs -> a choice; otherwise a slot transduction sized to
    the longest string seen, with the vocabulary it actually used. Sizing from
    the data rather than from a default is part of conciseness: an unnecessary
    slot is capacity the rule has to learn to leave blank.
    """
    texts = [o.text for o in outputs]
    distinct = sorted(set(texts))
    if classes is not None or len(distinct) <= 8:
        return ChoiceCodec(classes or distinct, out_domain=out_domain)
    vocab = "_" + "".join(sorted(set("".join(texts))))
    return TextSlotCodec(num_slots=max(len(t) for t in texts), vocab=vocab,
                         out_domain=out_domain)


def distill(
    library: RuleLibrary,
    source_name: str,
    new_name: str,
    example_set: ExampleSet,
    epochs: int = 12,
    seed: int = 0,
    device: str | None = None,
    log=None,
):
    """Train one net to imitate `source_name` (usually a composite chain).

    Labels come from running the source rule over the examples -- the chain
    teaches the net. Inputs it declines are dropped rather than labeled with a
    guess. Returns (rule, train_report, agreement) where `agreement` is the
    fraction of held-in examples on which the distilled rule reproduces the
    chain; it is NOT trust, and `verify.verify_rule` still has to be run.
    """
    from .train import train_rule

    source = library.get(source_name)
    outputs, kept = [], []
    for ex in example_set.examples:
        out = source.apply(ex.inp)
        if out is None:
            continue
        ex.out = out
        ex.label_source = f"rule:{source_name}"
        kept.append(ex)
        outputs.append(out)
    if not kept:
        raise ValueError(f"{source_name!r} produced no outputs on {example_set.name}")

    teaching = ExampleSet(f"{example_set.name}:distill:{new_name}", kept,
                          example_set.generator, dict(example_set.generator_params),
                          f"rule:{source_name}", example_set.seed)
    codec = _codec_for(outputs, source.domain_out)
    rule = NeuralRule(new_name, codec, source.domain_in,
                      description=f"distilled from {source_name} "
                                  f"({source.steps()} steps -> 1)", device=device)
    report = train_rule(rule, teaching, epochs=epochs, seed=seed, log=log)
    library.add(rule, replace=True)

    agree = sum(1 for ex in kept
                if normalize(answer_text(rule.apply(ex.inp)))
                == normalize(answer_text(ex.out)))
    return rule, report, agree / len(kept)


# ---------------------------------------------------------------------------
# Making the library smaller
# ---------------------------------------------------------------------------
@dataclass
class SimplifyAction:
    kind: str            # "drop_unused" | "drop_duplicate"
    rule: str
    reason: str
    bits_saved: float

    def line(self) -> str:
        return f"{self.kind}: {self.rule} (-{self.bits_saved:.0f} bits) -- {self.reason}"


def simplify(
    library: RuleLibrary,
    benchmark: Sequence[Task],
    probe: ExampleSet | None = None,
    apply_changes: bool = False,
    protect: Sequence[str] = (),
) -> tuple[list[SimplifyAction], LibraryReport, LibraryReport | None]:
    """Propose (and optionally make) the library smaller without losing ground.

    Two moves, in order:

      1. Duplicates. If two rules agree everywhere one of them applies, the
         more expensive one goes. This is where a learned rule that merely
         rediscovered a known identity gets absorbed -- and where the identity
         gets dropped instead, if the learned rule happens to be cheaper and
         covers more. A rule is never displaced by one the machine trusts LESS:
         agreement on a probe set is evidence about that probe set, and the
         cheapest way to "agree" with a reader on cells the machine drew itself
         is to copy the caption. Trading a verified rule for an unverified one
         is not a simplification, it is a loss of grounding that the bit count
         cannot see.
      2. Dead weight. A rule no benchmark proof uses is pure description cost.

    Every drop is re-checked by re-pricing the library afterwards, ONE AT A
    TIME, and a change that would lose a benchmark task is put straight back --
    "concise" that cannot prove what it could before is not a simplification.
    Checking them one at a time is what keeps a bad drop from cascading: remove
    the rule a proof depended on and every other rule stops being used too, so a
    single mistake in the first pass would otherwise empty the library before
    anything noticed. Members of surviving composites and anything in `protect`
    are never dropped.
    """
    before = library_report(library, benchmark)
    actions: list[SimplifyAction] = []
    protected = set(protect)
    for rule in library:
        if isinstance(rule, CompositeRule):
            protected.update(m.name for m in rule.members)

    def try_drop(name: str) -> bool:
        """Remove `name` and keep it removed only if the benchmark survives."""
        rule = library.get(name)
        library.remove(name)
        trial = library_report(library, benchmark)
        lost = [t for t, ok in before.solved.items() if ok and not trial.solved.get(t)]
        if lost:
            library.add(rule, replace=True)
            return False
        return True

    # 1. duplicates
    if probe is not None:
        names = sorted(library.rules)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a not in library or b not in library:
                    continue
                ra, rb = library.get(a), library.get(b)
                if (ra.domain_in, ra.domain_out) != (rb.domain_in, rb.domain_out):
                    continue
                same, rate, _ = equivalent(library, a, b, probe)
                if not same:
                    continue
                loser = a if ra.cost_bits() >= rb.cost_bits() else b
                keeper = b if loser == a else a
                if loser in protected:
                    continue
                if library.get(loser).trusted and not library.get(keeper).trusted:
                    continue                     # never trade verified for not
                if apply_changes and not try_drop(loser):
                    continue
                actions.append(SimplifyAction(
                    "drop_duplicate", loser,
                    f"agrees with {keeper} on {rate:.0%} of {probe.name} and costs more",
                    ra.cost_bits() if loser == a else rb.cost_bits()))

    # 2. dead weight
    report = library_report(library, benchmark) if apply_changes else before
    for name in report.unused:
        if name in protected or name not in library:
            continue
        bits = library.get(name).cost_bits()
        if apply_changes and not try_drop(name):
            continue
        actions.append(SimplifyAction("drop_unused", name,
                                      "no benchmark proof uses it", bits))

    after = library_report(library, benchmark) if apply_changes else None
    return actions, before, after
