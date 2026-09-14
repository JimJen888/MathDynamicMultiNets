"""
Checking a proof someone else wrote, and saying exactly what it rests on.

This is the thing this package is unusually placed to do. A proof assistant
checks that each step follows and then says `theorem proved`, which is the
right answer to the question it is asked. It is not the only question worth
asking. A proof can be perfectly valid and still rest on a citation nobody
looked up, an estimate quoted from elsewhere, or a hypothesis the author
supplied because they knew it was true. Validity does not distinguish those
from a step checked against a computation, and a reader who wants to know
how much of an argument is load-bearing has to reconstruct it by hand.

So the object produced here is not a yes or no. It is a LEDGER: for a given
conclusion, which steps were checked against an oracle and how many
instances, which were decided on their whole domain, which are exact
arithmetic, which lean on a classical fact nobody here proves, which were
asserted outright and by whom, and which are untrusted imports. The verdict
sits on top of that and is deliberately hard to overstate:

    COMPLETE            every step validated and nothing assumed
    MODULO ASSUMPTIONS  every step validated, and here is what they are
    MODULO ASSUMPTIONS AND UNTRUSTED STEPS
                        as above, and some step got through on a rule
                        nobody here has established
    INCOMPLETE          some step has no justification in this library

The middle ones are the common case and the interesting ones. Most real
arguments are complete modulo something, and the value is in the list.

The fourth verdict was added because the report earned it. On its first
run over a real decomposition it showed a step counted as validated that
had got there through an untrusted import, and "validated" was hiding the
difference. A step is validated when the machine finds a rule joining two
cells; whether that rule is any good is a separate question and now a
separate line.

## Why a proof arrives as a graph

A submitted proof is a list of `Claim`s, and each names the claims it
follows from rather than relying on the one above it. That is not a
stylistic choice. The first version of the decomposition experiment in this
repository was a flat list, which silently gave one step the wrong premise
and scored the machine three out of ten when the honest number was higher.
Proofs branch, and a format that cannot express branching will
misattribute failures.

A claim with no premises is an INPUT: something the proof assumes rather
than derives. Those are reported separately and counted, because in
practice they are where the mathematics went.

## What checking a step means

For a single premise, the machine searches for a rule or short chain
carrying the premise cell to the claim cell. For several premises it
saturates from all of them, which is what a conjunction needs. In neither
case is the submitter's justification consulted: the prose is recorded for
the reader and the machine finds the rule itself or reports that it
cannot. A justification that sounds right and matches no rule is an
INCOMPLETE step, which is the outcome worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Claim:
    """One step of a submitted proof."""

    key: str
    #: The statement, as a cell the machine can read. None when the
    #: submitter could not express it, which is itself a finding.
    cell: str | None
    #: Keys of the claims this follows from. Empty means an input.
    premises: tuple[str, ...] = ()
    claim: str = ""                      # what it says, in prose
    justification: str = ""              # why the submitter says it holds


@dataclass
class StepVerdict:
    key: str
    status: str
    detail: str
    rules: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in ("VALIDATED", "INPUT")


@dataclass
class Audit:
    """What a submitted proof was found to rest on."""

    verdicts: list[StepVerdict] = field(default_factory=list)
    target: str = ""
    #: rule name -> the facets that describe its standing
    standing: dict = field(default_factory=dict)
    #: the sentences the conclusion leans on, each with where it came from
    assumptions: list[str] = field(default_factory=list)

    @property
    def inputs(self) -> list[StepVerdict]:
        return [v for v in self.verdicts if v.status == "INPUT"]

    @property
    def derived(self) -> list[StepVerdict]:
        return [v for v in self.verdicts if v.status != "INPUT"]

    @property
    def unvalidated(self) -> list[StepVerdict]:
        return [v for v in self.derived if v.status != "VALIDATED"]

    def leaning(self) -> list[tuple[StepVerdict, list[str]]]:
        """Steps that validated THROUGH a rule nobody has established.

        The distinction the per-step status cannot carry, and the first
        thing this report caught that I had missed. "Validated" means the
        machine found a rule joining the two cells. It does not mean the
        rule is any good. A step can be validated by an untrusted import,
        which is a perfectly real step in a chain and a completely
        different thing from one validated by a proved rule, and an
        auditor is going to want them separated.
        """
        out = []
        for v in self.verdicts:
            if v.status != "VALIDATED":
                continue
            weak = [n for n in v.rules
                    if "untrusted import" in self.standing.get(n, ())]
            if weak:
                out.append((v, weak))
        return out

    def verdict(self) -> str:
        if self.unvalidated:
            return "INCOMPLETE"
        if self.leaning():
            return "MODULO ASSUMPTIONS AND UNTRUSTED STEPS"
        if self.assumptions or self.inputs:
            return "MODULO ASSUMPTIONS"
        return "COMPLETE"

    def report(self) -> str:
        lines = [f"verdict: {self.verdict()}"]
        lines.append(f"  steps derived: {len(self.derived)}, "
                     f"validated: {len(self.derived) - len(self.unvalidated)}")
        if self.unvalidated:
            lines.append("  steps with no justification in this library:")
            for v in self.unvalidated:
                lines.append(f"    {v.key}: {v.status} -- {v.detail}")
        if self.inputs:
            lines.append(f"  inputs the proof assumes rather than derives: "
                         f"{len(self.inputs)}")
            for v in self.inputs:
                lines.append(f"    {v.key}: {v.detail}")

        leaning = self.leaning()
        if leaning:
            lines.append("  steps that validated THROUGH an untrusted rule:")
            for v, weak in leaning:
                lines.append(f"    {v.key}: via {', '.join(weak)}")
        lines.append("  how each rule it used earned its place:")
        for facet in ("proved on its whole domain", "exact arithmetic",
                      "checked against an oracle", "leans on an assumption",
                      "untrusted import"):
            names = sorted(n for n, fs in self.standing.items() if facet in fs)
            if names:
                lines.append(f"    {facet:26} {', '.join(names)}")

        if self.assumptions:
            lines.append(f"  the conclusion rests on {len(self.assumptions)} "
                         f"assumption(s):")
            for text in self.assumptions:
                lines.append(f"    - {text}")
        else:
            lines.append("  nothing assumed")
        return "\n".join(lines)


def _facets(rule) -> list[str]:
    """Every way this rule's standing could be described, not just one.

    Deliberately not exclusive buckets. A band inequality is exact
    arithmetic AND proved on its whole domain AND leaning on Bernstein,
    and a reader auditing the proof needs all three, not whichever one a
    classifier picked first.
    """
    out = []
    if getattr(rule, "proved", ""):
        out.append("proved on its whole domain")
    if getattr(rule, "exact", False):
        out.append("exact arithmetic")
    if rule.stats.n_checked:
        out.append("checked against an oracle")
    if getattr(rule, "assumes", ()):
        out.append("leans on an assumption")
    if not rule.trusted:
        out.append("untrusted import")
    return out


def check_proof(machine, claims: Sequence[Claim], max_depth: int = 3,
                max_rounds: int = 4, trusted_only: bool = False) -> Audit:
    """Check a submitted proof step by step and ledger what it rests on.

    The submitter's justifications are recorded and never consulted. Each
    step is validated by the machine finding a rule for itself, or it is
    not validated and says so.
    """
    by_key = {c.key: c for c in claims}
    audit = Audit(target=claims[-1].cell or claims[-1].claim if claims else "")
    used: list[str] = []

    for c in claims:
        if not c.premises:
            audit.verdicts.append(StepVerdict(
                c.key, "INPUT", c.justification or "assumed, not derived"))
            continue
        if c.cell is None:
            audit.verdicts.append(StepVerdict(
                c.key, "NOT EXPRESSIBLE",
                "the statement has no cell the machine can read"))
            continue
        premises = [by_key[k].cell for k in c.premises if k in by_key]
        if len(premises) != len(c.premises) or any(p is None for p in premises):
            audit.verdicts.append(StepVerdict(
                c.key, "PREMISE MISSING",
                "a claim it follows from has no cell, so this cannot be "
                "stated either"))
            continue

        if len(premises) == 1:
            got = machine.prove(premises[0], c.cell, max_depth=max_depth,
                                trusted_only=trusted_only)
        else:
            got = machine.derive(premises, c.cell, max_rounds=max_rounds,
                                 trusted_only=trusted_only)
        if got.found:
            names = tuple(got.rule_names())
            used.extend(names)
            audit.verdicts.append(StepVerdict(
                c.key, "VALIDATED", " then ".join(names) or "already there",
                names))
        else:
            audit.verdicts.append(StepVerdict(
                c.key, "NO RULE",
                f"nothing in the library carries "
                f"{', '.join(c.premises)} to this cell"))

    seen: list[str] = []
    for name in used:
        if name in audit.standing:
            continue
        rule = machine.library.get(name)
        audit.standing[name] = _facets(rule)
        for text in getattr(rule, "assumes", ()):
            if text not in seen:
                seen.append(text)
    audit.assumptions = seen
    return audit
