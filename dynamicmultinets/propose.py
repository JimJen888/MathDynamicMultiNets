"""
Where a hypothesis comes from.

The rest of the package assumes somebody already decided which rule is worth
forming: `declare_rule` is handed a name, a mapping and an oracle, and
everything after that is fitting and checking. That is the second half of the
job. This module is the first half -- choosing what to hypothesise -- and the
paper is specific about how it happens:

    a case the machine cannot do is put beside cases it can, BOTH ARE MAPPED
    INTO THE SPECIFIC DOMAIN, and the controller summarises what they have in
    common as a rule to be tested.

The mapping into the specific domain is not decoration. Two products written as
`12*30` and `10*30+2*30` are different strings; drawn on the tape they are two
arrangements of the same marks, and the regrouping is a fact about the layout.
That is the architecture's claim -- structure is expressed as layout and
perception is what recovers it -- so a proposer that pattern-matched the
abstract strings would be answering an easier question than the one the paper
poses. Every case here goes through the machine's own `render` rule and reaches
the controller as pixels, with the caption alongside only as provenance.

Two shapes of analogy, and they are different CLAIMS with different tests.

  SHARED PATTERN   "the pattern established on the known instances also holds
                    on the unknown ones, under <condition>."

                    Same problem, new instances: the distributive rule verified
                    on two-digit products is claimed to hold for three-digit
                    ones. Or two problems sharing a structure: what is proven
                    of the 2-D Poincare conjecture is claimed to hold in 3-D
                    under the right hypotheses. Either way the claim names a
                    family where the pattern is ESTABLISHED and a family where
                    it is only CONJECTURED, and the condition is what is
                    supposed to license carrying it across.

                    Tested by running the pattern on the unknown family. A
                    transfer that holds is a rule to train and verify there; a
                    transfer that fails has produced counterexamples, which is
                    the more useful outcome and the one a single-family proposal
                    cannot even express.

  INTERCONVERSION   "the unproven case can be mapped to a solved one, so the
                    unproven one is established by transport."

                    Fermat's Last Theorem and the semistable case of
                    Taniyama-Shimura: the work is not to notice a shared
                    pattern but to build the correspondence. So what is
                    proposed is a SEARCH -- find a chain of mapping rules from
                    the unproven statement to the proven one, or back -- and
                    the machine already has the machinery, because that is what
                    `proof.search` does.

                    Tested by looking for the chain. Finding one is the
                    transport; failing to find one bounds the attempt at a
                    stated depth rather than leaving it open.

A proposal is not a rule and not code. Every part of it is a name the machine
already has -- a generator, an oracle, existing rules -- so the controller
selects from a closed set rather than emitting anything that gets executed,
which is the same restriction generators.py makes and for the same reason. A
proposal naming a generator nobody registered, or parameters the generator
cannot take, is rejected here rather than discovered halfway through training.

Offline, `heuristic_proposals` does the same job by trying every oracle against
the unsolved cases and keeping the ones that actually apply. It is much weaker
-- it can only notice that an oracle fits, never why -- but it keeps the
pipeline runnable without an API key, and it makes the LLM's contribution
measurable rather than assumed: the two proposers can be run on the same
analogy and compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .generators import GENERATORS
from .oracles import ORACLES
from .tapes import ABSTRACT, SPECIFIC, Content


@dataclass
class Case:
    """One worked problem, as the controller will see it.

    `image` is the cell the machine's own render rule produced. `derivation` is
    how a solved case was solved -- the rule names, in order -- because that is
    the part an unsolved case is missing, and saying which rules ran is more
    informative than saying that some did.
    """

    text: str
    solved: bool
    target: str = ""
    image: Any = None                       # np.ndarray, the specific-domain cell
    view_text: str = ""                     # the same cell written as characters
    derivation: list[str] = field(default_factory=list)
    note: str = ""

    def caption(self) -> str:
        if self.solved:
            via = " -> ".join(self.derivation) if self.derivation else "unknown route"
            return f"SOLVED   {self.text}   via {via} {self.note}".rstrip()
        return f"UNSOLVED {self.text}   {self.note or 'no rule chain reaches this'}"


@dataclass
class Analogy:
    """The material a hypothesis is summarised from."""

    solved: list[Case] = field(default_factory=list)
    unsolved: list[Case] = field(default_factory=list)
    goal: str = ""

    def summary(self, cells: bool = False) -> str:
        rows = [f"{len(self.solved)} solved, {len(self.unsolved)} unsolved"]
        for c in self.solved + self.unsolved:
            rows.append(f"  {c.caption()}")
            if cells and c.view_text:
                rows.append(c.view_text)
        return "\n".join(rows)


@dataclass
class RuleProposal:
    """"The pattern established on the known instances also holds on the
    unknown ones, under <condition>."

    Two families of the SAME generator, differing only in parameters: `known`
    is where the pattern is already established, `unknown` is where it is
    being claimed. Splitting them is the whole point -- a proposal naming one
    family can say "learn this here" but not "what holds there also holds
    here", which is the claim an analogy actually makes and the only one that
    can be refuted.

    `condition` is the "under proper condition" clause in words. It is not
    executed; nothing here can check that a stated hypothesis is the one really
    doing the work. What IS checked is the consequence: run the pattern over
    the unknown family and see whether it survives.

    Everything feeds the existing pipeline unchanged -- `as_data_args` to
    `generate_data`, `oracle` to `label_data`, `as_declare_args` to
    `declare_rule` -- and nothing is trusted on the way in.
    """

    name: str
    domain_in: str
    domain_out: str
    generator: str
    oracle: str
    known: dict[str, Any] = field(default_factory=dict)      # already established
    unknown: dict[str, Any] = field(default_factory=dict)    # claimed to extend to
    known_oracle: str = ""                  # the FORM it is established in
    condition: str = ""
    num_slots: int = 0
    from_oracle: bool = False               # output shape taken from the oracle
    kind: str = "shared_pattern"
    rationale: str = ""
    proposed_by: str = ""

    @property
    def generator_params(self) -> dict[str, Any]:
        """The family a rule would be TRAINED on: the one being claimed."""
        return dict(self.unknown or self.known)

    @property
    def established_as(self) -> str:
        """The form the pattern already holds in, which is usually the same
        one being claimed -- the transfer is then across instances."""
        return self.known_oracle or self.oracle

    def transfers_form(self) -> bool:
        """Is the pattern being carried to a different FORM of itself?

        The distributive law is the case that made this necessary. It is
        established as `distributive_rewrite`, which splits the left factor,
        and the interesting claim is that it also holds as
        `distributive_rewrite_right`, which splits the right one. That is not a
        wider family of the same rewrite; it is the same pattern in a mirrored
        form, and a proposal that could only vary generator parameters could
        state that the rewrite works on bigger numbers but never that it works
        on the other side.
        """
        return bool(self.known_oracle) and self.known_oracle != self.oracle

    def claim(self) -> str:
        known = json.dumps(self.known) if self.known else "the solved cases"
        unknown = json.dumps(self.unknown) if self.unknown else "the unsolved cases"
        if self.transfers_form():
            text = (f"the pattern established as {self.known_oracle!r} on "
                    f"{known} also holds as {self.oracle!r}")
            if self.unknown and self.unknown != self.known:
                text += f" on {unknown}"
        else:
            text = (f"the pattern {self.oracle!r}, established on {known}, "
                    f"also holds on {unknown}")
        return f"{text}, provided {self.condition}" if self.condition else text

    def as_declare_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "name": self.name, "domain_in": self.domain_in,
            "out_domain": self.domain_out, "description": self.rationale[:200],
        }
        if self.from_oracle:
            args["from_oracle"] = self.oracle
        elif self.num_slots:
            args["num_slots"] = self.num_slots
        return args

    def as_data_args(self, n: int = 2000, seed: int = 1, name: str = "") -> dict[str, Any]:
        return {"generator": self.generator, "n": n, "seed": seed,
                "name": name or f"{self.name}_train",
                "params": self.generator_params}

    def check(self, machine=None, n: int = 60) -> str:
        """Does the pattern survive on the unknown family?

        Only the oracle is run, so this says whether the pattern is even
        DEFINED where it is being claimed -- an oracle that declines every
        unknown instance has had its transfer refuted before a net is trained.
        Whether a learned rule attains it is what `verify_rule` measures after.
        """
        from .generators import generate

        try:
            fresh = generate(self.generator, n, seed=7, **self.generator_params)
        except Exception as err:
            return f"cannot even generate the unknown family: {err}"
        fn = ORACLES[self.oracle].fn
        held = sum(1 for ex in fresh.examples if _labels(fn, ex))
        if not held:
            return (f"REFUTED before training: {self.oracle!r} produces nothing "
                    f"on {json.dumps(self.generator_params)}, so the pattern "
                    f"does not even apply there")
        note = (f"the pattern applies to {held}/{len(fresh.examples)} of the "
                f"unknown family")

        # "Established" is half the claim, so check it rather than assume it.
        # A transfer FROM somewhere the pattern never held is not a transfer.
        if self.known:
            try:
                base = generate(self.generator, n, seed=11, **self.known)
            except Exception as err:
                return note + f"; but the known family does not generate: {err}"
            base_fn = ORACLES[self.established_as].fn
            base_held = sum(1 for ex in base.examples if _labels(base_fn, ex))
            if not base_held:
                return (f"the claim has no base: {self.established_as!r} produces "
                        f"nothing on {json.dumps(self.known)}, so there is "
                        f"nothing established to carry across")
            note += (f"; {self.established_as!r} holds on "
                     f"{base_held}/{len(base.examples)} of the known family")
        if machine is not None:
            already = [r.name for r in machine.library
                       if getattr(r, "recipe", None)
                       and r.recipe.oracle == self.oracle and r.trusted]
            if already:
                note += (f"; {', '.join(already)} already implements it on the "
                         f"known family, so this is a transfer to measure, not "
                         f"a rule to learn from scratch")
        return note + "; train and verify_rule to see whether a rule attains it"

    def summary(self) -> str:
        shape = (f"from_oracle={self.oracle}" if self.from_oracle
                 else f"num_slots={self.num_slots}")
        return (f"{self.name}  [shared_pattern]  "
                f"{self.domain_in}->{self.domain_out}\n"
                f"    claim: {self.claim()}\n"
                f"    because: {self.rationale}\n"
                f"    test it with: generate_data({self.generator}, "
                f"{json.dumps(self.generator_params)}) -> "
                f"label_data({self.oracle}) -> declare_rule({shape}) -> verify_rule")


@dataclass
class Interconversion:
    """"The unproven case maps to a solved one, so it is established by
    transport."

    Fermat's Last Theorem via the semistable case of Taniyama-Shimura is the
    shape: the content is not a pattern the two share but a CORRESPONDENCE
    between them, and the work is building it. So this proposes a search --
    find a chain of mapping rules from `source` to `target`, or back -- and
    `check` runs it, because chaining rules across domains is what the machine
    does anyway.

    `via` is a suggested route, if the controller had one in mind. It is a hint
    for the log; the search is not restricted to it, since a proposal that
    could only be confirmed the way it was imagined would confirm the
    imagination rather than the claim.
    """

    name: str
    source: str                             # the unproven statement
    target: str                             # the established one
    via: list[str] = field(default_factory=list)
    both_ways: bool = False                 # a one-to-one correspondence?
    domain: str = ABSTRACT
    kind: str = "interconversion"
    rationale: str = ""
    proposed_by: str = ""

    def claim(self) -> str:
        arrow = "<->" if self.both_ways else "->"
        hint = f" (perhaps via {' -> '.join(self.via)})" if self.via else ""
        return (f"the unproven {self.source!r} maps {arrow} the established "
                f"{self.target!r}, so establishing the second carries the "
                f"first{hint}")

    def check(self, machine, max_depth: int = 8) -> str:
        """Look for the chain. This is the test, not a rehearsal of it."""
        from .proof import search

        def start(a: str) -> Content:
            # Honour the stated domain. A correspondence claimed between two
            # DRAWINGS is a different search from one between two statements,
            # and starting both on the abstract tape would quietly test the
            # easier claim.
            return (Content.specific_text(a) if self.domain == SPECIFIC
                    else Content.abstract(a))

        def leg(a: str, b: str) -> str:
            proof = search(machine.library, start(a), b,
                           max_depth=max_depth, trusted_only=True)
            if proof.found:
                return (f"{a!r} -> {b!r}: FOUND in {proof.length} steps "
                        f"(confidence {proof.confidence:.4f}) via "
                        + " -> ".join(proof.rule_names()))
            return (f"{a!r} -> {b!r}: no chain within depth {max_depth} "
                    f"({proof.note}) -- the correspondence is unbuilt, which "
                    f"is a bound on the attempt, not a refutation")

        rows = [leg(self.source, self.target)]
        if self.both_ways:
            rows.append(leg(self.target, self.source))
        return "\n      ".join(rows)

    def summary(self) -> str:
        return (f"{self.name}  [interconversion]  in the {self.domain} domain\n"
                f"    claim: {self.claim()}\n"
                f"    because: {self.rationale}\n"
                f"    test it with: prove(start={self.source!r}, "
                f"target={self.target!r})"
                + (f" and the reverse" if self.both_ways else ""))


def _labels(fn, example) -> bool:
    try:
        out = fn(example)
    except Exception:
        return False
    return out is not None and bool(out.text)


# ---------------------------------------------------------------------------
# Building the analogy
# ---------------------------------------------------------------------------
def worked_examples(machine, via: Sequence[str], generator: str = "mul_pairs",
                    params: dict[str, Any] | None = None, n: int = 12,
                    seed: int = 0, expand_terms: Sequence[str] = ()) -> list[str]:
    """Solved cases DERIVED rather than asserted.

    An analogy is only as good as its solved side, and hand-writing that side
    has two failure modes that cost real runs here: the cases turn out not to
    be solvable at all (`40*19 => 40*10+40*9` was unreachable until the mirror
    rules existed), or they are solvable but share no structure with the open
    question (`9*7 => 63` is a table lookup, not an instance of a regrouping).
    Running the machine's own trusted chain over generated inputs avoids both:
    every case comes back with a derivation because it was just derived, and
    every case is an instance of exactly the pattern the chain performs.

    `expand_terms` continues the decomposition into each `+`-separated part, so
    one call produces the whole tree a person would write out --
    `46*19 => 40*19+6*19`, then `40*19 => 40*10+40*9`, and so on down to the
    times table -- rather than only its first level.
    """
    from .generators import generate

    rules = [machine.library.get(name) for name in via]
    followers = [machine.library.get(name) for name in expand_terms]
    fresh = generate(generator, max(n * 3, 12), seed=seed, **(params or {}))

    def run(text: str, chain) -> str:
        cell = Content.abstract(text)
        for rule in chain:
            cell = rule.apply(cell) if cell is not None else None
            if cell is None:
                return ""
        return cell.text if cell is not None and cell.text != text else ""

    out: list[str] = []
    seen: set[str] = set()
    for ex in fresh.examples:
        start = ex.inp.text
        got = run(start, rules)
        if not got or start in seen:
            continue
        seen.add(start)
        out.append(f"{start} => {got}")
        if followers:
            # The parts of what we just produced, taken one level further.
            for term in (t.strip() for t in got.split("+")):
                deeper = run(term, followers)
                if deeper and term not in seen:
                    seen.add(term)
                    out.append(f"{term} => {deeper}")
        if len(out) >= n:
            break
    return out[:n]


def split_case(spec: str) -> tuple[str, str]:
    """`"12*30 => 10*30+2*30"` -> ("12*30", "10*30+2*30"); no arrow -> no target."""
    for sep in ("=>", "->"):
        if sep in spec:
            start, target = spec.split(sep, 1)
            return start.strip(), target.strip()
    return spec.strip(), ""


def gather_analogy(machine, unsolved: Sequence[str], solved: Sequence[str] = (),
                   goal: str = "", max_depth: int = 6,
                   domain: str = ABSTRACT, observed: bool = False,
                   solved_domain: str | None = None) -> Analogy:
    """Put the cases side by side, each drawn onto the specific tape.

    A case is written `"start => target"`, and whether it is solved is MEASURED
    rather than asserted: the start is handed to proof search over the trusted
    library and either the target comes out or it does not. A caller that
    mislabels its own examples would otherwise be asking for a hypothesis about
    a distinction that is not there.

    Stating the target is what makes the question meaningful. `12*30` on its own
    is "solved" the moment any rule fires on it -- `eval_arith` returns 360 and
    the machine looks omniscient. `12*30 => 10*30+2*30` is a different question
    with a different answer, and it is the one the distributive rule exists for.
    Cases given without a target fall back to "does anything at all apply",
    which is weak and says so in the caption.

    `domain` decides where the case STARTS, and it changes the question
    completely. Posed on the abstract tape, `12*30 => 10*30+2*30` is already
    solved by `decimal_split -> distribute_symbolic` on a machine with no
    learned rules at all -- the identity is prior knowledge. Posed as a DRAWING
    with `observed=True`, the same case cannot be touched until something can
    read the screen, which is the gap the multiplication experiment exists to
    fill. `observed` also stops `transcribe_unsafe` from answering by copying
    the caption, so the cases stay genuinely unsolved rather than trivially
    solved by the machine's own handwriting.

    `solved_domain` lets the two sides live on DIFFERENT tapes, which is usually
    what an analogy needs. Worked instances of a regrouping -- `46*19` becoming
    `40*19+6*19`, each part bottoming out in the times table -- are things the
    machine really can do, but only as SYMBOLS. Pose everything as drawings and
    they are unsolved too, because nothing can read a screen yet, and an analogy
    whose solved side is empty compares nothing. Pose the worked instances on
    the abstract tape and the open cases as drawings, and the contrast is the
    real one: this regrouping I can do in symbols, and cannot yet do from a
    picture.
    """
    from .proof import search
    from .render import layout_text
    from .verify import normalize

    render = machine.library.get("render")
    analogy = Analogy(goal=goal or machine.goal)

    def make(spec: str, in_domain: str) -> Case:
        start, target = split_case(spec)
        cell = render.apply(Content.abstract(start))
        if in_domain == SPECIFIC:
            begin = Content.specific_text(start)
            begin.meta["observed"] = observed
        else:
            begin = Content.abstract(start)
        if target:
            proof = search(machine.library, begin, target,
                           max_depth=max_depth, trusted_only=True)
        else:
            proof = search(machine.library, begin,
                           lambda c, t=normalize(start): (c.domain == ABSTRACT
                                                          and normalize(c.text) != t),
                           max_depth=max_depth, trusted_only=True)
        image = cell.image if cell is not None else None
        # Both sides, when a target is given. The regrouping is not visible in
        # the starting cell -- `12*30` and `9*7` are both a single box -- it is
        # visible in what the case has to BECOME, and comparing the two layouts
        # is the whole analogy.
        view = layout_text(start)
        if target:
            view += "\n    must become:\n" + layout_text(target)
        case = Case(text=spec.strip(), target=target, solved=proof.found,
                    image=image, view_text=view,
                    derivation=list(proof.rule_names()) if proof.found else [])
        if not proof.found:
            case.note = proof.note or "no trusted chain reaches it"
        elif not target:
            case.note = "(no target given -- only shows that some rule applies)"
        return case

    reference_domain = solved_domain or domain
    for spec in solved:
        c = make(spec, reference_domain)
        if c.solved and reference_domain != domain:
            # Where a case was worked matters when the sides differ: "solved as
            # symbols" is not the same claim as "solved from the screen", and
            # the caption would otherwise read as though it were.
            c.note = f"(in the {reference_domain} domain)"
        (analogy.solved if c.solved else analogy.unsolved).append(c)
    for spec in unsolved:
        c = make(spec, domain)
        (analogy.unsolved if not c.solved else analogy.solved).append(c)
    return analogy


# ---------------------------------------------------------------------------
# Validation -- the closed registry, enforced
# ---------------------------------------------------------------------------
def validate(proposal, library=None, dry_run: bool = True) -> list[str]:
    """Everything wrong with a proposal, as sentences. Empty means usable.

    Names are checked against the registries, and then -- unless `dry_run` is
    off -- the recipe is actually EXECUTED for a handful of examples. Checking
    only the names is not enough, and the first live run said so: the model
    proposed `tail_digits=0`, which raises inside the generator, and
    `domain="integers"`, which is not a value `mul_pairs` knows and silently
    yields abstract cells for a rule declared to read the screen. Neither is
    visible in a parameter name. Six generated examples cost nothing and turn
    both into a rejection here rather than a crash, or worse a quietly
    mistrained rule, several tool calls later.
    """
    if isinstance(proposal, Interconversion):
        return _validate_interconversion(proposal, library)

    problems: list[str] = []
    if proposal.generator not in GENERATORS:
        problems.append(f"no generator {proposal.generator!r}; "
                        f"known: {sorted(GENERATORS)}")
    if proposal.oracle not in ORACLES:
        problems.append(f"no oracle {proposal.oracle!r}; known: {sorted(ORACLES)}")
    for field_name, value in (("domain_in", proposal.domain_in),
                              ("domain_out", proposal.domain_out)):
        if value not in (ABSTRACT, SPECIFIC):
            problems.append(f"{field_name}={value!r} is not "
                            f"{ABSTRACT!r} or {SPECIFIC!r}")
    if not proposal.name.isidentifier():
        problems.append(f"{proposal.name!r} is not a usable rule name")
    if library is not None and proposal.name in library:
        problems.append(f"{proposal.name!r} already exists; "
                        "a proposal must not silently replace a verified rule")
    if not proposal.from_oracle and proposal.num_slots <= 0:
        problems.append("give num_slots, or from_oracle for a choice rule")
    if proposal.generator in GENERATORS:
        allowed = set(GENERATORS[proposal.generator].params)
        unknown = sorted(set(proposal.generator_params) - allowed)
        if unknown:
            problems.append(f"{proposal.generator!r} takes no parameter(s) "
                            f"{unknown}; it takes {sorted(allowed)}")
    if proposal.generator in GENERATORS and proposal.known:
        allowed = set(GENERATORS[proposal.generator].params)
        stray = sorted(set(proposal.known) - allowed)
        if stray:
            problems.append(f"the known family names {stray}, which "
                            f"{proposal.generator!r} does not take")
    if proposal.known_oracle and proposal.known_oracle not in ORACLES:
        problems.append(f"no oracle {proposal.known_oracle!r} to be established "
                        f"in; known: {sorted(ORACLES)}")
    # Identical families are fine when the FORM differs -- "the same rewrite on
    # the other factor" is a transfer over exactly the same numbers, and that
    # is the sharpest kind. It is only vacuous when nothing at all changes.
    if (proposal.known and proposal.unknown
            and proposal.known == proposal.unknown
            and not proposal.transfers_form()):
        problems.append("the known and unknown families are identical and the "
                        "pattern is the same one, so there is no transfer "
                        "being claimed")
    if dry_run and not problems:
        problems += _dry_run(proposal)
    return problems


def _validate_interconversion(proposal: "Interconversion",
                              library=None) -> list[str]:
    """An interconversion names statements, not registries -- so what has to be
    true of it is different: two DIFFERENT statements, in a real domain, and a
    suggested route made only of rules that exist.

    A wrong route is not a wrong claim, so unknown names in `via` are dropped
    rather than fatal. They do get dropped: the first live run suggested a
    route of ORACLE names, and a summary presenting those as a chain of rules
    reads like a plan the machine could follow when it is not one.
    """
    problems: list[str] = []
    if library is not None and proposal.via:
        real = [n for n in proposal.via if n in library]
        if real != proposal.via:
            dropped = [n for n in proposal.via if n not in library]
            proposal.via = real
            proposal.rationale += (f"  [dropped from the suggested route, not "
                                   f"rules in this library: {dropped}]")
    if not proposal.source.strip() or not proposal.target.strip():
        problems.append("an interconversion needs both a source and a target "
                        "statement; one of them is empty")
    if proposal.source.strip() == proposal.target.strip():
        problems.append("source and target are the same statement, so there is "
                        "no correspondence to build")
    if proposal.domain not in (ABSTRACT, SPECIFIC):
        problems.append(f"domain={proposal.domain!r} is not "
                        f"{ABSTRACT!r} or {SPECIFIC!r}")
    if not proposal.name.isidentifier():
        problems.append(f"{proposal.name!r} is not a usable name")
    return problems


def _dry_run(proposal: RuleProposal, n: int = 6) -> list[str]:
    """Run the proposed recipe for a few examples and report what breaks."""
    from .generators import generate

    try:
        example_set = generate(proposal.generator, n, seed=0,
                               **proposal.generator_params)
    except Exception as err:
        return [f"generate_data({proposal.generator}, "
                f"{json.dumps(proposal.generator_params)}) fails: "
                f"{type(err).__name__}: {err}"]
    if not example_set.examples:
        return [f"{proposal.generator!r} produced nothing with those parameters"]

    got = example_set.examples[0].inp.domain
    if got != proposal.domain_in:
        return [f"the rule reads {proposal.domain_in} cells but "
                f"{proposal.generator!r} with those parameters writes {got} "
                f"ones -- for mul_pairs that is the 'domain' parameter, which "
                f"takes exactly 'abstract' or 'specific'"]

    fn = ORACLES[proposal.oracle].fn
    labelled = 0
    for ex in example_set.examples:
        try:
            labelled += fn(ex) is not None
        except Exception as err:
            return [f"oracle {proposal.oracle!r} raises on "
                    f"{proposal.generator!r} output: {type(err).__name__}: {err}"]
    if not labelled:
        return [f"oracle {proposal.oracle!r} declines every example "
                f"{proposal.generator!r} produces, so there would be nothing to "
                f"train on -- the two do not go together"]
    return []


# ---------------------------------------------------------------------------
# The offline proposer
# ---------------------------------------------------------------------------
def heuristic_proposals(machine, analogy: Analogy,
                        max_proposals: int = 3) -> list[RuleProposal]:
    """Propose by trying every oracle on the unsolved cases and keeping what
    applies -- no language model involved.

    This is deliberately not clever. It cannot see that two drawings share a
    place-value split; it can only notice that some oracle produces an answer
    for the cases nothing currently solves, and that no existing rule already
    learned that oracle. That is enough to keep the pipeline runnable offline
    and to give the LLM proposer something to be compared against, and the gap
    between the two is the honest measure of what the controller contributes.
    """
    taught = {r.recipe.oracle for r in machine.library
              if getattr(r, "recipe", None) and r.recipe.oracle}
    wanted = [split_case(c.text)[0] for c in analogy.unsolved]
    if not wanted:
        return []                           # nothing is unsolved; nothing to propose

    scored: list[tuple[int, str]] = []
    for name, spec in ORACLES.items():
        if name in taught or name not in _RECIPES:
            continue                        # already learned, or unpaired
        hits = 0
        for text in wanted:
            try:
                out = spec.fn(_as_example(text))
            except Exception:
                out = None
            # An oracle that echoes its input teaches nothing.
            if out is not None and out.text and out.text != text:
                hits += 1
        if hits:
            scored.append((hits, name))
    scored.sort(key=lambda t: (-t[0], t[1]))

    out: list[RuleProposal] = []
    for hits, oracle_name in scored[:max_proposals]:
        recipe = _RECIPES[oracle_name]
        out.append(RuleProposal(
            name=f"{oracle_name}_learned",
            domain_in=recipe["domain_in"], domain_out=recipe["domain_out"],
            generator=recipe["generator"], oracle=oracle_name,
            # One family, honestly reported as such. This proposer has no way
            # to tell which parameters separate the established instances from
            # the conjectured ones -- reading that off the cases is exactly the
            # part it cannot do -- so it claims the weaker thing and the
            # summary says "the solved cases" rather than naming a family.
            unknown=dict(recipe["params"]),
            num_slots=recipe.get("num_slots", 0),
            from_oracle=bool(ORACLES[oracle_name].classes),
            rationale=(f"{oracle_name} produces an answer for {hits} of the "
                       f"{len(wanted)} unsolved case(s), and no rule in the "
                       f"library has been taught it yet"),
            proposed_by="heuristic",
        ))
    return out


def _as_example(text: str):
    from .dataset import Example

    return Example(inp=Content.specific_text(text))


# How each oracle is turned into a trainable rule: which generator makes inputs
# it can label, and what shape the rule's output is. Stated rather than
# inferred -- the output DOMAIN especially, since `read_back` and
# `distributive_rewrite` both emit plain strings and only one of them is a
# reader. Guessing that from the oracle's return value would get it wrong.
_RECIPES: dict[str, dict[str, Any]] = {
    "distributive_rewrite": {
        "generator": "mul_pairs", "domain_in": SPECIFIC, "domain_out": SPECIFIC,
        "num_slots": 16,
        "params": {"a_digits": 2, "b_digits": 2, "domain": "specific"}},
    "distributive_rewrite_right": {
        "generator": "mul_pairs", "domain_in": SPECIFIC, "domain_out": SPECIFIC,
        "num_slots": 16,
        "params": {"a_digits": 2, "b_digits": 2, "domain": "specific"}},
    "product_by_definition": {
        "generator": "mul_pairs", "domain_in": SPECIFIC, "domain_out": ABSTRACT,
        "num_slots": 8,
        "params": {"a_digits": 2, "b_digits": 2, "domain": "specific"}},
    "read_back": {
        "generator": "rendered_expressions", "domain_in": SPECIFIC,
        "domain_out": ABSTRACT, "num_slots": 6,
        "params": {"max_terms": 1, "digits": 2}},
    "next_construction_step": {
        "generator": "triangle_scenes", "domain_in": SPECIFIC,
        "domain_out": SPECIFIC, "params": {"solved_fraction": 0.3}},
    "alternate_angle_facts": {
        "generator": "triangle_scenes", "domain_in": SPECIFIC,
        "domain_out": ABSTRACT, "params": {"solved_fraction": 0.4}},
    "escape_direction": {
        "generator": "robot_scenes", "domain_in": SPECIFIC,
        "domain_out": ABSTRACT, "params": {"n_obstacles": 2}},
}


# ---------------------------------------------------------------------------
# The LLM proposer
# ---------------------------------------------------------------------------
PROPOSE_PROMPT = """\
You are choosing what a Ren machine should try to learn next.

Below are worked cases from its benchmark. Some it can already derive with the
rules it has; some it cannot. EVERY case is shown to you twice: once as the
caption it was drawn from, and once as the actual cell on the SPECIFIC tape,
where structure lives as layout rather than as syntax.

For each case you are told what its cell DRAWS: how many boxes the expression
was laid out as, what joins them, and which factors are stacked inside each
box. Read the layout, not the caption. The regrouping that makes an unsolved
case tractable is usually visible as an arrangement the solved cases share and
the unsolved one does not yet have -- a term splitting into two boxes at a
place-value boundary, say, rather than any change to the characters.

Propose up to {max_proposals} hypotheses. Each is a CLAIM that can be wrong,
and there are exactly two shapes it may take.

SHARED PATTERN -- "the pattern that holds on the known instances also holds
on the unknown ones, under <condition>."

  Use this when the cases share a structure. They may be instances of one
  problem (a rewrite verified for two-digit products, claimed for three-digit
  ones) or two problems with the same shape (what is proven of the 2-D case,
  claimed for the 3-D case under the right hypotheses). Name the family where
  the pattern is ALREADY ESTABLISHED and the family where you are CLAIMING it
  -- as two parameter sets for one generator -- because "what holds there also
  holds here" is the claim, and a single family cannot state it.

  The pattern may also be carried to a different FORM of itself, which is
  often the sharper claim. Set "known_oracle" to the form it already holds in
  and "oracle" to the form you are claiming: the distributive law established
  as a split of the left factor, claimed as a split of the right one, over the
  very same numbers. Leave "known_oracle" out when the form is unchanged and
  only the instances differ.

INTERCONVERSION -- "the unproven case maps to a solved one, so the unproven one
is established by transport."

  Use this when the work is not a shared pattern but a CORRESPONDENCE, the way
  Fermat's Last Theorem was carried by the semistable case of Taniyama-Shimura.
  You are asking the machine to search for a chain of mapping rules from the
  unproven statement to the established one, or back. Give the two statements;
  the search finds the chain.

You are not writing code and not inventing data. Every generator and oracle you
name must be one of these, and the parameters must be ones it accepts:

GENERATORS
{generators}

ORACLES
{oracles}

Answer with a JSON array and no other text.

A shared-pattern element:

  {{"kind": "shared_pattern",
    "name": "snake_case_identifier",
    "domain_in": "specific" | "abstract",
    "domain_out": "specific" | "abstract",
    "generator": "<a name above>",
    "known":   {{<params for the family where it is established>}},
    "unknown": {{<params for the family you are claiming it extends to>}},
    "known_oracle": "<the form it already holds in; omit if unchanged>",
    "condition": "<what is supposed to license carrying it across>",
    "oracle": "<a name above -- the form you are claiming>",
    "num_slots": <int, the longest output string; omit if from_oracle>,
    "from_oracle": <true for a rule that picks one of the oracle's classes>,
    "rationale": "<what the layouts share, in one sentence>"}}

An interconversion element:

  {{"kind": "interconversion",
    "name": "snake_case_identifier",
    "source": "<the unproven statement, as it would be written on the tape>",
    "target": "<the established one>",
    "via": ["<optional suggested rule names>"],
    "both_ways": <true if you claim a one-to-one correspondence>,
    "domain": "abstract" | "specific",
    "rationale": "<why these two correspond, in one sentence>"}}
"""


def llm_proposals(machine, analogy: Analogy, client: Any = None,
                  model: str = "claude-opus-5", max_proposals: int = 3,
                  max_tokens: int = 4000, form: str = "text",
                  log: Callable[[str], None] = lambda _s: None) -> list[RuleProposal]:
    """Ask the controller to summarise the analogy as rules worth testing.

    The cases go up as the SPECIFIC-DOMAIN CELL, never as the caption they were
    drawn from -- otherwise the analogy would be found in the symbols, which is
    the easier question the architecture exists to avoid.

    `form="text"` is the default and reports what the cell DRAWS in ordinary
    readable text: how many boxes the expression was laid out as, what joins
    them, and which factors are stacked inside each box, with every glyph
    decoded to the characters it came from. That is the layout an analogy is
    supposed to be read off -- `12*30` is one box and `10*30+2*30` is two
    joined by `+`, so the distributive regrouping is a change in the number of
    boxes rather than a string edit -- and it arrives as text a model can
    actually read, which pixel-class art is not.

    `form="image"` sends the rendered PNG instead, for cells where the layout
    is genuinely pictorial: a geometry sketch says more as pixels than any
    description of it does.

    Anything that comes back naming a generator or oracle outside the
    registries is dropped with its reason logged, so a hallucinated name costs
    one rejected proposal rather than a confusing failure several tool calls
    later.
    """
    import base64

    from .render import png_bytes

    if client is None:
        try:
            import anthropic
        except ImportError as err:           # pragma: no cover
            raise RuntimeError(
                "proposing with the LLM needs the anthropic package: "
                "pip install anthropic (or use heuristic_proposals)"
            ) from err
        from .controller import load_credentials

        load_credentials()
        client = anthropic.Anthropic()

    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": PROPOSE_PROMPT.format(
            max_proposals=max_proposals,
            # Parameter DESCRIPTIONS, not just names. The registry documents
            # what each one accepts -- `domain` takes 'abstract' or 'specific'
            # and nothing else -- and a prompt listing bare names invites
            # plausible inventions like domain="integers", which the dry run
            # then rejects for a reason the model was never told.
            generators="\n".join(
                f"  {s.name}: {s.doc}\n" + (
                    "\n".join(f"    - {k}: {v}" for k, v in s.params.items())
                    or "    (no parameters)")
                for s in GENERATORS.values()),
            oracles="\n".join(f"  {s.name} [{s.kind}]: {s.doc}"
                              + (f"\n    classes: {s.classes}" if s.classes else "")
                              for s in ORACLES.values()),
        ),
    }]
    content.append({"type": "text", "text": f"\nGoal: {analogy.goal}\n"})
    for case in analogy.solved + analogy.unsolved:
        content.append({"type": "text", "text": case.caption()})
        if form == "image" and case.image is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": base64.b64encode(png_bytes(case.image)).decode()},
            })
        elif case.view_text:
            content.append({"type": "text", "text": case.view_text})

    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return parse_proposals(text, machine.library, log=log, proposed_by=model)


def parse_proposals(text: str, library=None, log: Callable[[str], None] = lambda _s: None,
                    proposed_by: str = "llm") -> list[RuleProposal]:
    """JSON from the model -> validated proposals. Invalid ones are dropped."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        log("  no JSON array in the reply; nothing proposed")
        return []
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError as err:
        log(f"  reply was not valid JSON ({err}); nothing proposed")
        return []

    out: list[Any] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "shared_pattern"))
        if kind == "interconversion":
            p: Any = Interconversion(
                name=str(item.get("name", "")),
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                via=[str(v) for v in (item.get("via") or [])],
                both_ways=bool(item.get("both_ways")),
                domain=str(item.get("domain", ABSTRACT)),
                rationale=str(item.get("rationale", "")),
                proposed_by=proposed_by,
            )
        else:
            # `generator_params` is still read, so a model that gives one
            # family instead of two is understood rather than discarded -- it
            # has just made the weaker claim, and the summary shows that.
            flat = dict(item.get("generator_params") or {})
            p = RuleProposal(
                name=str(item.get("name", "")),
                domain_in=str(item.get("domain_in", SPECIFIC)),
                domain_out=str(item.get("domain_out", ABSTRACT)),
                generator=str(item.get("generator", "")),
                oracle=str(item.get("oracle", "")),
                known=dict(item.get("known") or {}),
                unknown=dict(item.get("unknown") or flat),
                known_oracle=str(item.get("known_oracle", "")),
                condition=str(item.get("condition", "")),
                num_slots=int(item.get("num_slots") or 0),
                from_oracle=bool(item.get("from_oracle")),
                rationale=str(item.get("rationale", "")),
                proposed_by=proposed_by,
            )
        problems = validate(p, library)
        if problems:
            log(f"  rejected {p.name or '<unnamed>'}: {'; '.join(problems)}")
            continue
        out.append(p)
    return out
