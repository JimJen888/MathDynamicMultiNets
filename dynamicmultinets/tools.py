"""
The controller's instruction set.

Every action the controller can take is one of these, and they are the only
things it can take. The list is the machine's ISA: read and write the two
tapes, run an experiment, label it, form a rule, train it, verify it, apply it,
combine rules, distill a chain into one rule, search for a proof, price the
library, and shrink it.

Each tool is provider-neutral -- name, description, JSON schema, and a Python
callable -- so the same table drives the LLM controller and the scripted one.
That matters for testing: a scripted plan exercises exactly the code path an
LLM would drive, so a demo that runs offline is not a different program.

Two conventions the descriptions rely on:

  * Tools return short human-readable strings, not JSON blobs. The controller
    is a language model reading a terminal, and a paragraph it can act on beats
    a nested object it has to parse.
  * A failed tool call returns the error as a string rather than raising. A
    controller that gets "rule X does not apply to that cell because ..." can
    recover; one that gets a stack trace and a dead session cannot.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable

from . import generators, oracles
from .machine import RenMachine
from .tapes import ABSTRACT, SPECIFIC


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]

    def call(self, **kwargs: Any) -> str:
        try:
            return self.fn(**kwargs)
        except Exception as err:                  # controllers recover, sessions don't die
            return (f"ERROR in {self.name}: {type(err).__name__}: {err}\n"
                    + "".join(traceback.format_exc(limit=2).splitlines(True)[-2:]))


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}


def build_tools(m: RenMachine) -> dict[str, Tool]:
    """Bind the instruction set to one machine."""
    tools: list[Tool] = []

    def tool(name: str, description: str, properties: dict[str, Any],
             required: list[str] | None = None):
        def wrap(fn: Callable[..., str]) -> Callable[..., str]:
            tools.append(Tool(name, description, _schema(properties, required or []), fn))
            return fn

        return wrap

    # -- looking around ------------------------------------------------------
    @tool("inspect_machine",
          "Show the two tapes, every rule with its cost and verification status, "
          "and the datasets. Call this first, and again whenever you are unsure "
          "what state the machine is in.",
          {})
    def inspect_machine() -> str:
        return m.state()

    @tool("show_catalogue",
          "List the available data generators and labelling oracles, with the "
          "parameters each accepts. You must pick generators and oracles from "
          "this list by name; you cannot write new ones.",
          {})
    def show_catalogue() -> str:
        return ("GENERATORS (make experiments):\n" + generators.catalogue()
                + "\n\nORACLES (label them; 'kind' is how strong the evidence is):\n"
                + oracles.catalogue())

    @tool("show_trace", "Show the machine's recent actions.",
          {"limit": {**_INT, "description": "how many entries (default 30)"}})
    def show_trace(limit: int = 30) -> str:
        return m.trace(limit)

    # -- the tapes -----------------------------------------------------------
    @tool("write_tape",
          "Write content to a tape. domain='abstract' writes symbols; "
          "domain='specific' renders those symbols as an image on the specific "
          "tape. Use this to set up the expression or statement you want to work on.",
          {"domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]},
           "text": {**_STR, "description": "the symbols to write or draw"}},
          ["domain", "text"])
    def write_tape(domain: str, text: str) -> str:
        return m.write(domain, text).summary()

    @tool("put_example_on_tape",
          "Copy one generated experiment onto its tape, so you can reason about "
          "it. Use this to start from a sketch, which has no textual form to type.",
          {"dataset": _STR, "index": _INT}, ["dataset"])
    def put_example_on_tape(dataset: str, index: int = 0) -> str:
        return m.put_example(dataset, index).summary()

    @tool("apply_rule",
          "Apply one rule to the head cell of its input tape and write the "
          "result to its output tape. This is the machine's single primitive "
          "step. Fails if the rule does not match the cell.",
          {"rule": _STR}, ["rule"])
    def apply_rule(rule: str) -> str:
        return m.apply_rule(rule).summary()

    # -- forming a rule ------------------------------------------------------
    @tool("generate_data",
          "Run a generator to produce experiments (inputs only, no labels yet). "
          "Call this when you have decided which rule to form and need data to "
          "form it from. Extra generator parameters go in 'params'.",
          {"generator": _STR, "n": {**_INT, "description": "how many examples"},
           "seed": _INT, "name": {**_STR, "description": "name to store it under"},
           "params": {"type": "object", "description": "generator-specific parameters"}},
          ["generator", "n"])
    def generate_data(generator: str, n: int, seed: int = 0, name: str = "",
                      params: dict | None = None) -> str:
        es = m.generate_data(generator, n, seed=seed, name=name or None, **(params or {}))
        sample = "; ".join(e.inp.text for e in es.examples[:3])
        return f"{es.summary()}\n  first inputs: {sample}"

    @tool("label_data",
          "Label a dataset with an oracle. Check the oracle's 'kind' first: "
          "'definitional' evidence is strong, 'constructed' only tells you the "
          "machine can read its own handwriting.",
          {"dataset": _STR, "oracle": _STR}, ["dataset", "oracle"])
    def label_data(dataset: str, oracle: str) -> str:
        es = m.label_data(dataset, oracle)
        pairs = "; ".join(e.summary() for e in es.labeled[:3])
        return f"{es.summary()}\n  first pairs: {pairs}"

    @tool("declare_rule",
          "Create a new, untrained learned rule -- decide what mapping it "
          "performs. If it is meant to be trained by an oracle with named "
          "classes, pass from_oracle and the output shape is taken from it. "
          "For a rule that rewrites an expression, leave from_oracle empty and "
          "set num_slots to the longest output you expect.",
          {"name": _STR,
           "domain_in": {**_STR, "enum": [ABSTRACT, SPECIFIC]},
           "out_domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]},
           "description": _STR,
           "from_oracle": {**_STR, "description": "oracle whose classes define the output"},
           "num_slots": {**_INT, "description": "output length for a rewrite rule"},
           "kind": {**_STR, "enum": ["auto", "scene_action"],
                    "description": "'scene_action' for a rule that edits a sketch "
                                   "and writes the redrawn scene back"}},
          ["name", "domain_in", "out_domain"])
    def declare_rule(name: str, domain_in: str, out_domain: str, description: str = "",
                     from_oracle: str = "", num_slots: int = 0,
                     kind: str = "auto") -> str:
        rule = m.declare_rule(name, domain_in, out_domain, description,
                              num_slots=num_slots or None,
                              from_oracle=from_oracle or None, kind=kind)
        return (f"declared {rule.signature()} -- untrained and untrusted; "
                f"train it, then verify it before using it in a proof")

    @tool("train_rule",
          "Fit a declared rule on a labeled dataset. Reports holdout accuracy; "
          "training alone never makes a rule trusted.",
          {"rule": _STR, "dataset": _STR, "epochs": _INT}, ["rule", "dataset"])
    def train_rule(rule: str, dataset: str, epochs: int = 12) -> str:
        return m.train(rule, dataset, epochs=epochs).summary()

    @tool("grow_ensemble",
          "Find the cases a trained rule still gets wrong, train a specialist on "
          "them, and compose the two so the specialist overrides the base where "
          "it applies. Use this when a rule verifies well but not well enough.",
          {"rule": _STR, "dataset": _STR, "epochs": _INT}, ["rule", "dataset"])
    def grow_ensemble(rule: str, dataset: str, epochs: int = 12) -> str:
        from .rules import EnsembleRule

        ens, report, note = m.grow_ensemble(rule, dataset, epochs=epochs)
        if report is None:
            # Not necessarily "this rule has no hard cases": mining the base's
            # own training data finds none however wrong the rule is on fresh
            # data, and the note says which of the two this was.
            return f"no specialist for {rule}; left unchanged -- {note}"
        if not isinstance(ens, EnsembleRule):
            # The specialist was trained and then thrown away: it did not beat
            # the base on held-out examples. Saying "created" here would be a
            # lie the controller then acts on.
            return (f"trained a specialist for {rule} and DISCARDED it -- it did "
                    f"not improve on the base rule where it claims competence "
                    f"({note}). {rule} is unchanged. ({report.summary()})")
        return (f"created ensemble {ens.name!r} = {rule} + specialist override "
                f"({note}); it is a new rule and needs verifying on fresh data "
                f"before use. specialist {report.summary()}")

    # -- verification --------------------------------------------------------
    @tool("verify_rule",
          "Check a rule on FRESH data against an oracle. This is what makes a "
          "rule trusted and therefore usable in a proof. Generate a new dataset "
          "with a different seed (or a harder tail) before calling this.",
          {"rule": _STR, "dataset": _STR, "oracle": _STR,
           "threshold": {**_NUM, "description": "accuracy needed to trust it (default 0.99)"}},
          ["rule", "dataset"])
    def verify_rule(rule: str, dataset: str, oracle: str = "",
                    threshold: float = 0.99) -> str:
        return m.verify(rule, dataset, oracle or None, threshold).summary()

    @tool("verify_against_rules",
          "Check a rule against a chain of rules the machine already trusts, "
          "instead of an oracle. Use this when a learned rule should agree with "
          "something derivable -- it is how a rule learned from pictures gets "
          "checked against algebra.",
          {"rule": _STR, "reference_chain": {"type": "array", "items": _STR},
           "dataset": _STR, "threshold": _NUM},
          ["rule", "reference_chain", "dataset"])
    def verify_against_rules(rule: str, reference_chain: list, dataset: str,
                             threshold: float = 0.99) -> str:
        return m.verify_via_rules(rule, reference_chain, dataset, threshold).summary()

    # -- making rules more concise ------------------------------------------
    @tool("compose_rules",
          "Name a sequence of rules as a single new rule, so a derivation you "
          "keep repeating becomes one move.",
          {"rules": {"type": "array", "items": _STR}, "new_name": _STR,
           "description": _STR},
          ["rules", "new_name"])
    def compose_rules(rules: list, new_name: str, description: str = "") -> str:
        r = m.compose(rules, new_name, description)
        return (f"{new_name}: {r.steps()} primitive steps, {r.cost_bits():.0f} bits, "
                f"confidence {r.confidence():.4f}, "
                f"{'trusted' if r.trusted else 'NOT trusted (a member is unverified)'}")

    @tool("propose_rules",
          "Decide WHAT to learn next. Give cases the machine cannot derive and "
          "cases it can; both are drawn onto the specific tape and compared "
          "there, and what they share comes back as rules worth forming -- a "
          "generator, an oracle and an output shape, ready for declare_rule. "
          "Call this before declare_rule when you do not already know which "
          "rule is missing. Nothing is added to the library: a proposal is a "
          "question, and verify_rule is what answers it.",
          {"unsolved": {"type": "array", "items": _STR,
                        "description": "cases no current rule chain reaches"},
           "solved": {"type": "array", "items": _STR,
                      "description": "cases it can already derive, for contrast"},
           "use_llm": {"type": "boolean",
                       "description": "summarise the drawings with the model "
                                      "(needs credentials); otherwise a much "
                                      "weaker offline heuristic is used"},
           "max_proposals": _INT,
           "domain": {**_STR, "enum": [ABSTRACT, SPECIFIC],
                      "description": "where the cases start. 'specific' poses "
                                     "them as drawings, which is a different "
                                     "and usually harder question"},
           "observed": {"type": "boolean",
                        "description": "true for drawings the machine did not "
                                       "write itself, so the caption cannot be "
                                       "copied instead of read"},
           "solved_domain": {**_STR, "enum": [ABSTRACT, SPECIFIC],
                             "description": "where the SOLVED cases are posed, "
                                            "when that differs. Worked instances "
                                            "are usually things the machine can "
                                            "do as symbols while the open cases "
                                            "are drawings; posing both as "
                                            "drawings leaves nothing solved and "
                                            "nothing to compare"},
           "form": {**_STR, "enum": ["text", "image"],
                    "description": "how the specific-domain cells are shown: "
                                   "'text' (default) writes each cell out as "
                                   "characters at the resolution it was drawn; "
                                   "'image' sends the rendered pixels, worth it "
                                   "when layout is genuinely pictorial"},
           "show_cells": {"type": "boolean",
                          "description": "include the cells in the reply, so "
                                         "the analogy can be read here too"}},
          ["unsolved"])
    def propose_rules(unsolved: list, solved: list = None, use_llm: bool = False,
                      max_proposals: int = 3, domain: str = ABSTRACT,
                      observed: bool = False, solved_domain: str = "",
                      form: str = "text", show_cells: bool = False) -> str:
        analogy, proposals = m.propose_rules(unsolved, solved or [],
                                             use_llm=use_llm,
                                             max_proposals=max_proposals,
                                             domain=domain, observed=observed,
                                             solved_domain=solved_domain or None,
                                             form=form)
        head = analogy.summary(cells=show_cells)
        if not proposals:
            return (f"{head}\n\nNo proposal survived validation. "
                    "Every rule must come from a registered generator and "
                    "oracle; show_catalogue lists them.")
        rows = []
        for p in proposals:
            # Each kind knows its own test, and running it here is cheap: a
            # transfer that the property does not even apply to is refuted
            # before a net exists, and a correspondence either has a chain or
            # does not. Reporting the claim without testing it would hand the
            # controller a hypothesis dressed as a finding.
            rows.append(p.summary() + "\n    checked: " + p.check(m))
        return (f"{head}\n\n" + "\n\n".join(rows)
                + "\n\nNone of these is trusted, or even declared. A shared "
                  "property goes through generate_data -> label_data -> "
                  "declare_rule -> train_rule -> verify_rule; an "
                  "interconversion is settled by prove.")

    @tool("iterate_rule",
          "Turn a rule that stays in one domain into one that runs its own "
          "loop and keeps the BEST cell it produced, judged by another rule. "
          "Use this for a construction: proof search follows one action per "
          "drawing, so a single wrong step ends the proof with nowhere to go, "
          "and folding the loop inside makes a wrong step just one more "
          "candidate. The judge must be a different rule -- a rule scoring its "
          "own output is not evidence.",
          {"step": _STR, "judge": _STR,
           "judge_target": {**_STR,
                            "description": "the judge's answer that means "
                                           "'this cell is the wanted one'"},
           "new_name": _STR, "max_iters": _INT, "description": _STR},
          ["step", "judge", "judge_target"])
    def iterate_rule(step: str, judge: str, judge_target: str, new_name: str = "",
                     max_iters: int = 12, description: str = "") -> str:
        r = m.iterate_rule(step, judge, judge_target, new_name, max_iters, description)
        return (f"{r.name}: {step} up to {max_iters} times, best cell chosen by "
                f"{judge}=={judge_target!r}, {r.cost_bits():.0f} bits, "
                f"confidence {r.confidence():.4f}, "
                f"{'trusted' if r.trusted else 'NOT trusted (a member is unverified)'}")

    @tool("distill_rule",
          "Train a single net to imitate an existing rule or chain, collapsing "
          "it to one step. The distilled rule is a new empirical claim: verify "
          "it before trusting it.",
          {"source": _STR, "new_name": _STR, "dataset": _STR, "epochs": _INT},
          ["source", "new_name", "dataset"])
    def distill_rule(source: str, new_name: str, dataset: str, epochs: int = 12) -> str:
        rule, report, agreement = m.distill(source, new_name, dataset, epochs=epochs)
        return (f"{report.summary()}\n  agreement with {source}: {agreement:.3f}; "
                f"{m.library.get(source).steps()} steps collapsed to 1")

    @tool("library_report",
          "Price the library: description bits plus the rule applications needed "
          "to solve the benchmark tasks. Lower J is better. This is the "
          "objective you are minimising -- consult it before and after any "
          "change to the library.",
          {"trusted_only": _BOOL})
    def library_report(trusted_only: bool = True) -> str:
        return m.report(trusted_only).summary()

    @tool("simplify_library",
          "Propose (and optionally make) the library smaller: drop rules that "
          "duplicate another rule's behaviour, and rules no benchmark proof "
          "uses. Run it as a dry run first.",
          {"probe": {**_STR, "description": "dataset used to compare rule behaviour"},
           "apply_changes": _BOOL})
    def simplify_library(probe: str = "", apply_changes: bool = False) -> str:
        actions, before, after = m.simplify(probe or None, apply_changes)
        if not actions:
            return f"nothing to simplify; J = {before.objective:.0f} bits"
        head = "\n".join("  " + a.line() for a in actions)
        tail = (f"J {before.objective:.0f} -> {after.objective:.0f} bits"
                if after else f"dry run; J currently {before.objective:.0f} bits")
        return f"{len(actions)} actions:\n{head}\n{tail}"

    # -- proving -------------------------------------------------------------
    @tool("add_task",
          "Add a benchmark task: a starting cell and the result that must be "
          "reached. The benchmark defines what 'useful' means for the library, "
          "so add the tasks you actually care about before optimising.",
          {"name": _STR, "start": _STR, "target": _STR,
           "domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]},
           "observed": {**_BOOL, "description": "mark a specific-domain start as "
                                 "SEEN rather than drawn, so it carries no caption "
                                 "to copy and must actually be read"}},
          ["name", "start", "target"])
    def add_task(name: str, start: str, target: str, domain: str = ABSTRACT,
                 observed: bool = False) -> str:
        t = m.add_task(name, start, target, domain, observed=observed)
        return f"task {t.name}: {start!r} => {target!r} on the {domain} tape"

    @tool("add_task_from_example",
          "Add a benchmark task whose starting cell is a generated experiment "
          "(a sketch, say) rather than typed text.",
          {"name": _STR, "dataset": _STR, "index": _INT, "target": _STR,
           "max_depth": _INT},
          ["name", "dataset", "target"])
    def add_task_from_example(name: str, dataset: str, target: str, index: int = 0,
                              max_depth: int = 12) -> str:
        t = m.add_task_from_example(name, dataset, index, target, max_depth)
        return f"task {t.name}: {dataset}[{index}] => {target!r}"

    @tool("prove",
          "Search for a chain of rules leading from a starting cell to a target. "
          "Only trusted rules are used unless trusted_only is false. Chains may "
          "cross between the two domains.",
          {"start": _STR, "target": _STR,
           "domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]},
           "target_domain": {**_STR, "enum": [ABSTRACT, SPECIFIC],
                             "description": "where the conclusion must land "
                                            "(default abstract)"},
           "observed": _BOOL, "max_depth": _INT, "trusted_only": _BOOL},
          ["start", "target"])
    def prove(start: str, target: str, domain: str = ABSTRACT, max_depth: int = 6,
              trusted_only: bool = True, target_domain: str = ABSTRACT,
              observed: bool = False) -> str:
        return m.prove(start, target, domain, max_depth, trusted_only,
                       target_domain=target_domain, observed=observed).as_text()

    @tool("prove_from_tape",
          "Prove starting from the cell the head is pointing at, rather than "
          "from text you type. This is the only way to start from a drawing.",
          {"domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]}, "target": _STR,
           "target_domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]},
           "max_depth": _INT, "trusted_only": _BOOL},
          ["domain", "target"])
    def prove_from_tape(domain: str, target: str, target_domain: str = ABSTRACT,
                        max_depth: int = 8, trusted_only: bool = True) -> str:
        return m.prove_from_tape(domain, target, max_depth, trusted_only,
                                 target_domain).as_text()

    @tool("keep_proof",
          "Turn the last successful proof of the given start/target into a named "
          "rule, so the machine never has to search for it again.",
          {"start": _STR, "target": _STR, "new_name": _STR,
           "domain": {**_STR, "enum": [ABSTRACT, SPECIFIC]}, "max_depth": _INT},
          ["start", "target", "new_name"])
    def keep_proof(start: str, target: str, new_name: str, domain: str = ABSTRACT,
                   max_depth: int = 6) -> str:
        p = m.prove(start, target, domain, max_depth)
        if not p.found:
            return f"cannot keep a proof that was not found: {p.note}"
        r = m.keep_proof(p, new_name)
        return (f"kept {new_name} = {' -> '.join(p.rule_names())} "
                f"({'trusted' if r.trusted else 'not trusted'})")

    @tool("halting_budget",
          "Calibrate a search budget from the proofs found so far, using the "
          "statistical anytime algorithm: returns a step threshold with a stated "
          "error rate instead of an arbitrary depth limit. Solve some benchmark "
          "tasks first -- the proofs are the sample it calibrates from, and a "
          "tighter lam needs quadratically more of them.",
          {"eps": _NUM, "delta": _NUM,
           "lam": {**_NUM, "description": "must be < eps; omit to fit it to the "
                                          "number of proofs available"}})
    def halting_budget(eps: float = 0.05, delta: float = 0.05,
                       lam: float = 0.0) -> str:
        return m.halting_budget(eps=eps, delta=delta, lam=lam or None).summary()

    @tool("finish",
          "Stop. Call this when the goal is met or you cannot make further "
          "progress, with a summary of the rules formed and what they cost.",
          {"summary": _STR}, ["summary"])
    def finish(summary: str) -> str:
        m.note("finish", summary)
        return "stopped"

    return {t.name: t for t in tools}


def anthropic_schemas(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    """The tool table in Messages API form."""
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools.values()]
