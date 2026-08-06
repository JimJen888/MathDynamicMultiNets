"""
RenMachine: the controller, the two tapes, and the rule library in one object.

This is the assembled architecture of Figure 1. Everything the machine can do
is a method here, and everything a method does is written to the journal, so
the trace of a run is the trace of the paper's workflow diagrams.

The division of labour is worth stating plainly, because it is what the whole
package is arranged around:

    the machine   holds state and performs operations, all of them bounded and
                  none of them clever -- generate, label, train, verify,
                  compose, distill, prove, price, simplify.
    the controller decides WHICH operation to perform next, and on what. That
                  is the open-ended part, and it is where the LLM sits
                  (controller.py). It never executes code; it chooses among
                  these methods by name.

That split is deliberate. The paper's controller is instructed by the contents
of both tapes and decides what rule to form next; it is not a code generator,
and giving a language model an exec() would add a failure mode the architecture
does not call for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from . import compose as compose_mod
from . import generators, oracles, proof as proof_mod, verify as verify_mod
from .codec import ChoiceCodec, Codec, SceneActionCodec, TextSlotCodec
from .dataset import DatasetStore, ExampleSet
from .prior import install_prior_rules
from .rules import NeuralRule, Rule, RuleLibrary, default_device
from .tapes import ABSTRACT, AbstractTape, Content, SpecificTape


@dataclass
class JournalEntry:
    t: float
    op: str
    detail: str

    def line(self) -> str:
        return f"[{self.op}] {self.detail}"


class RenMachine:
    """A non-Turing machine: two domains, one library of mapping rules."""

    def __init__(self, name: str = "ren", device: str | None = None,
                 with_prior: bool = True, goal: str = ""):
        """`device=None` detects: a GPU is used when one is available.

        Pass "cpu" explicitly to pin it -- worth doing when you need two runs
        to match exactly, since cuDNN kernels do not reproduce CPU kernels
        bit-for-bit even under the same seed.
        """
        self.name = name
        self.device = device or default_device()
        self.goal = goal
        self.abstract = AbstractTape()
        self.specific = SpecificTape()
        self.library = RuleLibrary()
        self.datasets = DatasetStore()
        self.benchmark: list[compose_mod.Task] = []
        self.journal: list[JournalEntry] = []
        self.note("init", f"learned rules will run on {self.device!r}")
        if with_prior:
            install_prior_rules(self.library)
            self.note("init", f"installed {len(self.library)} prior rules")

    # -- journal -------------------------------------------------------------
    def note(self, op: str, detail: str) -> str:
        self.journal.append(JournalEntry(time.time(), op, detail))
        return detail

    def trace(self, limit: int = 30) -> str:
        return "\n".join(e.line() for e in self.journal[-limit:]) or "(nothing yet)"

    # -- tapes ---------------------------------------------------------------
    def tape(self, domain: str):
        return self.abstract if domain == ABSTRACT else self.specific

    def write(self, domain: str, text: str, highlight: str | None = None) -> Content:
        content = (Content.abstract(text) if domain == ABSTRACT
                   else Content.specific_text(text, highlight=highlight))
        self.tape(domain).append(content)
        self.note("write", f"{domain}: {text!r}")
        return content

    def write_scene(self, scene: dict, caption: str = "scene") -> Content:
        content = Content.specific_sketch(scene, caption=caption)
        self.specific.append(content)
        self.note("write", f"specific: sketch {caption!r}")
        return content

    def head(self, domain: str) -> Content:
        return self.tape(domain).read()

    def state(self) -> str:
        return (
            f"machine {self.name!r}  device: {self.device}  "
            f"goal: {self.goal or '(none set)'}\n"
            f"--- abstract tape (head at {self.abstract.position}) ---\n"
            f"{self.abstract.dump()}\n"
            f"--- specific tape (head at {self.specific.position}) ---\n"
            f"{self.specific.dump()}\n"
            f"--- rules ({len(self.library)}) ---\n{self.library.table()}\n"
            f"--- datasets ---\n{self.datasets.table()}"
        )

    # -- rule application ----------------------------------------------------
    def apply_rule(self, rule_name: str, domain: str | None = None) -> Content:
        """Apply a rule to the head cell of its input tape and write the result
        to the head of its output tape. The one primitive step of the machine."""
        rule = self.library.get(rule_name)
        source = domain or rule.domain_in
        cell = self.head(source)
        out = rule.apply(cell)
        if out is None:
            raise ValueError(
                f"{rule_name!r} does not apply to {cell.summary()} -- check the "
                "input domain and the cell's form"
            )
        self.tape(out.domain).append(out)
        self.note("apply", f"{rule_name}: {cell.text!r} -> {out.text!r}")
        return out

    # -- experiments ---------------------------------------------------------
    def generate_data(self, generator: str, n: int, seed: int = 0,
                      name: str | None = None, **params: Any) -> ExampleSet:
        es = generators.generate(generator, n, seed=seed, **params)
        if name:
            es.name = name
        self.datasets.put(es)
        self.note("generate", es.summary())
        return es

    def label_data(self, dataset: str, oracle: str) -> ExampleSet:
        es = self.datasets[dataset]
        oracles.label(es, oracle)
        self.note("label", f"{es.name} labeled by {oracle} "
                           f"[{oracles.oracle_kind(oracle)}]: "
                           f"{len(es.labeled)}/{len(es)} usable")
        return es

    def put_example(self, dataset: str, index: int = 0) -> Content:
        """Copy one experiment's input cell onto its tape.

        How a generated sketch becomes something the machine can reason about:
        generators make experiments, and this puts one under the head.
        """
        cell = self.datasets[dataset].examples[index].inp
        self.tape(cell.domain).append(cell)
        self.note("write", f"{cell.domain}: {dataset}[{index}] {cell.text!r}")
        return cell

    # -- rule formation ------------------------------------------------------
    def declare_rule(
        self,
        name: str,
        domain_in: str,
        out_domain: str,
        description: str = "",
        classes: Sequence[str] | None = None,
        num_slots: int | None = None,
        vocab: str | None = None,
        from_oracle: str | None = None,
        kind: str = "auto",
    ) -> NeuralRule:
        """Create an untrained learned rule.

        The output shape comes from one of two places, and picking it is the
        real decision: a rule whose oracle has named classes is a CHOICE rule
        (DetourNet's head), and a rule that emits an expression is a SLOT
        transduction. Passing `from_oracle` derives it automatically, which is
        the path the controller normally takes -- a rule that cannot express
        what its oracle produces is unlearnable, and catching that here is
        cheaper than discovering it after training.
        """
        codec: Codec
        if kind == "scene_action":
            # A rule that edits a drawing rather than describing it: the class
            # is an action, the output cell is the redrawn scene. Only consult
            # the oracle when there is one -- `oracle_classes("")` is a KeyError,
            # not a way of asking for the default action set.
            actions = list(classes) if classes else (
                oracles.oracle_classes(from_oracle) if from_oracle else [])
            codec = SceneActionCodec(actions or ("move_up", "rotate_cw",
                                                 "rotate_ccw", "done"))
        elif from_oracle and oracles.oracle_classes(from_oracle):
            codec = ChoiceCodec(oracles.oracle_classes(from_oracle), out_domain=out_domain)
        elif classes:
            codec = ChoiceCodec(list(classes), out_domain=out_domain)
        else:
            codec = TextSlotCodec(num_slots=num_slots or 24,
                                  vocab=vocab or TextSlotCodec().vocab,
                                  out_domain=out_domain)
        rule = NeuralRule(name, codec, domain_in, description, device=self.device)
        self.library.add(rule, replace=True)
        self.note("declare", f"{rule.signature()} via {type(codec).__name__} "
                             f"({codec.num_slots} slots x {codec.num_classes} classes)")
        return rule

    def train(self, rule_name: str, dataset: str, epochs: int = 12, seed: int = 0,
              log=None, **kwargs):
        from .train import train_rule

        rule = self.library.get(rule_name)
        if not isinstance(rule, NeuralRule):
            raise TypeError(f"{rule_name!r} is not a learned rule")
        report = train_rule(rule, self.datasets[dataset], epochs=epochs, seed=seed,
                            log=log, **kwargs)
        self.note("train", report.summary())
        return report

    def grow_ensemble(self, rule_name: str, dataset: str, epochs: int = 12, log=None):
        from .rules import EnsembleRule
        from .train import grow_ensemble

        rule, report, note = grow_ensemble(self.library, rule_name,
                                           self.datasets[dataset], epochs=epochs,
                                           log=log)
        # Three outcomes, not two: a report exists both when the specialist was
        # kept and when it was trained and thrown away for failing to beat the
        # base. Only the returned rule's type distinguishes them.
        if isinstance(rule, EnsembleRule):
            outcome = "specialist added"
        elif report is None:
            outcome = "too few failures, unchanged"
        else:
            outcome = "specialist trained but discarded (no held-out gain), unchanged"
        self.note("ensemble", f"{rule.name}: {outcome} -- {note}")
        return rule, report, note

    # -- verification --------------------------------------------------------
    def verify(self, rule_name: str, dataset: str, oracle: str | None = None,
               threshold: float = 0.99):
        report = verify_mod.verify_rule(self.library, rule_name, self.datasets[dataset],
                                        oracle, threshold)
        self.note("verify", report.summary().splitlines()[0])
        return report

    def verify_via_rules(self, rule_name: str, reference_chain: Sequence[str],
                         dataset: str, threshold: float = 0.99):
        report = verify_mod.verify_against_rules(self.library, rule_name,
                                                 reference_chain,
                                                 self.datasets[dataset], threshold)
        self.note("verify", report.summary().splitlines()[0])
        return report

    # -- composing and shrinking --------------------------------------------
    def compose(self, names: Sequence[str], new_name: str, description: str = ""):
        rule = compose_mod.compose(self.library, names, new_name, description)
        self.note("compose", f"{new_name} = {' -> '.join(names)} "
                             f"({rule.steps()} steps, {rule.cost_bits():.0f} bits)")
        return rule

    def distill(self, source: str, new_name: str, dataset: str, epochs: int = 12,
                log=None):
        rule, report, agreement = compose_mod.distill(
            self.library, source, new_name, self.datasets[dataset],
            epochs=epochs, device=self.device, log=log)
        self.note("distill", f"{new_name} imitates {source} "
                             f"({self.library.get(source).steps()} steps -> 1), "
                             f"agreement {agreement:.3f}")
        return rule, report, agreement

    def add_task(self, name: str, start_text: str, target: str,
                 domain: str = ABSTRACT, max_depth: int = 8,
                 observed: bool = False,
                 target_domain: str = ABSTRACT) -> compose_mod.Task:
        """`observed=True` marks a specific-domain start as something the machine
        SAW rather than drew. It is the only honest test of a reader rule: a
        cell with no caption cannot be transcribed, only read."""
        start = (Content.abstract(start_text) if domain == ABSTRACT
                 else Content.specific_text(start_text))
        if observed:
            start.meta["observed"] = True
        task = compose_mod.Task(name, start, target, max_depth, target_domain)
        self.benchmark.append(task)
        self.note("task", f"{name}: {start_text!r} ({domain}) => {target!r} "
                          f"({target_domain})")
        return task

    def add_task_from_example(self, name: str, dataset: str, index: int, target: str,
                              max_depth: int = 12,
                              target_domain: str = ABSTRACT) -> compose_mod.Task:
        """A benchmark task whose starting cell is a generated experiment --
        the only way to benchmark reasoning that starts from a picture."""
        start = self.datasets[dataset].examples[index].inp
        task = compose_mod.Task(name, start, target, max_depth, target_domain)
        self.benchmark.append(task)
        self.note("task", f"{name}: {dataset}[{index}] => {target!r}")
        return task

    def report(self, trusted_only: bool = True) -> compose_mod.LibraryReport:
        rep = compose_mod.library_report(self.library, self.benchmark, trusted_only)
        self.note("report", f"J = {rep.objective:.0f} bits "
                            f"({rep.total_bits:.0f} description + "
                            f"{rep.total_steps:.0f} steps)")
        return rep

    def simplify(self, probe: str | None = None, apply_changes: bool = False,
                 protect: Sequence[str] = ()):
        actions, before, after = compose_mod.simplify(
            self.library, self.benchmark,
            self.datasets[probe] if probe else None,
            apply_changes=apply_changes, protect=protect)
        delta = (after.objective - before.objective) if after else 0.0
        self.note("simplify", f"{len(actions)} actions"
                              + (f", J {before.objective:.0f} -> {after.objective:.0f} "
                                 f"({delta:+.0f})" if after else " (dry run)"))
        return actions, before, after

    # -- proving -------------------------------------------------------------
    def prove(self, start_text: str, target: str, domain: str = ABSTRACT,
              max_depth: int = 6, trusted_only: bool = True,
              max_nodes: int = 4000, target_domain: str = ABSTRACT,
              observed: bool = False) -> proof_mod.Proof:
        start = (Content.abstract(start_text) if domain == ABSTRACT
                 else Content.specific_text(start_text))
        if observed:
            start.meta["observed"] = True
        p = proof_mod.search(self.library, start, target, max_depth=max_depth,
                             trusted_only=trusted_only, max_nodes=max_nodes,
                             target_domain=target_domain)
        self.note("prove", f"{start_text!r} => {target!r}: "
                           f"{'proved in ' + str(p.length) + ' steps' if p.found else 'not proved'}"
                           f" ({p.nodes_expanded} nodes)")
        return p

    def prove_from_tape(self, domain: str, target: str, max_depth: int = 8,
                        trusted_only: bool = True, target_domain: str = ABSTRACT,
                        max_nodes: int = 4000) -> proof_mod.Proof:
        """Prove starting from whatever the head is pointing at.

        The only way to start a proof from a SKETCH: a drawing has no textual
        form to pass as an argument, so the cell itself has to be the start.
        """
        start = self.head(domain)
        p = proof_mod.search(self.library, start, target, max_depth=max_depth,
                             trusted_only=trusted_only, max_nodes=max_nodes,
                             target_domain=target_domain)
        self.note("prove", f"from {domain} head ({start.text!r}) => {target!r}: "
                           f"{'proved in ' + str(p.length) + ' steps' if p.found else 'not proved'}"
                           f" ({p.nodes_expanded} nodes)")
        return p

    def keep_proof(self, p: proof_mod.Proof, name: str, description: str = ""):
        rule = proof_mod.proof_to_rule(self.library, p, name, description)
        self.note("keep", f"{name} = {' -> '.join(p.rule_names())}")
        return rule

    def halting_budget(self, eps: float = 0.05, lam: float | None = None,
                       delta: float = 0.05):
        """Calibrate a search budget from the proofs found so far (section 5).

        Uses the lengths of every proof in the journal's benchmark report as the
        sample of halting programs, and the least confident rule in the library
        as the per-step error sigma.

        `lam=None` fits lambda to however many proofs there are. A benchmark has
        a handful, and a fixed lambda=0.02 needs 3745, so pinning lambda here
        would mean this method never returned anything at all.
        """
        from .halting import halting_budget_for_library

        rep = compose_mod.library_report(self.library, self.benchmark)
        lengths = [p.length for p in rep.proofs.values() if p.found]
        conf = min((r.confidence() for r in self.library.trusted()), default=1.0)
        cal = halting_budget_for_library(lengths, conf, eps=eps, lam=lam, delta=delta)
        self.note("halting", cal.summary().splitlines()[0])
        return cal

    # -- persistence ---------------------------------------------------------
    def save(self, directory: str) -> str:
        path = self.library.save(directory)
        self.note("save", f"library -> {path}")
        return path

    def load(self, directory: str) -> RuleLibrary:
        self.library = RuleLibrary.load(directory, device=self.device)
        self.note("load", f"library <- {directory} ({len(self.library)} rules)")
        return self.library
