"""
Rules: the mapping arrows of Figure 1, and the library that holds them.

A rule maps a tape cell to a tape cell. Which tapes it connects is the whole
taxonomy of the architecture:

    abstract -> abstract    calculation, algebraic identities, substitution
    abstract -> specific    rendering (a built-in write function)
    specific -> specific    a structural rewrite, learned from experiments
    specific -> abstract    reading/recognition, learned (the paper's YOLO step)

Reasoning on this machine IS the transfer of mapping rules, so the library is
the machine's real state -- the tapes are scratch space. Four kinds of rule
share one interface:

    PythonRule    prior knowledge already inside the machine (exact, cheap)
    TableRule     a memorized lookup (the 9x9 table), exact, cost grows with size
    NeuralRule    a learned rule: RuleNet + Codec, approximate, verified
    CompositeRule a chain of rules called as one -- how a multi-step derivation
                  becomes a single new rule
    EnsembleRule  a base rule plus specialists that override it on the hard
                  cases they were trained for. This is section 3's recipe for
                  pushing a <90% base rule to ~100%: run them all, let the
                  specialist win where it claims competence.

## Description length

`cost_bits` is what the machine pays to KEEP a rule, and it is what the
conciseness objective minimises. The stance taken here matters:

  * A symbolic rule costs its source text.
  * A table costs its entries.
  * A learned rule costs its RECIPE -- generator, oracle, architecture, seed --
    not its weights. That is not a fudge: the machine can regenerate the data
    and retrain, so the weights are derivable and the recipe is the actual
    description. It also gives the right pressure: a net is worth a few hundred
    bits, so replacing a six-step chain with one distilled net is a genuine
    saving, while replacing a two-symbol identity with a net is not.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .codec import Codec, codec_from_config
from .tapes import ABSTRACT, Content


def require_torch(what: str = "learned rules") -> None:
    """Fail early and actionably when torch is missing.

    Half of this package runs on numpy alone, which is deliberate -- but it
    means a missing torch does not surface until something tries to build a
    net, and by then the failure is a bare ModuleNotFoundError several tool
    calls deep in a run that has already "succeeded" at generating data.
    Raising here, at the moment a learned rule is declared, puts the error next
    to the decision that caused it and says what to do about it.
    """
    try:
        import torch  # noqa: F401
    except ImportError as err:
        import sys

        raise ImportError(
            f"PyTorch is required for {what}, and is not installed in this\n"
            f"interpreter ({sys.executable}).\n"
            "Create the project environment:\n"
            "    conda env create -f environment.yml\n"
            "    conda activate dynamicmultinet\n"
            "or install torch into the interpreter you are using:\n"
            "    pip install torch\n"
            "Everything symbolic (tapes, renderer, prior rules, proof search, "
            "the objective) works without it."
        ) from err


def default_device() -> str:
    """"cuda" when a GPU is usable, else "cpu".

    Same convention as DetourPredictor -- detect, don't hard-code. Written as a
    lazy function rather than a module constant for two reasons: importing
    torch at module scope would break the numpy-only half of this package, and
    a constant evaluated at import time cannot see a device that becomes
    available (or unavailable) later in a long session.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class RuleStats:
    """What verification has established about a rule so far."""

    n_checked: int = 0
    n_correct: int = 0
    last_checked: float = 0.0
    checked_against: str = ""
    counterexamples: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_checked if self.n_checked else 0.0

    def merge(self, n_checked: int, n_correct: int, against: str,
              counterexamples: Sequence[str]) -> None:
        if n_checked == 0:
            return          # a check that examined nothing is not a check
        self.n_checked += n_checked
        self.n_correct += n_correct
        self.last_checked = time.time()
        self.checked_against = against
        # Replace, do not fall back to the previous list: counterexamples belong
        # to the check that produced them, and a rule that has just been re-checked
        # clean must not keep reporting failures it no longer makes.
        self.counterexamples = list(counterexamples)[:32]

    def summary(self) -> str:
        if not self.n_checked:
            return "unverified"
        return (f"{self.accuracy:.3f} over {self.n_checked} checks "
                f"vs {self.checked_against or 'unknown'}")


class Rule(ABC):
    """One mapping arrow."""

    def __init__(self, name: str, domain_in: str, domain_out: str,
                 description: str = ""):
        self.name = name
        self.domain_in = domain_in
        self.domain_out = domain_out
        self.description = description
        self.stats = RuleStats()
        self.trusted = False        # set by verify.py once it clears threshold

    # -- behaviour -----------------------------------------------------------
    @abstractmethod
    def apply(self, content: Content) -> Content | None:
        """Map a cell, or return None when the rule does not apply here."""

    def applicable(self, content: Content) -> bool:
        return content.domain == self.domain_in and not content.is_blank

    # -- economics -----------------------------------------------------------
    @abstractmethod
    def cost_bits(self) -> float:
        """Bits to write this rule down. See the module docstring."""

    def steps(self) -> int:
        """Primitive rule applications this rule costs when used once."""
        return 1

    def confidence(self) -> float:
        """Laplace-smoothed accuracy: an unverified rule is not 100% trusted,
        it is unknown, and a single lucky check is not proof."""
        return (self.stats.n_correct + 1.0) / (self.stats.n_checked + 2.0)

    # -- persistence ---------------------------------------------------------
    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "name": self.name,
            "domain_in": self.domain_in,
            "domain_out": self.domain_out,
            "description": self.description,
            "trusted": self.trusted,
            "stats": {"n_checked": self.stats.n_checked,
                      "n_correct": self.stats.n_correct,
                      "checked_against": self.stats.checked_against},
        }

    def signature(self) -> str:
        return f"{self.name}: {self.domain_in}->{self.domain_out}"

    def __repr__(self) -> str:  # pragma: no cover -- logging only
        return (f"<{type(self).__name__} {self.signature()} "
                f"{self.cost_bits():.0f} bits, {self.stats.summary()}>")


# ---------------------------------------------------------------------------
# Prior knowledge and memorized tables
# ---------------------------------------------------------------------------
class PythonRule(Rule):
    """A rule the machine already had -- arithmetic evaluation, rendering, a
    known algebraic identity. Exact by construction, so it is trusted on
    creation, and cheap to keep."""

    def __init__(self, name: str, fn: Callable[[Content], Content | None],
                 domain_in: str, domain_out: str, description: str = "",
                 source: str = "", exact: bool = True):
        super().__init__(name, domain_in, domain_out, description)
        self.fn = fn
        self.source = source or description
        self.trusted = exact
        self.exact = exact

    def apply(self, content: Content) -> Content | None:
        if not self.applicable(content):
            return None
        try:
            return self.fn(content)
        except Exception:            # a precondition failure is "not applicable"
            return None

    def cost_bits(self) -> float:
        return 8.0 * max(len(self.source), 8)

    def confidence(self) -> float:
        # An exact rule is exact; charging it the Laplace prior would make a
        # four-step chain of known identities look like a coin flip.
        return 1.0 if self.exact else super().confidence()

    def to_manifest(self) -> dict[str, Any]:
        m = super().to_manifest()
        m["prior_fn"] = self.name        # resolved through prior.PRIOR_RULES on load
        return m


class TableRule(Rule):
    """A memorized lookup table (the 9x9 multiplication table of Figure 2).

    Exact within its keys and useless outside them, which is the honest model
    of memorization. Its cost grows with the number of entries, so the library
    prefers a short general rule to a big table once one is available -- that
    pressure is the point.
    """

    def __init__(self, name: str, table: dict[str, str], domain: str = ABSTRACT,
                 description: str = ""):
        super().__init__(name, domain, domain, description)
        self.table = dict(table)
        self.trusted = True

    def apply(self, content: Content) -> Content | None:
        if not self.applicable(content):
            return None
        key = content.text.replace(" ", "")
        if key not in self.table:
            return None
        out = self.table[key]
        return (Content.abstract(out) if self.domain_out == ABSTRACT
                else Content.specific_text(out))

    def cost_bits(self) -> float:
        return 8.0 * sum(len(k) + len(v) + 2 for k, v in self.table.items())

    def confidence(self) -> float:
        return 1.0          # exact within its keys, and it declines outside them

    def to_manifest(self) -> dict[str, Any]:
        m = super().to_manifest()
        m["table"] = self.table
        return m


# ---------------------------------------------------------------------------
# Learned rules
# ---------------------------------------------------------------------------
@dataclass
class Recipe:
    """How a learned rule was made -- and how it could be remade.

    This is the description the library actually stores for a NeuralRule, and
    therefore what it is charged for. Keeping it faithful is not bookkeeping:
    if the recipe is complete, the weights are a cache, and the machine can
    rebuild any learned rule from a library manifest alone.
    """

    generator: str = ""
    generator_params: dict[str, Any] = field(default_factory=dict)
    oracle: str = ""
    n_examples: int = 0
    epochs: int = 0
    seed: int = 0
    architecture: str = ""

    def as_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))


class NeuralRule(Rule):
    """A mapping rule realized by a RuleNet through a Codec.

    Untrained on creation and untrusted until `verify.verify_rule` says
    otherwise; `apply` still works before then (that is how you collect the
    counterexamples that a specialist is later trained on), but the low
    confidence propagates into every chain that uses it.
    """

    def __init__(self, name: str, codec: Codec, domain_in: str,
                 description: str = "", net: Any = None, recipe: Recipe | None = None,
                 device: str | None = None):
        require_torch(f"the learned rule {name!r}")
        super().__init__(name, domain_in, codec.out_domain, description)
        self.codec = codec
        self.recipe = recipe or Recipe()
        self.device = device or default_device()
        self._net = net

    # -- the network ---------------------------------------------------------
    @property
    def net(self):
        """Built lazily so that importing the package (and running every
        symbolic part of it) does not require torch.

        `preprocess=False`: the net is fed one-hot class channels, never raw
        RGB. Both the training path (dataset index maps) and the inference path
        (`apply` below) quantize first, so the expensive nearest-anchor match
        happens exactly once per image instead of once per epoch.
        """
        if self._net is None:
            from .nets import RuleNet

            self._net = RuleNet(num_classes=self.codec.num_classes,
                                num_slots=self.codec.num_slots,
                                preprocess=False).to(self.device)
            self.recipe.architecture = (
                f"RuleNet(classes={self.codec.num_classes},"
                f"slots={self.codec.num_slots})"
            )
        return self._net

    def apply(self, content: Content) -> Content | None:
        if not self.applicable(content):
            return None
        import torch

        from .dataset import encode_pair
        from .nets import class_index_to_channels

        ia, ib = encode_pair(self.codec, content)
        ta = class_index_to_channels(torch.from_numpy(ia.astype(np.int64))[None]).to(self.device)
        tb = class_index_to_channels(torch.from_numpy(ib.astype(np.int64))[None]).to(self.device)
        with torch.no_grad():
            idx, probs = self.net.predict(ta, tb)
        out = self.codec.decode(idx[0].cpu().numpy(), content)
        out.meta["rule"] = self.name
        out.meta["p"] = float(probs[0].max(dim=-1).values.mean())
        return out

    def rank_options(self, content: Content, k: int = 3) -> list[tuple[str, float]]:
        """Options best first, for a single-decision rule.

        DetourNet.rank exists because the planner walks the ranked candidates
        and takes the first one the collision net clears -- so "the right answer
        is in the top 3" is the operational criterion, not "argmax is exact".
        Same here: a rule that proposes an action is used through this, not
        through `apply`.
        """
        if self.codec.num_slots != 1 or not hasattr(self.codec, "classes"):
            raise TypeError(f"{self.name!r} does not make a single choice")
        import torch

        from .dataset import encode_pair
        from .nets import class_index_to_channels

        ia, ib = encode_pair(self.codec, content)
        ta = class_index_to_channels(torch.from_numpy(ia.astype(np.int64))[None]).to(self.device)
        tb = class_index_to_channels(torch.from_numpy(ib.astype(np.int64))[None]).to(self.device)
        # eval(), like `apply` gets for free through RuleNet.predict. Without it
        # the trunk's dropout is live and the ranking is a different list every
        # call -- and a net is in training mode whenever it has not just been
        # through train_rule, which includes every rule reloaded from a manifest.
        self.net.eval()
        with torch.no_grad():
            probs = self.net(ta, tb).softmax(dim=-1)[0].cpu().numpy()
        order = np.argsort(-probs)[:k]
        return [(self.codec.classes[int(i)], float(probs[int(i)])) for i in order]

    def cost_bits(self) -> float:
        # The recipe, plus a flat charge for the architecture itself. Not the
        # weights -- see the module docstring.
        return 8.0 * len(self.recipe.as_json()) + 64.0

    def to_manifest(self) -> dict[str, Any]:
        m = super().to_manifest()
        m["codec"] = self.codec.config()
        m["recipe"] = self.recipe.__dict__
        m["weights"] = f"{self.name}.pt"
        return m


class EnsembleRule(Rule):
    """Base rule + specialists, composed by prioritising the specialists.

    Section 3's construction, implemented literally: run everything on the same
    input, and take the first specialist that claims the case (its gate says
    the input is one it was trained for); otherwise take the base. This is how
    a base rule that verifies at, say, 0.9 becomes a rule that verifies at
    0.99+ without retraining the base and without a single monolithic model.

    A specialist's `gate` is a predicate over the input cell. In practice it is
    produced by hard-case mining: the cases the base got wrong are clustered,
    and the gate is "looks like one of those".
    """

    def __init__(self, name: str, base: Rule,
                 specialists: Sequence[tuple[Callable[[Content], bool], Rule]] = (),
                 description: str = ""):
        super().__init__(name, base.domain_in, base.domain_out, description)
        self.base = base
        self.specialists: list[tuple[Callable[[Content], bool], Rule]] = list(specialists)

    def add_specialist(self, gate: Callable[[Content], bool], rule: Rule) -> None:
        self.specialists.append((gate, rule))

    def apply(self, content: Content) -> Content | None:
        for gate, rule in self.specialists:          # specialists win, in order
            try:
                claims = gate(content)
            except Exception:
                claims = False
            if claims:
                out = rule.apply(content)
                if out is not None:
                    out.meta["rule"] = f"{self.name}/{rule.name}"
                    return out
        return self.base.apply(content)

    def cost_bits(self) -> float:
        return self.base.cost_bits() + sum(r.cost_bits() + 32 for _, r in self.specialists)

    def to_manifest(self) -> dict[str, Any]:
        m = super().to_manifest()
        m["base"] = self.base.name
        m["specialists"] = [r.name for _, r in self.specialists]
        return m


class CompositeRule(Rule):
    """A chain of rules invoked as a single rule.

    This is how a derivation becomes knowledge: once a chain is found and
    verified, the machine keeps it under one name and stops re-searching for
    it. Its confidence is the product of its members' (errors compound along a
    chain -- the 0.99999^1000 argument in the conclusions), and its `steps` is
    the true number of primitive applications, so the conciseness objective can
    see that a distilled replacement would be shorter to run.
    """

    def __init__(self, name: str, members: Sequence[Rule], description: str = ""):
        if not members:
            raise ValueError("a composite rule needs at least one member")
        super().__init__(name, members[0].domain_in, members[-1].domain_out, description)
        self.members = list(members)

    def apply(self, content: Content) -> Content | None:
        cur: Content | None = content
        for rule in self.members:
            if cur is None:
                return None
            cur = rule.apply(cur)
        if cur is not None:
            cur.meta["rule"] = self.name
        return cur

    def cost_bits(self) -> float:
        # The chain itself is just a list of names; the members are already
        # charged for individually in the library.
        return 8.0 * sum(len(r.name) + 1 for r in self.members) + 16.0

    def steps(self) -> int:
        return sum(r.steps() for r in self.members)

    def confidence(self) -> float:
        p = 1.0
        for r in self.members:
            p *= r.confidence()
        return p

    def to_manifest(self) -> dict[str, Any]:
        m = super().to_manifest()
        m["members"] = [r.name for r in self.members]
        return m


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
class RuleLibrary:
    """The machine's knowledge base: every rule it knows, by name."""

    def __init__(self) -> None:
        self.rules: dict[str, Rule] = {}

    # -- container -----------------------------------------------------------
    def add(self, rule: Rule, replace: bool = False) -> Rule:
        if rule.name in self.rules and not replace:
            raise ValueError(f"rule {rule.name!r} already exists; pass replace=True")
        self.rules[rule.name] = rule
        return rule

    def remove(self, name: str) -> None:
        self.rules.pop(name, None)

    def get(self, name: str) -> Rule:
        if name not in self.rules:
            raise KeyError(f"no rule named {name!r}; known: {sorted(self.rules)}")
        return self.rules[name]

    def __contains__(self, name: object) -> bool:
        return name in self.rules

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterable[Rule]:
        return iter(self.rules.values())

    # -- queries -------------------------------------------------------------
    def from_domain(self, domain: str) -> list[Rule]:
        return [r for r in self.rules.values() if r.domain_in == domain]

    def trusted(self) -> list[Rule]:
        return [r for r in self.rules.values() if r.trusted]

    def total_bits(self) -> float:
        return sum(r.cost_bits() for r in self.rules.values())

    def table(self) -> str:
        rows = [f"{'name':<26}{'map':<22}{'bits':>8}  {'conf':>5}  status"]
        for r in sorted(self.rules.values(), key=lambda r: r.name):
            mapping = f"{r.domain_in}->{r.domain_out}"
            flag = "trusted" if r.trusted else r.stats.summary()
            rows.append(f"{r.name:<26}{mapping:<22}{r.cost_bits():>8.0f}  "
                        f"{r.confidence():>5.2f}  {flag}")
        return "\n".join(rows)

    # -- persistence ---------------------------------------------------------
    def save(self, directory: str) -> str:
        """Manifest + weights. The manifest alone is enough to rebuild every
        rule from its recipe; the weights are a cache that saves retraining."""
        import os

        os.makedirs(directory, exist_ok=True)
        manifest = {"rules": [r.to_manifest() for r in self.rules.values()]}
        for rule in self.rules.values():
            if isinstance(rule, NeuralRule) and rule._net is not None:
                import torch

                torch.save(rule._net.state_dict(), os.path.join(directory, f"{rule.name}.pt"))
        path = os.path.join(directory, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        return path

    @staticmethod
    def load(directory: str, device: str | None = None) -> "RuleLibrary":
        import os

        from .prior import PRIOR_RULES, install_prior_rules

        lib = RuleLibrary()
        install_prior_rules(lib)                 # built-ins are code, not data
        with open(os.path.join(directory, "manifest.json")) as f:
            manifest = json.load(f)

        deferred = []
        for m in manifest["rules"]:
            kind = m["kind"]
            if kind == "PythonRule":
                if m["name"] not in lib and m["name"] in PRIOR_RULES:
                    lib.add(PRIOR_RULES[m["name"]]())
                continue
            if kind == "TableRule":
                rule: Rule = TableRule(m["name"], m["table"], m["domain_in"],
                                       m["description"])
            elif kind == "NeuralRule":
                rule = NeuralRule(m["name"], codec_from_config(m["codec"]),
                                  m["domain_in"], m["description"],
                                  recipe=Recipe(**m["recipe"]), device=device)
                wpath = os.path.join(directory, m["weights"])
                if os.path.exists(wpath):
                    import torch

                    rule.net.load_state_dict(torch.load(wpath, map_location=device))
            else:
                deferred.append(m)               # composites/ensembles need members
                continue
            rule.trusted = m.get("trusted", False)
            rule.stats.n_checked = m["stats"]["n_checked"]
            rule.stats.n_correct = m["stats"]["n_correct"]
            rule.stats.checked_against = m["stats"]["checked_against"]
            lib.add(rule, replace=True)

        for m in deferred:
            if m["kind"] == "CompositeRule":
                members = [lib.get(n) for n in m["members"]]
                lib.add(CompositeRule(m["name"], members, m["description"]), replace=True)
            elif m["kind"] == "EnsembleRule":
                ens = EnsembleRule(m["name"], lib.get(m["base"]), description=m["description"])
                for n in m["specialists"]:
                    # Gates are behavioural, not serialisable; a reloaded
                    # ensemble defers to a specialist whenever it is confident.
                    spec = lib.get(n)
                    ens.add_specialist(_confidence_gate(spec), spec)
                lib.add(ens, replace=True)
        return lib


def _confidence_gate(rule: Rule, threshold: float = 0.9) -> Callable[[Content], bool]:
    """Default gate for a reloaded specialist: claim the case when the
    specialist's own output is confident. Weaker than the mined gate it
    replaces, but it never silently claims a case it is unsure about."""

    def gate(content: Content) -> bool:
        out = rule.apply(content)
        return out is not None and float(out.meta.get("p", 0.0)) >= threshold

    return gate
