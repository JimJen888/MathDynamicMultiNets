"""
Experiments and the datasets they become.

An `Example` is one thought/tool experiment: an input cell the machine wrote
for itself, and -- once an oracle has been asked -- the output cell that is
correct for it. An `ExampleSet` is a batch of them plus the provenance needed
to reproduce it (which generator, which parameters, which oracle, which seed).

Provenance is not paperwork here. `rules.NeuralRule` is charged for its RECIPE
rather than its weights precisely because the recipe reproduces the weights, so
an ExampleSet that cannot say where it came from would break the library's
accounting.

`RuleDataset` is the torch view of an ExampleSet for one specific rule: it
applies that rule's codec, and it pre-quantizes pixels to class indices once
(see nets.class_index_to_channels) instead of re-deriving them every epoch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from .palette import rgb_to_class_index
from .tapes import Content


@dataclass
class Example:
    """One (input, correct output) pair, plus why we believe the output."""

    inp: Content
    out: Content | None = None
    label_source: str = ""          # which oracle or rule chain produced `out`
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def labeled(self) -> bool:
        return self.out is not None

    def summary(self) -> str:
        tail = self.out.text if self.out is not None else "?"
        return f"{self.inp.text!r} -> {tail!r}"


@dataclass
class ExampleSet:
    """A named batch of experiments."""

    name: str
    examples: list[Example] = field(default_factory=list)
    generator: str = ""
    generator_params: dict[str, Any] = field(default_factory=dict)
    oracle: str = ""
    seed: int = 0

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    @property
    def labeled(self) -> list[Example]:
        return [e for e in self.examples if e.labeled]

    def split(self, holdout: float = 0.2) -> tuple["ExampleSet", "ExampleSet"]:
        """Split into train/holdout WITHOUT shuffling.

        Generators emit in a deterministic order, and the holdout is the tail.
        The paper's claim is that a rule learned on small operands generalises
        to unseen and larger ones, so the honest test is a tail the generator
        was asked to make harder -- not a random subset of the same
        distribution. Generators that support it take a `harder_tail` flag.
        """
        cut = int(len(self.examples) * (1.0 - holdout))
        head = ExampleSet(f"{self.name}:train", self.examples[:cut], self.generator,
                          dict(self.generator_params), self.oracle, self.seed)
        tail = ExampleSet(f"{self.name}:holdout", self.examples[cut:], self.generator,
                          dict(self.generator_params), self.oracle, self.seed)
        return head, tail

    def subset(self, indices: Sequence[int], name: str | None = None) -> "ExampleSet":
        return ExampleSet(name or f"{self.name}:subset",
                          [self.examples[i] for i in indices], self.generator,
                          dict(self.generator_params), self.oracle, self.seed)

    def provenance(self) -> dict[str, Any]:
        return {"generator": self.generator, "generator_params": self.generator_params,
                "oracle": self.oracle, "seed": self.seed, "n": len(self.examples)}

    def summary(self) -> str:
        return (f"{self.name}: {len(self.examples)} examples "
                f"({len(self.labeled)} labeled) from {self.generator or '?'}"
                f" / {self.oracle or 'unlabeled'}")

    def as_json(self) -> str:
        return json.dumps({"name": self.name, **self.provenance()},
                          sort_keys=True, separators=(",", ":"))


def encode_pair(codec, content: Content) -> tuple[np.ndarray, np.ndarray]:
    """Codec-encode a cell and quantize both views to class indices."""
    va, vb = codec.encode(content)
    return rgb_to_class_index(va).astype(np.uint8), rgb_to_class_index(vb).astype(np.uint8)


class RuleDataset:
    """torch Dataset over an ExampleSet, encoded for one rule's codec.

    Imported lazily by train.py so the rest of the package works without torch.
    """

    def __init__(self, example_set: ExampleSet, codec) -> None:
        import torch                                   # noqa: F401  (fail fast)

        self.codec = codec
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.dropped: list[str] = []
        for ex in example_set.labeled:
            try:
                target = codec.target(ex.out)
            except ValueError as err:                   # outside vocab / too long
                self.dropped.append(f"{ex.summary()}: {err}")
                continue
            ia, ib = encode_pair(codec, ex.inp)
            self.items.append((ia, ib, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import torch

        ia, ib, y = self.items[i]
        return (torch.from_numpy(ia.astype(np.int64)),
                torch.from_numpy(ib.astype(np.int64)),
                torch.from_numpy(np.asarray(y)))


class DatasetStore(dict):
    """Named ExampleSets on the machine, so a tool call can refer to one."""

    def put(self, es: ExampleSet) -> ExampleSet:
        self[es.name] = es
        return es

    def table(self) -> str:
        if not self:
            return "(no datasets)"
        return "\n".join(es.summary() for es in self.values())
