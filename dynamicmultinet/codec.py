"""
Codecs: how a learned rule sees a tape cell, and what its logits mean.

A RuleNet is just a tensor function. What makes it a *mapping rule between two
domains* is the pair of conversions on either side of it:

    Content --encode--> (view_a, view_b) --RuleNet--> indices --decode--> Content

The codec owns both, plus `target()`, which turns a supervised output Content
back into label indices. Keeping all three in one object is what lets
`train_rule`, `verify_rule` and `apply_rule` share a single definition of what
the rule is supposed to do -- a mismatch between the training target and the
inference decode is the classic silent bug in this kind of system, and here it
is impossible by construction.

Two codecs cover everything in the paper:

    TextSlotCodec   transduction: an expression in, an expression out. Used for
                    the distributive rewrite (specific -> specific) and for the
                    reader that maps a rendered image back to symbols
                    (specific -> abstract; the paper's YOLO step).
    ChoiceCodec     one decision out of K. Used for the sketch-to-direction
                    rule of Appendix A, and for yes/no verifiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from .tapes import ABSTRACT, SPECIFIC, Content

PAD = "_"
DEFAULT_VOCAB = PAD + "0123456789+-*/=().^ "


class Codec(ABC):
    """Input encoding + output decoding for one learned rule."""

    out_domain: str = ABSTRACT
    num_classes: int = 0
    num_slots: int = 1

    @abstractmethod
    def encode(self, content: Content) -> tuple[np.ndarray, np.ndarray]:
        """Tape cell -> the two views the shared encoder consumes."""

    @abstractmethod
    def decode(self, idx: np.ndarray, source: Content | None = None) -> Content:
        """Predicted indices -> the tape cell to write. `idx` is () or (slots,).

        `source` is the input cell. Most codecs ignore it -- a rewrite rule's
        output is fully determined by the predicted symbols. A rule that EDITS
        its input needs it: deciding "rotate the line clockwise" only becomes a
        new tape cell once you know which drawing is being rotated.
        """

    @abstractmethod
    def target(self, content: Content) -> np.ndarray:
        """Supervised output cell -> label indices, same shape as `decode` takes."""

    def config(self) -> dict[str, Any]:
        """Everything needed to rebuild this codec from a manifest."""
        return {"type": type(self).__name__}


class TextSlotCodec(Codec):
    """Expression in (as pixels), expression out (as characters).

    Inputs are always read as an IMAGE. If the incoming cell is abstract it is
    rendered first -- that is the abstract->specific mapping rule applied
    implicitly, and it is exactly what a person does when they write an
    expression down before rewriting it.

    `out_domain` decides what the rule produces: SPECIFIC re-renders the
    predicted string (a specific->specific rewrite, Figure 2's distributive
    step), ABSTRACT writes the bare symbols (the reader/YOLO step).
    """

    def __init__(self, num_slots: int = 24, vocab: str = DEFAULT_VOCAB,
                 out_domain: str = SPECIFIC):
        self.vocab = vocab
        self.num_slots = num_slots
        self.num_classes = len(vocab)
        self.out_domain = out_domain
        self._index = {ch: i for i, ch in enumerate(vocab)}

    def encode(self, content: Content) -> tuple[np.ndarray, np.ndarray]:
        if content.image is None:
            content = Content.specific_text(content.text)
        return content.views()

    def decode(self, idx: np.ndarray, source: Content | None = None) -> Content:
        text = "".join(self.vocab[int(i)] for i in np.atleast_1d(idx))
        text = text.replace(PAD, "").strip()
        return (Content.specific_text(text) if self.out_domain == SPECIFIC
                else Content.abstract(text))

    def target(self, content: Content) -> np.ndarray:
        text = content.text
        if len(text) > self.num_slots:
            raise ValueError(
                f"{text!r} needs {len(text)} slots but the rule has "
                f"{self.num_slots}; declare the rule with more slots"
            )
        unknown = sorted(set(text) - set(self.vocab))
        if unknown:
            raise ValueError(f"characters {unknown} are outside this rule's vocabulary")
        padded = text + PAD * (self.num_slots - len(text))
        return np.array([self._index[ch] for ch in padded], dtype=np.int64)

    def config(self) -> dict[str, Any]:
        return {"type": "TextSlotCodec", "num_slots": self.num_slots,
                "vocab": self.vocab, "out_domain": self.out_domain}


class ChoiceCodec(Codec):
    """One decision out of K named options -- DetourNet's original head.

    The chosen option's NAME is written to the abstract tape, because a
    decision ("go 0.40 in -y", "collision", "distributive applies") is a symbol,
    not a picture. That is the specific->abstract arrow of Figure 1, and in the
    robotics appendix it is exactly step 4 of the VSRA loop.
    """

    num_slots = 1

    def __init__(self, classes: Sequence[str], out_domain: str = ABSTRACT):
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.out_domain = out_domain
        self._index = {c: i for i, c in enumerate(self.classes)}

    def encode(self, content: Content) -> tuple[np.ndarray, np.ndarray]:
        if content.image is None:
            content = Content.specific_text(content.text)
        return content.views()

    def decode(self, idx: np.ndarray, source: Content | None = None) -> Content:
        name = self.classes[int(np.atleast_1d(idx)[0])]
        return (Content.abstract(name) if self.out_domain == ABSTRACT
                else Content.specific_text(name))

    def target(self, content: Content) -> np.ndarray:
        label = content.meta.get("label", content.text)
        if label not in self._index:
            raise ValueError(f"{label!r} is not one of this rule's classes {self.classes}")
        return np.array(self._index[label], dtype=np.int64)

    def config(self) -> dict[str, Any]:
        return {"type": "ChoiceCodec", "classes": self.classes,
                "out_domain": self.out_domain}


class SceneActionCodec(Codec):
    """Look at a sketch, decide an action, and write the RESULTING sketch.

    A specific -> specific rule in the sense of Figure A1: the perception is
    learned, the update is arithmetic on scene parameters, and the output cell
    is a new drawing rather than a label. Splitting it any other way does not
    work -- a rule that emits the word "rotate_cw" has left the specific domain
    and cannot be chained with another look at the picture.

    The `done` action returns the input unchanged and flags it, which is what
    makes a construction loop terminate: proof search stops expanding a state
    that maps to itself.
    """

    num_slots = 1
    out_domain = SPECIFIC

    def __init__(self, actions: Sequence[str] = ("move_up", "rotate_cw",
                                                 "rotate_ccw", "done"),
                 step: float = 0.12, angle_step: float = 0.15):
        self.actions = list(actions)
        self.num_classes = len(self.actions)
        self.step = step
        self.angle_step = angle_step
        self._index = {a: i for i, a in enumerate(self.actions)}

    def encode(self, content: Content) -> tuple[np.ndarray, np.ndarray]:
        return content.views()

    def decode(self, idx: np.ndarray, source: Content | None = None) -> Content:
        action = self.actions[int(np.atleast_1d(idx)[0])]
        if source is None or "scene" not in source.meta:
            return Content.abstract(action)          # nothing to edit; name the action
        scene = dict(source.meta["scene"])
        if action == "done":
            out = Content.specific_sketch(scene, caption=source.text)
            out.meta["done"] = True
            out.meta["action_taken"] = action
            # What this rule DECIDED, for verification. The cell's text is a
            # caption for a drawing and cannot be compared with an oracle's
            # label; the decision can.
            out.meta["decision"] = action
            return out
        if action == "move_up":
            # Offsets are negative below the apex, so "up" is towards zero.
            scene["line_offset"] = float(scene.get("line_offset", 0.0)) + self.step
        elif action == "rotate_cw":
            scene["line_angle"] = float(scene.get("line_angle", 0.0)) - self.angle_step
        elif action == "rotate_ccw":
            scene["line_angle"] = float(scene.get("line_angle", 0.0)) + self.angle_step
        out = Content.specific_sketch(scene, caption=f"{source.text}|{action}")
        # Deliberately NOT "action", which is the key prior.sketch_action reads:
        # the update has already been applied here, and leaving that key set
        # would let the built-in rule fire again and repeat the move.
        out.meta["action_taken"] = action
        # What this rule actually DECIDED, for verification. The cell's text is
        # a caption for a drawing and cannot be compared with an oracle's
        # label; the decision can.
        out.meta["decision"] = action
        return out

    def target(self, content: Content) -> np.ndarray:
        label = content.meta.get("label", content.text)
        if label not in self._index:
            raise ValueError(f"{label!r} is not one of {self.actions}")
        return np.array(self._index[label], dtype=np.int64)

    def config(self) -> dict[str, Any]:
        return {"type": "SceneActionCodec", "actions": self.actions,
                "step": self.step, "angle_step": self.angle_step}


def codec_from_config(cfg: dict[str, Any]) -> Codec:
    """Rebuild a codec saved in a library manifest."""
    kind = cfg["type"]
    if kind == "TextSlotCodec":
        return TextSlotCodec(num_slots=cfg["num_slots"], vocab=cfg["vocab"],
                             out_domain=cfg["out_domain"])
    if kind == "ChoiceCodec":
        return ChoiceCodec(cfg["classes"], out_domain=cfg["out_domain"])
    if kind == "SceneActionCodec":
        return SceneActionCodec(cfg["actions"], cfg["step"], cfg["angle_step"])
    raise ValueError(f"unknown codec type {kind!r}")
