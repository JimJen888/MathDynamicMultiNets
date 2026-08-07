"""
The two tapes and their read/write heads.

This is the structural claim of the paper, and it is the one thing in this
package that is NOT an ordinary ML utility: a Turing machine reads back exactly
the alphabet it wrote, whereas here the controller writes symbols on one tape
and images on the other, and the only way across is a mapping rule (learned or
built in). Keeping the two tapes as separate objects with separate alphabets is
what forces every cross-domain step in this codebase to go through a Rule --
you cannot "just read the text" off the specific tape, because a cell there
holds pixels.

    AbstractTape   alphabet: symbol strings ("12 * 30", "A1 = B1")
                   supports: exact arithmetic, rigid definitions, formal logic
    SpecificTape   alphabet: (64, 384, 3) two-view images
                   supports: rendering, perception, structure-as-layout

`Content.text` is carried alongside a specific cell as PROVENANCE (what the
renderer was asked to draw). It is there for logging, for supervising a reader
rule, and for cheap assertions in tests. Rules that operate in the specific
domain must consume `Content.image`; `Content.require_image()` is the guard
that keeps that honest, and NeuralRule always goes through it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

ABSTRACT = "abstract"
SPECIFIC = "specific"


@dataclass
class Content:
    """One tape cell's value."""

    domain: str
    text: str = ""
    image: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- constructors --------------------------------------------------------
    @staticmethod
    def abstract(text: str, **meta: Any) -> "Content":
        return Content(ABSTRACT, text=text.strip(), meta=dict(meta))

    @staticmethod
    def specific_text(text: str, highlight: str | None = None, **meta: Any) -> "Content":
        from .render import render_text

        return Content(SPECIFIC, text=text.strip(),
                       image=render_text(text, highlight=highlight), meta=dict(meta))

    @staticmethod
    def specific_sketch(scene: dict, **meta: Any) -> "Content":
        from .render import render_scene

        m = dict(meta)
        m["scene"] = scene
        return Content(SPECIFIC, text=meta.get("caption", "sketch"),
                       image=render_scene(scene), meta=m)

    # -- access --------------------------------------------------------------
    def require_image(self) -> np.ndarray:
        if self.image is None:
            raise ValueError(
                f"cell in domain {self.domain!r} has no image; a specific-domain "
                "rule must be handed rendered pixels, not symbols -- render it "
                "first with the abstract->specific mapping rule"
            )
        return self.image

    @property
    def is_blank(self) -> bool:
        return not self.text and self.image is None

    def views(self) -> tuple[np.ndarray, np.ndarray]:
        from .render import split_views

        return split_views(self.require_image())

    def summary(self) -> str:
        if self.domain == ABSTRACT:
            return f"[abstract] {self.text!r}"
        shape = "-" if self.image is None else "x".join(map(str, self.image.shape))
        return f"[specific] image({shape}) drawn from {self.text!r}"

    def __repr__(self) -> str:  # pragma: no cover -- logging only
        return f"Content({self.summary()})"


BLANK_ABSTRACT = Content(ABSTRACT)
BLANK_SPECIFIC = Content(SPECIFIC)


@dataclass
class TapeEvent:
    """One head operation. The journal of these IS the machine's trace, and it
    is what the controller reads back when it asks what has happened so far."""

    t: float
    tape: str
    op: str
    position: int
    detail: str


class Tape:
    """A one-sided infinite tape with a single read/write head.

    Cells are stored sparsely and default to blank, so 'infinite' costs nothing
    and the head can move right forever.
    """

    def __init__(self, domain: str, name: str | None = None):
        self.domain = domain
        self.name = name or f"{domain}_tape"
        self.cells: dict[int, Content] = {}
        self.position = 0
        self.journal: list[TapeEvent] = []

    # -- head ----------------------------------------------------------------
    def read(self, at: int | None = None) -> Content:
        pos = self.position if at is None else at
        cell = self.cells.get(pos)
        if cell is None:
            cell = Content(self.domain)
        self._log("read", pos, cell.summary())
        return cell

    def write(self, content: Content, at: int | None = None) -> None:
        if content.domain != self.domain:
            raise ValueError(
                f"cannot write {content.domain!r} content to the {self.domain!r} "
                "tape -- cross-domain moves must go through a mapping rule"
            )
        pos = self.position if at is None else at
        self.cells[pos] = content
        self._log("write", pos, content.summary())

    def move(self, delta: int = 1) -> int:
        self.position = max(0, self.position + delta)
        self._log("move", self.position, f"delta={delta:+d}")
        return self.position

    def append(self, content: Content) -> int:
        """Write at the first blank cell at or after the head, then park the
        head there. Most rule applications want this rather than overwriting."""
        pos = self.position
        while pos in self.cells and not self.cells[pos].is_blank:
            pos += 1
        self.position = pos
        self.write(content, pos)
        return pos

    # -- inspection ----------------------------------------------------------
    def occupied(self) -> list[int]:
        return sorted(p for p, c in self.cells.items() if not c.is_blank)

    def dump(self, limit: int = 12) -> str:
        rows = []
        for p in self.occupied()[-limit:]:
            head = ">" if p == self.position else " "
            rows.append(f"{head}{p:3d}: {self.cells[p].summary()}")
        return "\n".join(rows) or "(empty)"

    def __iter__(self) -> Iterator[tuple[int, Content]]:
        for p in self.occupied():
            yield p, self.cells[p]

    def _log(self, op: str, pos: int, detail: str) -> None:
        self.journal.append(TapeEvent(time.time(), self.name, op, pos, detail))


class AbstractTape(Tape):
    """Digits and symbols. Calculation, rigid concept definition, formal logic."""

    def __init__(self) -> None:
        super().__init__(ABSTRACT, "abstract_tape")


class SpecificTape(Tape):
    """Images and texts as rendered pixels; affected by the outside world when
    a cell is written from a camera rather than from the built-in renderer."""

    def __init__(self) -> None:
        super().__init__(SPECIFIC, "specific_tape")

    def observe(self, image: np.ndarray, caption: str = "observation") -> int:
        """Write a cell straight from the outside world (a camera frame).

        The distinction from `write` matters: cells written by the renderer are
        things the machine imagined (thought experiments); cells written here
        are things it saw. Verification against 'real-world facts' only counts
        when it bottoms out in one of these.
        """
        return self.append(Content(SPECIFIC, text=caption, image=image,
                                   meta={"observed": True}))
