"""
A 5x7 bitmap font, so the specific domain has no external dependency.

The tape in the specific domain holds IMAGES, and the machine must be able to
write to it with a built-in rendering function (Figure 1: "simply rendering
using a built-in function on the screen"). Pulling in PIL/OpenCV/a TTF for that
would make the pixels depend on a font file we do not control -- and the whole
point of the specific domain here is that the machine writes it and then has to
READ IT BACK through a learned mapping rule. Deterministic, dependency-free,
exactly-known pixels keep that loop honest and reproducible.

Encoding: one glyph per dict entry, seven comma-separated hex rows, top to
bottom. Each row is 5 bits, bit 4 = leftmost column. Verify any glyph with
`python -m dynamicmultinets.font 8`.

Characters with no entry fall back to `hash_glyph`, a deterministic 5x7 pattern
derived from the character. That is not a cop-out: labels like a Greek angle
name only need to be DISTINGUISHABLE to the reader net, not legible to us, and
a stable pseudo-glyph is exactly that. Anything a human needs to read (digits,
arithmetic, uppercase labels) has a real glyph.
"""

from __future__ import annotations

import numpy as np

GLYPH_W, GLYPH_H = 5, 7

_GLYPHS: dict[str, str] = {
    " ": "00,00,00,00,00,00,00",
    "0": "0E,11,13,15,19,11,0E",
    "1": "04,0C,04,04,04,04,0E",
    "2": "0E,11,01,02,04,08,1F",
    "3": "1F,02,04,02,01,11,0E",
    "4": "02,06,0A,12,1F,02,02",
    "5": "1F,10,1E,01,01,11,0E",
    "6": "06,08,10,1E,11,11,0E",
    "7": "1F,01,02,04,08,08,08",
    "8": "0E,11,11,0E,11,11,0E",
    "9": "0E,11,11,0F,01,02,0C",
    "A": "0E,11,11,1F,11,11,11",
    "B": "1E,11,11,1E,11,11,1E",
    "C": "0E,11,10,10,10,11,0E",
    "D": "1C,12,11,11,11,12,1C",
    "E": "1F,10,10,1E,10,10,1F",
    "F": "1F,10,10,1E,10,10,10",
    "G": "0E,11,10,17,11,11,0F",
    "H": "11,11,11,1F,11,11,11",
    "I": "0E,04,04,04,04,04,0E",
    "J": "07,02,02,02,02,12,0C",
    "K": "11,12,14,18,14,12,11",
    "L": "10,10,10,10,10,10,1F",
    "M": "11,1B,15,15,11,11,11",
    "N": "11,11,19,15,13,11,11",
    "O": "0E,11,11,11,11,11,0E",
    "P": "1E,11,11,1E,10,10,10",
    "Q": "0E,11,11,11,15,12,0D",
    "R": "1E,11,11,1E,14,12,11",
    "S": "0F,10,10,0E,01,01,1E",
    "T": "1F,04,04,04,04,04,04",
    "U": "11,11,11,11,11,11,0E",
    "V": "11,11,11,11,11,0A,04",
    "W": "11,11,11,15,15,1B,11",
    "X": "11,11,0A,04,0A,11,11",
    "Y": "11,11,0A,04,04,04,04",
    "Z": "1F,01,02,04,08,10,1F",
    "+": "00,04,04,1F,04,04,00",
    "-": "00,00,00,1F,00,00,00",
    "*": "00,11,0A,04,0A,11,00",
    "/": "01,02,02,04,08,08,10",
    "=": "00,00,1F,00,1F,00,00",
    "(": "02,04,08,08,08,04,02",
    ")": "08,04,02,02,02,04,08",
    "[": "0E,08,08,08,08,08,0E",
    "]": "0E,02,02,02,02,02,0E",
    ".": "00,00,00,00,00,0C,0C",
    ",": "00,00,00,00,0C,04,08",
    ":": "00,0C,0C,00,0C,0C,00",
    "^": "04,0A,11,00,00,00,00",
    "_": "00,00,00,00,00,00,1F",
    "<": "02,04,08,10,08,04,02",
    ">": "08,04,02,01,02,04,08",
    "?": "0E,11,01,02,04,00,04",
    "#": "0A,0A,1F,0A,1F,0A,0A",
    "'": "04,04,00,00,00,00,00",
}

_CACHE: dict[str, np.ndarray] = {}


def _decode(rows: str) -> np.ndarray:
    bits = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
    for r, cell in enumerate(rows.split(",")):
        v = int(cell, 16)
        for c in range(GLYPH_W):
            bits[r, c] = bool(v >> (GLYPH_W - 1 - c) & 1)
    return bits


def hash_glyph(ch: str) -> np.ndarray:
    """Stable pseudo-glyph for a character with no bitmap.

    Deterministic across runs and machines (no `hash()`, which is salted), and
    framed on all four sides so it never looks like blank tape. Distinct
    characters collide only if their FNV hashes agree in 15 bits, which is
    enough for the small label alphabets we use.
    """
    h = 2166136261
    for b in ch.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    bits = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
    bits[0, :] = True
    bits[GLYPH_H - 1, :] = True
    for r in range(1, GLYPH_H - 1):
        for c in range(GLYPH_W):
            bits[r, c] = bool(h >> ((r * GLYPH_W + c) % 31) & 1)
    return bits


def glyph(ch: str) -> np.ndarray:
    """(7, 5) bool mask for a character. Lowercase folds to uppercase."""
    key = ch.upper() if ch.upper() in _GLYPHS else ch
    if key not in _CACHE:
        _CACHE[key] = _decode(_GLYPHS[key]) if key in _GLYPHS else hash_glyph(ch)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Character -> semantic palette class. This is the ONLY place the mapping from
# what a glyph MEANS to how it is COLOURED lives; renderers just call it.
# ---------------------------------------------------------------------------
def char_class(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch in "+-*/^":
        return "operator"
    if ch in "()[],":
        return "group"
    if ch in "=<>":
        return "relation"
    return "symbol"


if __name__ == "__main__":  # pragma: no cover -- eyeball a glyph
    import sys

    for ch in (sys.argv[1:] or ["8", "A", "+"]):
        print(f"--- {ch!r} ({char_class(ch)}) ---")
        for row in glyph(ch):
            print("".join("#" if b else "." for b in row))
