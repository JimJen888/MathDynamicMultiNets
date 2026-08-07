"""
Semantic colour palette shared by every renderer in the specific domain.

Lifted, deliberately, from collisionNet.PALETTE / rgb_to_class_channels: colour
here is SEMANTIC, not appearance. The renderer paints a digit stroke, an
operator stroke, a geometry line, an obstacle, ... each in its own fixed colour,
and the network's first operation is to re-quantize RGB back into one-hot class
channels. So a rule net never has to learn "yellow means digit"; it reads the
geometry of a tape it already understands categorically.

Why keep this contract instead of feeding raw RGB:
  * The tape in the specific domain is written by OUR renderer, so the class of
    every pixel is known exactly at write time. Throwing that away and making
    the net rediscover it from RGB wastes capacity on bookkeeping.
  * It makes rules trained on one renderer (text) and rules trained on another
    (sketch) share an input contract, which is what lets a chain hop between
    them (Figure 1's "specific -> specific" arrow).

The anchors below are far enough apart in RGB that nearest-anchor matching is
exact for renderer output, and stays correct under nearest-neighbour resizing
(which introduces no new colours). Anti-aliasing is deliberately NOT used.
"""

from __future__ import annotations

import numpy as np

# Index-aligned with PALETTE. Index 0 must stay "background".
CLASS_NAMES = (
    "background",   # 0: dark surround
    "panel",        # 1: light panel / box fill -- the "structured display" frame
    "digit",        # 2: 0-9 strokes
    "operator",     # 3: + - * / ^ strokes
    "group",        # 4: ( ) [ ] , -- grouping marks
    "relation",     # 5: = < > -- relation marks
    "symbol",       # 6: letters and any other glyph (variables, angle labels)
    "highlight",    # 7: the sub-expression currently under the read/write head
    "line",         # 8: geometry lines, robot links
    "object",       # 9: obstacles / solid bodies in a sketch
    "goal",         # 10: goal marker, and the query arrow's head
)

PALETTE = np.array(
    [
        [30, 30, 30],      # 0 background  (achromatic, dark)
        [225, 225, 225],   # 1 panel       (achromatic, light)
        [250, 250, 60],    # 2 digit       (yellow)
        [250, 60, 60],     # 3 operator    (red)
        [60, 250, 250],    # 4 group       (cyan)
        [60, 250, 60],     # 5 relation    (green)
        [250, 60, 250],    # 6 symbol      (magenta)
        [250, 150, 20],    # 7 highlight   (orange)
        [60, 60, 250],     # 8 line        (blue)
        [160, 100, 40],    # 9 object      (brown)
        [120, 255, 180],   # 10 goal       (spring green)
    ],
    dtype=np.uint8,
)

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Same two-stage rule as collisionNet: achromatic pixels are one of the two grey
# surfaces (background / panel) and are split by brightness; everything else is
# matched to the nearest COLOURED anchor. Without the grey split, nearest-RGB
# folds the light panel into "digit" (both are bright).
GREY_CHROMA = 40.0    # max-min channel spread below this == achromatic
PANEL_VALUE = 128.0   # mean brightness >= this == panel, else background
_BG_CLASS, _PANEL_CLASS, _FIRST_COLOUR = 0, 1, 2


def rgb_to_class_index(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8/float RGB -> (H, W) int64 class indices. NumPy twin of
    the torch path in nets.rgb_to_class_channels; used by dataset builders and
    by the ASCII debug dump, so both agree on what a pixel means."""
    px = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    chroma = px.max(axis=1) - px.min(axis=1)
    is_grey = chroma < GREY_CHROMA
    bright = px.mean(axis=1) >= PANEL_VALUE

    pal = PALETTE[_FIRST_COLOUR:].astype(np.float32)             # (K-2, 3)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)     # (N, K-2)
    coloured = d.argmin(axis=1) + _FIRST_COLOUR

    grey = np.where(bright, _PANEL_CLASS, _BG_CLASS)
    idx = np.where(is_grey, grey, coloured)
    return idx.reshape(img.shape[0], img.shape[1]).astype(np.int64)


def colour(name: str) -> np.ndarray:
    """Palette RGB for a class name -- the renderer's only way to pick a colour."""
    return PALETTE[CLASS_INDEX[name]]
