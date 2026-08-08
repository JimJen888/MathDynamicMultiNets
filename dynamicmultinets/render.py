"""
The write head of the tape in the specific domain: content -> pixels.

Everything the machine puts on the specific tape is drawn here, in exactly two
side-by-side views, because that is the input contract every rule net inherits
from DetourNet/MotionCollisionNet: a shared encoder sees view A and view B, and
the fusion adds `fa - fb`. In the robot nets the two views were two fixed
cameras and the difference was a disparity cue. Here they are two *readings of
the same content*:

    view A  inline    -- the expression written out on one line, large glyphs
    view B  structured -- the same expression laid out as framed boxes, the
                          "structured display" of Figure 2/3, small glyphs

`fa - fb` therefore carries "does the flat reading agree with the structured
reading" -- which is precisely the cue a transformation like the distributive
rule turns on (it is a statement about structure, not about the digit string).
For sketches the two views go back to their original meaning: top view and side
view of the same 3D scene.

Deliberately no anti-aliasing anywhere: every pixel is an exact palette anchor,
so `rgb_to_class_channels` is lossless and a rendered tape cell can be compared
byte-for-byte across runs.
"""

from __future__ import annotations

import numpy as np

from .font import GLYPH_H, GLYPH_W, char_class, glyph
from .palette import colour

VIEW_H, VIEW_W = 64, 192          # one view
SPLIT_W = VIEW_W                  # the two views are concatenated at this column
FULL_W = 2 * VIEW_W

CELL_W, CELL_H = GLYPH_W + 1, GLYPH_H + 1     # one character cell, scale 1


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def blank_view(fill: str = "background") -> np.ndarray:
    v = np.empty((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    v[:, :] = colour(fill)
    return v


def blit_char(view: np.ndarray, ch: str, x: int, y: int, scale: int = 1,
              cls: str | None = None) -> None:
    """Draw one glyph with its top-left at (x, y). Clipped at the view edge."""
    rgb = colour(cls or char_class(ch))
    mask = glyph(ch)
    if scale > 1:
        mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    h, w = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(VIEW_W, x + w), min(VIEW_H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    view[y0:y1, x0:x1][sub] = rgb


def blit_text(view: np.ndarray, text: str, x: int, y: int, scale: int = 1,
              cls: str | None = None) -> int:
    """Draw a run of characters left to right; returns the x past the last one."""
    for ch in text:
        blit_char(view, ch, x, y, scale, cls)
        x += CELL_W * scale
    return x


def draw_frame(view: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               cls: str = "panel") -> None:
    """One-pixel rectangle outline -- the box around a term in Figure 2."""
    rgb = colour(cls)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(VIEW_W - 1, x1), min(VIEW_H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return
    view[y0, x0:x1 + 1] = rgb
    view[y1, x0:x1 + 1] = rgb
    view[y0:y1 + 1, x0] = rgb
    view[y0:y1 + 1, x1] = rgb


def draw_line(view: np.ndarray, p0, p1, cls: str = "line", width: int = 1) -> None:
    """Integer Bresenham-ish line; no anti-aliasing, see module docstring."""
    rgb = colour(cls)
    (x0, y0), (x1, y1) = (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(n + 1):
        x = int(round(x0 + (x1 - x0) * i / n))
        y = int(round(y0 + (y1 - y0) * i / n))
        for dy in range(-(width // 2), width // 2 + 1):
            for dx in range(-(width // 2), width // 2 + 1):
                if 0 <= y + dy < VIEW_H and 0 <= x + dx < VIEW_W:
                    view[y + dy, x + dx] = rgb


def draw_disc(view: np.ndarray, centre, radius: float, cls: str = "object") -> None:
    rgb = colour(cls)
    cx, cy = centre
    ys, xs = np.ogrid[:VIEW_H, :VIEW_W]
    view[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = rgb


def draw_arrow(view: np.ndarray, p0, p1, cls: str = "highlight") -> None:
    """Query arrow: shaft plus a small solid head, as in the robot renders."""
    draw_line(view, p0, p1, cls, width=1)
    draw_disc(view, p1, 2.0, cls)


# ---------------------------------------------------------------------------
# View A: inline text
# ---------------------------------------------------------------------------
def _inline_view(text: str, scale: int = 2, highlight: str | None = None) -> np.ndarray:
    """The expression on one line (wrapping when it does not fit).

    `highlight` is a substring drawn in the 'highlight' class instead of its
    normal token colour -- this is how the read head marks WHICH sub-expression
    it is pointing at, so a rule can be applied to part of a tape cell.
    """
    view = blank_view()
    cols = VIEW_W // (CELL_W * scale)
    rows = VIEW_H // (CELL_H * scale)
    hi0 = text.find(highlight) if highlight else -1
    hi1 = hi0 + len(highlight) if hi0 >= 0 and highlight else -1

    for i, ch in enumerate(text[: cols * rows]):
        r, c = divmod(i, cols)
        cls = "highlight" if hi0 <= i < hi1 else None
        blit_char(view, ch, c * CELL_W * scale, r * CELL_H * scale, scale, cls)
    return view


# ---------------------------------------------------------------------------
# View B: structured layout
# ---------------------------------------------------------------------------
def split_top_level(text: str, seps: str = "+-") -> list[str]:
    """Split on top-level separators, respecting brackets. Separators are kept
    as their own single-character entries so the layout can draw them between
    boxes."""
    parts, buf, depth = [], "", 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and ch in seps and buf.strip():
            parts.append(buf.strip())
            parts.append(ch)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _term_lines(term: str) -> list[str]:
    """A product is stacked vertically ('10' over '* 30'), the way the paper
    draws multiplication in the specific domain; anything else stays on one
    line. This is the whole reason view B is informative: the STRUCTURE of the
    term is expressed as layout, not as characters."""
    inner = term.strip().strip("()")
    if "*" in inner:
        head, *rest = [p.strip() for p in inner.split("*")]
        return [head] + [f"*{p}" for p in rest]
    return [inner]


def _structured_view(text: str, scale: int = 1) -> np.ndarray:
    view = blank_view()
    # '=' separates too: an equation should read as two boxed sides, which is
    # what makes a rewrite rule's before/after visible as layout.
    parts = split_top_level(text, "+-=")
    x, y, row_h = 2, 2, 0

    for part in parts:
        if part in "+-=":
            blit_char(view, part, x + 1, y + 6 * scale, scale)
            x += CELL_W * scale + 3
            continue
        lines = _term_lines(part)
        w = max(len(s) for s in lines) * CELL_W * scale + 4
        h = len(lines) * CELL_H * scale + 4
        if x + w >= VIEW_W:                      # wrap to the next band
            x, y = 2, y + row_h + 3
            row_h = 0
        draw_frame(view, x, y, x + w, y + h)
        for i, s in enumerate(lines):
            blit_text(view, s, x + 2, y + 2 + i * CELL_H * scale, scale)
        x += w + 3
        row_h = max(row_h, h)
    return view


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------
def render_text(text: str, highlight: str | None = None) -> np.ndarray:
    """Text -> the (64, 384, 3) two-view image written to the specific tape."""
    out = np.empty((VIEW_H, FULL_W, 3), dtype=np.uint8)
    out[:, :SPLIT_W] = _inline_view(text, highlight=highlight)
    out[:, SPLIT_W:] = _structured_view(text)
    return out


def render_sketch(scene: dict) -> np.ndarray:
    """Sketch -> two-view image, the Appendix-A "simplified 3D simulative space".

    scene = {
        "base":      (x, y, z),          robot base
        "eef":       (x, y, z),          end effector
        "goal":      (x, y, z),
        "obstacles": [(x, y, z, r), ...] collision spheres
    }
    World coordinates are in metres in [-1, 1]; view A projects (x, y) (top
    view) and view B projects (x, z) (side view), which is the pair the VSRA
    pseudocode asks for ("images taken from the top and the right side").
    """
    def to_px(a, b):
        return (int((a + 1.0) * 0.5 * (VIEW_W - 1)),
                int((1.0 - (b + 1.0) * 0.5) * (VIEW_H - 1)))

    out = np.empty((VIEW_H, FULL_W, 3), dtype=np.uint8)
    for k, (i, j) in enumerate(((0, 1), (0, 2))):      # (x,y) top, (x,z) side
        view = blank_view()
        view[VIEW_H - 4:, :] = colour("panel")          # floor band
        base, eef, goal = scene["base"], scene["eef"], scene["goal"]
        for ob in scene.get("obstacles", ()):
            draw_disc(view, to_px(ob[i], ob[j]), max(2.0, ob[3] * VIEW_H * 0.5))
        draw_line(view, to_px(base[i], base[j]), to_px(eef[i], eef[j]), "line", width=2)
        draw_disc(view, to_px(goal[i], goal[j]), 3.0, "goal")
        draw_arrow(view, to_px(eef[i], eef[j]), to_px(goal[i], goal[j]))
        out[:, k * SPLIT_W:(k + 1) * SPLIT_W] = view
    return out


def render_geometry(scene: dict) -> np.ndarray:
    """Geometry sketch -> two-view image (the workflow of Figure 3).

    scene = {
        "kind":        "geometry",
        "tri":         [(x, y) apex, (x, y) left base, (x, y) right base],
        "line_offset": how far above the apex the auxiliary line sits (world y),
        "line_angle":  its angle in radians,
        "annotated":   draw the angle markers,
    }
    The proof needs the auxiliary line to pass THROUGH the apex and be PARALLEL
    to the opposite edge; both conditions are visible in the drawing and in
    nothing else, which is what makes deciding the next action a perception
    problem rather than a lookup.

    view A  the figure
    view B  the same figure with angle markers, i.e. the annotation mapping of
            Figure 3 applied. Keeping the annotation in its own view means the
            reader net can compare marked and unmarked structure via fa - fb.
    """
    def to_px(x, y):
        return (int((x + 1.0) * 0.5 * (VIEW_W - 1)),
                int((1.0 - (y + 1.0) * 0.5) * (VIEW_H - 1)))

    apex, left, right = [tuple(p) for p in scene["tri"]]
    offset = float(scene.get("line_offset", 0.0))
    angle = float(scene.get("line_angle", 0.0))
    out = np.empty((VIEW_H, FULL_W, 3), dtype=np.uint8)

    for k in (0, 1):
        view = blank_view()
        for p, q in ((apex, left), (left, right), (right, apex)):
            draw_line(view, to_px(*p), to_px(*q), "line", width=1)

        cx, cy = apex[0], apex[1] + offset          # the auxiliary line's anchor
        dx, dy = np.cos(angle), np.sin(angle)
        draw_line(view, to_px(cx - 1.6 * dx, cy - 1.6 * dy),
                  to_px(cx + 1.6 * dx, cy + 1.6 * dy), "goal", width=1)

        if k == 1 and scene.get("annotated", True):
            ax, ay = to_px(cx, cy)
            draw_disc(view, (ax - 7, ay + 4), 2.0, "highlight")     # alpha1
            draw_disc(view, (ax + 7, ay + 4), 2.0, "highlight")     # alpha2
            draw_disc(view, (ax, ay + 8), 2.0, "goal")              # alpha3
            for base in (left, right):                              # beta1, beta2
                bx, by = to_px(*base)
                draw_disc(view, (bx, by - 5), 2.0, "highlight")
            blit_text(view, "A1", ax - 20, ay + 2, 1, "symbol")
            blit_text(view, "A2", ax + 12, ay + 2, 1, "symbol")
        out[:, k * SPLIT_W:(k + 1) * SPLIT_W] = view
    return out


def render_scene(scene: dict) -> np.ndarray:
    """Dispatch a scene dict to the renderer for its kind."""
    if scene.get("kind") == "geometry":
        return render_geometry(scene)
    return render_sketch(scene)


def split_views(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, 2W, 3) -> the two (H, W, 3) views the shared encoder consumes."""
    return img[:, :SPLIT_W], img[:, SPLIT_W:]


def to_ascii(img: np.ndarray, cols: int = 96) -> str:
    """Debug dump of a tape cell: one character per class, so you can read what
    the machine actually wrote without opening an image viewer."""
    from .palette import rgb_to_class_index

    marks = " .0*()#!|@G"          # index-aligned with palette.CLASS_NAMES
    idx = rgb_to_class_index(img)
    step = max(1, img.shape[1] // cols)
    rows = []
    for r in range(0, idx.shape[0], step * 2):
        rows.append("".join(marks[idx[r, c]] for c in range(0, idx.shape[1], step)))
    return "\n".join(rows)


def layout_text(text: str) -> str:
    """What the two views DRAW, written as ordinary readable text.

    The specific tape holds pixels, and the information that matters about them
    is not the strokes but the arrangement: which terms are boxed separately,
    what separator sits between the boxes, and which factors are stacked inside
    one box. This reports exactly that, with every glyph decoded to the
    characters it was drawn from -- the drawn `10` comes back as `10`, not as a
    grid of marks.

    Derived from the same `split_top_level` and `_term_lines` the renderer uses,
    so it is a faithful transcript of what was drawn rather than a second
    opinion about it. It is a transcript, though, not a perceptual reading: no
    net looks at the pixels here. A rule that has to READ an unseen drawing
    still has to be learned, and `transcribe_unsafe` exists to keep that
    distinction honest.

    `12*30`      -> one box, the factors stacked
    `10*30+2*30` -> two boxes with `+` between them

    which is the whole distributive analogy, visible as a change in the number
    of boxes rather than as a string edit.
    """
    parts = split_top_level(text, "+-=")
    terms = [p for p in parts if p not in "+-="]
    seps = [p for p in parts if p in "+-="]

    rows = [f"    reads as:  {text}"]
    if seps:
        rows.append(f"    layout:    {len(terms)} boxes, joined by "
                    + " ".join(f"'{s}'" for s in seps))
    else:
        rows.append(f"    layout:    {len(terms)} box")
    for i, term in enumerate(terms, 1):
        lines = _term_lines(term)
        rows.append(f"      box {i}: {lines[0]}")
        for line in lines[1:]:
            rows.append(f"             {line}")
    return "\n".join(rows)


def png_bytes(img: np.ndarray) -> bytes:
    """A tape cell as PNG, in memory. Uses a minimal zlib/PNG writer so the
    package still has no image-library dependency.

    Separate from `save_png` because a specific-domain cell has two destinations
    that both need real pixels: the renders directory, where a human checks what
    a rule was looking at, and `propose.py`, which shows the same cell to the
    controller. A hypothesis summarised from a caption would not be a hypothesis
    summarised from the specific domain.
    """
    import struct
    import zlib

    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def save_png(img: np.ndarray, path: str) -> None:
    """Write a tape cell to disk."""
    with open(path, "wb") as f:
        f.write(png_bytes(img))


def _slug(text: str, limit: int = 40) -> str:
    """Caption -> something safe to put in a filename."""
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in text.strip()]
    return ("".join(keep).strip("_") or "cell")[:limit]


def save_gallery(items, out_dir: str, prefix: str = "", reset: bool = False) -> int:
    """Write a numbered run of tape cells to `out_dir`, for eyeballing.

    `items` is an iterable of (caption, image) pairs. The caption goes into the
    filename in slugged form AND verbatim into `index.txt` alongside it --
    validation means reading "what the machine thinks this is" next to the
    pixels, and a filename cannot carry a full prediction/truth pair without
    becoming unreadable.

    Appends to `index.txt` so several galleries can share one output directory;
    pass `reset=True` on the first call of a run to drop the previous run's
    index (the PNGs themselves are left alone -- they get overwritten).
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if reset:
        (out / "index.txt").unlink(missing_ok=True)
    lines, n = [], 0
    for i, (caption, img) in enumerate(items):
        if img is None:
            continue
        name = f"{prefix}{i:02d}_{_slug(caption)}.png"
        save_png(img, str(out / name))
        lines.append(f"{name}\t{caption}")
        n += 1
    if lines:
        with open(out / "index.txt", "a") as f:
            f.write("\n".join(lines) + "\n")
    return n
