"""
RuleNet -- the one network architecture every learned mapping rule is built on.

It is DetourNet (detourNet.py) with one generalisation. DetourNet is:

    two views -> shared ViewEncoder -> [fa, fb, fa-fb] -> trunk -> 19 logits

which answers a single multiple-choice question about a scene. A mapping rule
in the specific domain often has to answer a SEQUENCE of them -- "12 * 30"
rewritten as "10*30+2*30" is thirteen character decisions, not one -- so the
head is generalised from `num_classes` logits to `num_slots x num_classes`
logits, and `num_slots == 1` reproduces DetourNet exactly.

Everything else is kept verbatim, and for the same reasons:

  * RGB is re-quantized to one-hot semantic class channels up front, so the net
    reasons about geometry rather than colour bookkeeping (collisionNet).
  * CoordConv coordinate channels are prepended, because reading a rendered
    expression is a WHERE question -- which column a digit sits in is the whole
    signal -- exactly as the detour label was a sidedness question.
  * The encoder is shared across views and pools avg+max, so a one-pixel-wide
    stroke is not washed out by average pooling.
  * The fa - fb difference is fused in. For sketches it is the old disparity
    cue; for text it is "does the inline reading agree with the structured
    reading", which is what a structural rewrite rule actually turns on.

The slot head reads from the CONV FEATURE MAP rather than the pooled vector.
Pooling to 128 numbers and asking for 24 correlated character decisions back
throws away the column structure that makes the task easy; pooling the map to
one column strip per slot keeps it. The pooled global vector is concatenated to
every slot anyway, so a slot can still condition on the whole expression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .palette import GREY_CHROMA, NUM_CLASSES, PALETTE, PANEL_VALUE

_PALETTE_T = torch.tensor(PALETTE, dtype=torch.float32)
_BG_CLASS, _PANEL_CLASS, _FIRST_COLOUR = 0, 1, 2


def rgb_to_class_channels(img: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) RGB in [0,255] -> (B, K, H, W) one-hot class channels.

    Torch twin of palette.rgb_to_class_index; the two MUST agree, which
    tests/test_core.py checks on a rendered cell.
    """
    B, _, H, W = img.shape
    pal = _PALETTE_T.to(img.device, img.dtype)
    px = img.permute(0, 2, 3, 1).reshape(-1, 3)

    chroma = px.amax(dim=1) - px.amin(dim=1)
    is_grey = chroma < GREY_CHROMA
    bright = px.mean(dim=1) >= PANEL_VALUE

    coloured = torch.cdist(px, pal[_FIRST_COLOUR:]).argmin(dim=1) + _FIRST_COLOUR
    grey = torch.where(bright,
                       torch.full_like(coloured, _PANEL_CLASS),
                       torch.full_like(coloured, _BG_CLASS))
    idx = torch.where(is_grey, grey, coloured)

    oh = F.one_hot(idx, pal.size(0)).to(img.dtype)
    return oh.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


def class_index_to_channels(idx: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """(B, H, W) int class indices -> (B, K, H, W) one-hot channels.

    The fast path. Quantizing RGB costs a cdist over every pixel against every
    anchor, and in training that is paid again on every epoch for pixels that
    never change. Datasets therefore store the quantized INDEX map (one byte
    per pixel instead of eleven floats) and expand it here, which is a gather.
    The net is then built with preprocess=False. Inference from a live camera
    still goes through rgb_to_class_channels, since there the RGB is new.
    """
    return F.one_hot(idx.long(), num_classes).permute(0, 3, 1, 2).float()


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized (x, y) ramps -- CoordConv (Liu et al., 2018)."""
    b, _, h, w = x.shape
    xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, xs, ys], dim=1)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ViewEncoder(nn.Module):
    """Shared CNN over one 64x192 semantic-channel view.

    `feature_map` exposes the pre-pool map for the slot head; `forward` returns
    the pooled vector, byte-compatible in role with collisionNet.ViewEncoder.
    """

    FEAT_CH = 64

    def __init__(self, in_channels: int = NUM_CLASSES + 2, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(in_channels, 16),   # 192x64 -> 96x32
            conv_block(16, 32),            #        -> 48x16
            conv_block(32, 64),            #        -> 24x8
            conv_block(64, self.FEAT_CH),  #        -> 12x4
        )
        self.avg = nn.AdaptiveAvgPool2d((4, 4))
        self.max = nn.AdaptiveMaxPool2d((4, 4))   # keeps thin strokes alive
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * self.FEAT_CH * 4 * 4, feat_dim),
            nn.ReLU(inplace=True),
        )

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)
        return self.proj(torch.cat([self.avg(f), self.max(f)], dim=1))


class RuleNet(nn.Module):
    """A learned mapping rule's parameters.

    Args:
        num_classes: size of the output alphabet. For a single decision this is
                     the number of options (19 for DetourNet's action set); for
                     a transduction it is the vocabulary size.
        num_slots:   number of output positions. 1 == DetourNet.
        feat_dim:    per-view feature width.
        aux_dim:     optional numeric side-input width. 0 (the default) drops
                     the branch entirely -- DetourNet removed its motion vector
                     after measuring that the net had learned to ignore it, and
                     a dead input that is out of distribution at inference time
                     is worse than no input.
        use_diff:    fuse fa - fb.
        coordconv:   prepend coordinate ramps.
        preprocess:  forward() takes RGB in [0,255] and quantizes internally.
    """

    def __init__(
        self,
        num_classes: int,
        num_slots: int = 1,
        feat_dim: int = 128,
        in_channels: int = NUM_CLASSES,
        aux_dim: int = 0,
        use_diff: bool = True,
        coordconv: bool = True,
        preprocess: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.aux_dim = aux_dim
        self.use_diff = use_diff
        self.coordconv = coordconv
        self.preprocess = preprocess

        enc_in = in_channels + (2 if coordconv else 0)
        self.encoder = ViewEncoder(enc_in, feat_dim)         # shared across views

        fused = (3 if use_diff else 2) * feat_dim + (32 if aux_dim else 0)
        if aux_dim:
            self.aux_mlp = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        if num_slots == 1:
            self.head = nn.Linear(128, num_classes)
        else:
            # Per-slot head over the conv map: one width-strip per output slot,
            # concatenated with the global vector broadcast across slots.
            self.slot_head = nn.Sequential(
                nn.Conv1d(ViewEncoder.FEAT_CH + 128, 192, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(192, num_classes, kernel_size=1),
            )

    # -- plumbing ------------------------------------------------------------
    def _to_channels(self, img: torch.Tensor) -> torch.Tensor:
        ch = rgb_to_class_channels(img) if self.preprocess else img
        return add_coord_channels(ch) if self.coordconv else ch

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        xa, xb = self._to_channels(img_a), self._to_channels(img_b)
        fa, fb = self.encoder(xa), self.encoder(xb)
        parts = [fa, fb]
        if self.use_diff:
            parts.append(fa - fb)
        if self.aux_dim:
            if aux is None:
                aux = torch.zeros(img_a.size(0), self.aux_dim,
                                  device=img_a.device, dtype=img_a.dtype)
            parts.append(self.aux_mlp(aux))
        z = self.trunk(torch.cat(parts, dim=1))               # (B, 128)

        if self.num_slots == 1:
            return self.head(z)                               # (B, C)

        fmap = self.encoder.feature_map(xa)                    # (B, 64, h, w)
        strips = F.adaptive_avg_pool2d(fmap, (1, self.num_slots)).squeeze(2)  # (B,64,S)
        glob = z.unsqueeze(2).expand(-1, -1, self.num_slots)   # (B,128,S)
        logits = self.slot_head(torch.cat([strips, glob], dim=1))   # (B, C, S)
        return logits.permute(0, 2, 1).contiguous()            # (B, S, C)

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def predict(self, img_a: torch.Tensor, img_b: torch.Tensor,
                aux: torch.Tensor | None = None):
        """Returns (indices, probabilities). Shapes are (B,)/(B, C) for a single
        decision and (B, S)/(B, S, C) for a transduction."""
        self.eval()
        logits = self(img_a, img_b, aux)
        probs = logits.softmax(dim=-1)
        return probs.argmax(dim=-1), probs

    @torch.no_grad()
    def rank(self, img_a: torch.Tensor, img_b: torch.Tensor,
             mask: torch.Tensor | None = None):
        """Options best first -- the planner's trial loop (DetourNet.rank).
        Single-decision nets only; `mask` marks ALLOWED classes."""
        if self.num_slots != 1:
            raise ValueError("rank() is only meaningful for a single-decision rule")
        self.eval()          # as predict() does; dropout would reshuffle the tail
        logits = self(img_a, img_b)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits.argsort(dim=1, descending=True), logits.softmax(dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def batch_views(pairs: list[tuple[np.ndarray, np.ndarray]], device: str = "cpu"):
    """[(view_a, view_b), ...] as HWC uint8 -> two (B, 3, H, W) float tensors."""
    a = np.stack([p[0] for p in pairs]).astype(np.float32)
    b = np.stack([p[1] for p in pairs]).astype(np.float32)
    ta = torch.from_numpy(a).permute(0, 3, 1, 2).to(device)
    tb = torch.from_numpy(b).permute(0, 3, 1, 2).to(device)
    return ta, tb


if __name__ == "__main__":  # pragma: no cover -- shape smoke test
    for slots in (1, 24):
        net = RuleNet(num_classes=19 if slots == 1 else 18, num_slots=slots)
        a = torch.randint(0, 256, (2, 3, 64, 192), dtype=torch.float32)
        out = net(a, a)
        print(f"slots={slots:2d}  out={tuple(out.shape)}  params={net.n_params()/1e6:.2f}M")
