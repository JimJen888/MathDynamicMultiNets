"""
Experiment 6: a rule formed from data, and a transfer claim that can be TESTED.

Experiment 5 ends at a conjecture the machine cannot check: the zeta spacings
look GUE-like, and whether they are is open. This one takes the leftover of
that run -- the DEVIATION from GUE, which experiment 5 measured and set aside --
and asks whether it is lawful enough to state as a rule.

The rule the machine forms:

    p_zeta(s; t)  =  p_GUE(s)  +  g(s) / L  +  o(1/L),      L = log(t / 2pi)

a fixed shape g(s), amplitude falling as one over the log of the height. And
the reason this experiment exists next to `run_riemann.py` is the contrast in
what happens NEXT. Robin's criterion generalises to an infinite family and the
transfer can never be run. This one generalises to HIGHER t, and higher t is
reachable: fit g on low bands, predict bands the fit never saw, and the claim
either survives or does not. It is the same shape of claim -- established here,
conjectured there -- with the difference that this one is falsifiable in an
afternoon.

Measured, on 80000 zeros to t=61394, fitting on t<33190 and predicting above:

    residual against GUE alone     0.0748  0.0703  0.0715  0.0682
    residual against the rule      0.0372  0.0466  0.0449  0.0429
    improvement                     50.3%   33.8%   37.2%   37.1%     mean 39.6%
    shuffled-shape null            -46.4% +/- 10.5%  (max -14.6%)

What the shape says, physically: at these heights the zeros are MORE RIGID than
GUE. There is a deficit of small gaps (stronger repulsion than the random-matrix
law), an excess near the mean spacing, and a deficit again in the tail -- the
distribution is more tightly concentrated about its mean than GUE is.

This is very probably NOT a new result. A 1/log(t) correction is the standard
scale for finite-height effects in this subject, and Odlyzko's computations
reached 10^22 while this reaches 6*10^4. What is offered here is the
measurement, its controls, and a pipeline that produced it without being told
what to look for.

    python examples/run_finite_height.py            # ~2 min
    python examples/run_finite_height.py --quick    # ~30 s, 20000 zeros
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamicmultinets.render import SPACING_BINS, SPACING_MAX      # noqa: E402
from dynamicmultinets.zeta import (ensemble_spacings,              # noqa: E402
                                   unfolded_spacings, zeta_zeros,
                                   zero_counting_function)

TWO_PI = 2.0 * np.pi
DS = SPACING_MAX / SPACING_BINS
CENTRES = (np.arange(SPACING_BINS) + 0.5) * DS


def density(s: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(s, bins=SPACING_BINS, range=(0.0, SPACING_MAX), density=True)
    return h


def l1(x: np.ndarray) -> float:
    return float(np.abs(x).sum() * DS)


def bands(zeros: np.ndarray, n_bands: int):
    """Split the spacings into equal-count height bands.

    Equal COUNT, not equal width in t. Each band then carries the same
    statistical weight, so the band-to-band comparison that the whole
    experiment rests on is not dominated by whichever band happened to be
    widest.
    """
    sp = unfolded_spacings(zeros)
    heights = zeros[1:]
    edges = np.linspace(0, len(sp), n_bands + 1).astype(int)
    for lo, hi in zip(edges, edges[1:]):
        t_mid = float(np.exp(np.mean(np.log(heights[lo:hi]))))
        yield (np.log(t_mid / TWO_PI), float(heights[lo]), float(heights[hi - 1]),
               density(sp[lo:hi]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="20000 zeros instead of 80000")
    ap.add_argument("--zeros", type=int, default=0, help="override the zero count")
    ap.add_argument("--bands", type=int, default=8)
    args = ap.parse_args()

    n_zeros = args.zeros or (20000 if args.quick else 80000)
    rng = np.random.default_rng(0)

    print(f"computing {n_zeros} zeros by Riemann-Siegel ...")
    z = zeta_zeros(n_zeros)
    missing = zero_counting_function(float(z[-1])) - n_zeros
    print(f"  reached t={z[-1]:.0f}; completeness check N(T)-found = {missing:+.1f} "
          f"(a skipped PAIR would show here, and would fake level repulsion)")
    if abs(missing) > 3:
        print("  WARNING: zeros appear to be missing; the deviation below is not "
              "trustworthy, since what gets skipped is the closest pairs.")

    # The GUE reference has to be far better resolved than any single band, or
    # its own noise enters every comparison as if it were signal.
    ref = np.concatenate([ensemble_spacings("gue", 200000, rng, dim=60)
                          for _ in range(3)])
    p_gue = density(ref)
    print(f"  GUE reference from {len(ref)} spacings")

    # Power: the deviation is only ~2x the per-band noise floor, so bands need
    # roughly 3000 spacings before it is measurable at all. Measured gains on
    # held-out bands -- 1500/band: +3%, 3000/band: +37%, 5000/band: +47%. A run
    # that splits too finely will report no effect and mean nothing by it.
    per_band = (n_zeros - 1) // args.bands
    if per_band < 2500:
        print(f"  NOTE: {per_band} spacings per band is below the ~3000 needed "
              f"to see the deviation; use fewer bands or more zeros.")

    rows = list(bands(z, args.bands))
    Ls = np.array([r[0] for r in rows])
    devs = np.array([r[3] - p_gue for r in rows])

    print(f"\n{'band':5}{'t range':>22}{'L':>6}{'L1 vs GUE':>11}")
    for i, (L, lo, hi, _) in enumerate(rows):
        print(f"{i:<5}{f'{lo:.0f}-{hi:.0f}':>22}{L:6.2f}{l1(devs[i]):11.4f}")

    # ---- the rule, fitted on the LOWER half only -------------------------
    half = len(rows) // 2
    g = np.mean([devs[i] * Ls[i] for i in range(half)], axis=0)
    print(f"\nrule fitted on bands 0-{half-1} (t < {rows[half][1]:.0f}) only:")
    print(f"  p_zeta(s;t) = p_GUE(s) + g(s)/L,  L = log(t/2pi)")
    print(f"  g integrates to {float(g.sum() * DS):+.4f} (must be ~0: both are densities)")

    print(f"\n  s      g(s)")
    for i in range(0, SPACING_BINS, 2):
        bar = "#" * int(min(abs(g[i]) * 40, 60))
        print(f"  {CENTRES[i]:4.2f}  {g[i]:+7.4f}  {'-' if g[i] < 0 else '+'}{bar}")

    # ---- transfer: predict the bands the fit never saw --------------------
    print(f"\n--- transfer to bands {half}-{len(rows)-1}, never used in the fit ---")
    print(f"{'band':5}{'L':>6}{'vs GUE':>10}{'vs rule':>10}{'improvement':>13}")
    imp = []
    for i in range(half, len(rows)):
        a, b = l1(devs[i]), l1(devs[i] - g / Ls[i])
        imp.append(1.0 - b / a)
        print(f"{i:<5}{Ls[i]:6.2f}{a:10.4f}{b:10.4f}{1.0 - b / a:12.1%}")
    mean_imp = float(np.mean(imp))
    print(f"{'mean':5}{'':6}{'':10}{'':10}{mean_imp:12.1%}")

    # Is it the SHAPE doing the work, or just the amplitude? Scrambling g
    # across bins keeps its magnitude and destroys its shape.
    null = np.array([
        float(np.mean([1.0 - l1(devs[i] - g[rng.permutation(len(g))] / Ls[i]) / l1(devs[i])
                       for i in range(half, len(rows))]))
        for _ in range(400)])
    print(f"\n  shuffled-shape null: {null.mean():+.1%} +/- {null.std():.1%} "
          f"(max {null.max():+.1%})")
    print(f"  observed {mean_imp:+.1%} -> percentile {float((null < mean_imp).mean()):.1%}")

    verdict = ("SURVIVES" if mean_imp > null.max() else "DOES NOT SURVIVE")
    print(f"""
--- what this is ---
  The transfer {verdict}. A shape fitted below t={rows[half][1]:.0f} predicts the
  deviation above it, on data the fit never saw, and a scrambled shape of the
  same size makes the fit WORSE -- so the structure, not the magnitude, is
  carrying the prediction.

  Read it as a measurement with controls, not as a theorem and not as news.
  A 1/log(t) correction is the standard scale for finite-height effects here,
  and published computations run to heights this one cannot approach. What is
  demonstrated is the pipeline: the machine formed a rule from its own
  experiments, stated the transfer, and then actually RAN it -- which is the
  step `run_riemann.py` can never take, because there the unknown family is
  infinite and here it is merely higher up.""")


if __name__ == "__main__":
    main()
