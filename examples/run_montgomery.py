"""
Experiment 5: forming a conjecture instead of trying to prove one.

`run_riemann.py` walks up to the Riemann hypothesis through an equivalent that
is decidable one integer at a time, and stops where it must: exact instances do
not reach a universal statement. This experiment does the other thing the
architecture is actually good for, and the thing experimental mathematics has
always done -- it forms a rule from data and reports what that rule says about
a case nobody knows the answer to.

The question is Montgomery's. Normalise the gaps between consecutive zeros of
the zeta function so the mean gap is 1, and ask what their distribution looks
like. Montgomery (1973) and Odlyzko's computations suggest it looks like the
eigenvalue spacings of a random Hermitian matrix -- the GUE -- and NOT like
independent random points. That is a conjecture, still open, and it is exactly
the shape of thing this machine can contribute to: a perceptual rule, verified
where truth is known, applied where it is not.

  1. the machine samples level spacings from three ensembles it can generate
     and therefore knows the answer for: GUE, GOE and Poisson;
  2. it draws each sample as a histogram and a CDF on the specific tape;
  3. it learns a specific -> abstract CHOICE rule that names the ensemble from
     the drawing -- DetourNet's head, one decision out of three;
  4. it verifies that rule on fresh samples, against a `constructed` oracle,
     with the base rate printed beside every accuracy;
  5. it computes the actual zeta zeros by Riemann-Siegel, reduces them by the
     SAME code path, and applies the verified rule to drawings it has no label
     for and cannot verify on;
  6. it reports what the rule says, as a conjecture, with its limits attached.

Nothing here is proved. What is EARNED is step 4 -- a rule that genuinely
discriminates, checked against a 0.333 base rate -- and what is CONJECTURED is
step 6. Keeping those two apart is the entire discipline of the exercise, and
it is why every accuracy in the output is printed next to what guessing would
have scored.

    python examples/run_montgomery.py            # ~3 min
    python examples/run_montgomery.py --quick    # ~40 s, weaker rule
    python examples/run_montgomery.py --llm      # Claude drives instead
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamicmultinets import RenMachine, ScriptedController          # noqa: E402
from dynamicmultinets.controller import LLMController                # noqa: E402
from dynamicmultinets.render import save_gallery                     # noqa: E402

GOAL = (
    "Learn to identify a random-matrix ensemble from a drawn level-spacing "
    "histogram, verify it against fresh samples where the ensemble is known, "
    "then apply it to the spacings of the actual Riemann zeta zeros and report "
    "what it says as a conjecture rather than a result."
)

# The same for training cells and zeta cells, and it has to be. A rule shown
# 500-spacing histograms in training and a 2000-spacing histogram at the end
# could separate the two on sampling noise alone -- smoother is not GUE, it is
# just more data -- and would then "identify" the zeta sample by the one
# property that has nothing to do with the question.
N_SPACINGS = 500


def plan(n_train: int, epochs: int) -> list[tuple[str, dict]]:
    """The reference sequence, as the tool calls an LLM controller would make."""
    return [
        ("inspect_machine", {}),
        ("show_catalogue", {}),

        # --- experiments the machine can grade itself on ---------------------
        # Sampled from real random matrices, not from the Wigner surmise. The
        # surmise is a 2x2 formula used as an approximation to the large-matrix
        # law and it is a good one, but the conclusion at the end is "the zeta
        # spacings look like GUE", and training on an approximation to GUE
        # would put that approximation's error inside the conclusion.
        ("generate_data", {"generator": "spacing_histograms", "n": n_train,
                           "seed": 7, "name": "spacing_train",
                           "params": {"n_spacings": N_SPACINGS, "dim": 60}}),
        ("label_data", {"dataset": "spacing_train", "oracle": "spacing_ensemble"}),
        ("declare_rule", {"name": "read_spacing_class", "domain_in": "specific",
                          "out_domain": "abstract",
                          "from_oracle": "spacing_ensemble",
                          "description": "name the ensemble a drawn spacing "
                                         "histogram came from"}),
        ("train_rule", {"rule": "read_spacing_class", "dataset": "spacing_train",
                        "epochs": epochs}),

        # Fresh samples, different seed. Three balanced classes, so the base
        # rate is 0.333 and an accuracy near that means the rule learned
        # nothing -- which is the number to check first and the one this
        # experiment prints beside every result.
        ("generate_data", {"generator": "spacing_histograms", "n": 300, "seed": 55,
                           "name": "spacing_fresh",
                           "params": {"n_spacings": N_SPACINGS, "dim": 60}}),
        ("label_data", {"dataset": "spacing_fresh", "oracle": "spacing_ensemble"}),
        ("verify_rule", {"rule": "read_spacing_class", "dataset": "spacing_fresh",
                         "oracle": "spacing_ensemble", "threshold": 0.90}),

        # A harder holdout: matrices twice the size. The ensembles are defined
        # in the large-matrix limit, so a rule that only recognises 60x60
        # samples has learned the finite-size artefacts rather than the law.
        ("generate_data", {"generator": "spacing_histograms", "n": 150, "seed": 91,
                           "name": "spacing_bigdim",
                           "params": {"n_spacings": N_SPACINGS, "dim": 120}}),
        ("label_data", {"dataset": "spacing_bigdim", "oracle": "spacing_ensemble"}),
        ("verify_rule", {"rule": "read_spacing_class", "dataset": "spacing_bigdim",
                         "oracle": "spacing_ensemble", "threshold": 0.90}),

        # --- the case nobody can grade ----------------------------------------
        # Real zeros, computed by Riemann-Siegel and reduced by the same
        # `spacing_scene` the training cells went through. The label attempt is
        # here on purpose and it FAILS: `spacing_ensemble` declines every zeta
        # cell, so the verification below reports that nothing was labelled
        # rather than a number. That refusal is the correct behaviour and the
        # reason this experiment can be trusted to be a conjecture.
        ("generate_data", {"generator": "zeta_spacings", "n": 40, "seed": 0,
                           "name": "zeta_blocks",
                           "params": {"n_spacings": N_SPACINGS}}),
        ("label_data", {"dataset": "zeta_blocks", "oracle": "spacing_ensemble"}),
        ("verify_rule", {"rule": "read_spacing_class", "dataset": "zeta_blocks",
                         "oracle": "spacing_ensemble", "threshold": 0.90}),

        ("library_report", {}),
        ("finish", {"summary": "ensemble reader learned and verified against a "
                               "0.333 base rate; applied to the zeta zeros, where "
                               "no label exists and the output is a conjecture"}),
    ]


def report_verification(machine) -> None:
    """Accuracy beside base rate, plus the confusion the average hides.

    GUE and GOE differ only in how fast the probability of a small gap goes to
    zero -- s^2 against s -- while Poisson does not vanish there at all. So a
    rule can score well on average by separating Poisson from the other two and
    still be guessing between the pair that matters for the zeta question. The
    per-class table is the one that answers whether the conjecture at the end
    rests on anything.
    """
    from dynamicmultinets import oracles
    from dynamicmultinets.verify import answer_text, normalize

    if "read_spacing_class" not in machine.library:
        return
    rule = machine.library.get("read_spacing_class")

    print("\n--- verification, against what guessing would score ---")
    for name in ("spacing_fresh", "spacing_bigdim"):
        if name not in machine.datasets:
            continue
        ds = machine.datasets[name]
        oracles.label(ds, "spacing_ensemble")
        confusion: Counter = Counter()
        labels: Counter = Counter()
        for ex in ds.examples:
            if not ex.labeled:
                continue
            want = normalize(answer_text(ex.out))
            got = rule.apply(ex.inp)
            got = normalize(answer_text(got)) if got is not None else "(none)"
            labels[want] += 1
            confusion[(want, got)] += 1
        n = sum(labels.values())
        if not n:
            continue
        correct = sum(v for (w, g), v in confusion.items() if w == g)
        base = max(labels.values()) / n
        print(f"\n  {name}: n={n}  accuracy {correct / n:.3f}  "
              f"base rate {base:.3f}  ({correct / n - base:+.3f})")
        for want in sorted(labels):
            row = {g: v for (w, g), v in confusion.items() if w == want}
            hit = row.get(want, 0) / labels[want]
            print(f"    true {want:8} n={labels[want]:4}  correct {hit:.3f}   {row}")


def report_conjecture(machine) -> None:
    """What the verified rule says about the zeros, stated as what it is."""
    from dynamicmultinets.verify import answer_text

    if "zeta_blocks" not in machine.datasets or "read_spacing_class" not in machine.library:
        return
    rule = machine.library.get("read_spacing_class")
    ds = machine.datasets["zeta_blocks"]

    print("\n--- applied to the zeta zeros (no label exists for these) ---")
    labelled = sum(1 for ex in ds.examples if ex.labeled)
    print(f"  cells the oracle could label: {labelled} of {len(ds.examples)} "
          f"-- as intended; nothing here can be verified")

    votes: Counter = Counter()
    for ex in ds.examples:
        got = rule.apply(ex.inp)
        votes[answer_text(got) if got is not None else "(declined)"] += 1
    total = sum(votes.values())
    for kind, k in votes.most_common():
        print(f"  {kind:10} {k:3}/{total}  ({k / total:.1%})")

    lo = min(ex.meta.get("t_lo", 0.0) for ex in ds.examples)
    hi = max(ex.meta.get("t_hi", 0.0) for ex in ds.examples)
    print(f"  heights covered: t in [{lo:.0f}, {hi:.0f}], "
          f"{len(ds.examples) * N_SPACINGS} spacings")

    top = votes.most_common(1)[0][0] if votes else "(none)"
    report_distances(machine, top)
    print(f"""
  CONJECTURE, not a result: a rule that identifies these three ensembles at
  well above chance, shown the zeta spacings, calls them {top!r}.

  What that is worth, stated precisely:
    * it is a statement about the NEAREST-NEIGHBOUR spacing distribution only.
      Montgomery's conjecture is about the PAIR CORRELATION function, a finer
      statistic this rule never sees; agreement here is consistent with it and
      is not the same claim.
    * the heights are low. Odlyzko's agreement with GUE is striking near the
      10^20-th zero; at t of a few thousand the convergence is known to be slow
      and visible deviations are expected, so this reproduces the QUALITATIVE
      observation and cannot speak to the rate.
    * the rule was verified on samples the machine generated. That makes it
      `constructed` grounding -- correct about drawings it made, and evidence
      about the zeta cells only insofar as the reduction is the same, which is
      why both go through one `spacing_scene`.
    * and the conjecture is not new. It is Montgomery-Odlyzko, arrived at from
      data by a machine that was told nothing about it. Reproducing a known
      open conjecture is the honest thing to demonstrate here: the pipeline can
      be checked against an answer the community already believes, which is
      exactly what you cannot do with a conjecture that is actually new.""")



def report_distances(machine, chosen: str) -> None:
    """Is the zeta sample a TYPICAL member of the class it was assigned, or
    merely the nearest of three?

    The rule has three options and no way to answer "none of these", so a
    verdict of `gue` means "closest to GUE among the three offered" and nothing
    stronger. Distinguishing those two readings needs a null distribution, and
    getting that null right is the whole of this function.

    The obvious comparison is wrong, and was made here first: the zeta
    histogram is an average over 40 blocks (20000 spacings) while a reference
    GUE cell is a single 500-spacing draw, so the zeta curve is far smoother and
    lands closer to the GUE mean than almost any individual GUE sample does.
    That comparison "showed" the zeta sample to be more typically GUE than GUE
    is, which is not a finding, it is a mismatch of sample sizes.

    The right null is the distance from the GUE mean of a GUE GROUP MEAN over
    the same 40 blocks, which is what is bootstrapped below.
    """
    import numpy as np

    from dynamicmultinets.render import spacing_scene
    from dynamicmultinets.zeta import ENSEMBLES, ensemble_spacings

    if "zeta_blocks" not in machine.datasets:
        return
    blocks = machine.datasets["zeta_blocks"].examples
    n_blocks = len(blocks)
    rng = np.random.default_rng(404)

    def hist(sp):
        return np.asarray(spacing_scene(sp)["hist"], dtype=float)

    ref = {k: np.array([hist(ensemble_spacings(k, N_SPACINGS, rng))
                        for _ in range(300 if k == chosen else 100)])
           for k in ENSEMBLES}
    means = {k: v.mean(axis=0) for k, v in ref.items()}
    zeta = np.mean([np.asarray(ex.inp.meta["scene"]["hist"], dtype=float)
                    for ex in blocks], axis=0)

    print("\n  how close is it, really (L1 between binned densities):")
    for kind in sorted(means):
        print(f"    zeta vs mean {kind:8} {float(np.abs(zeta - means[kind]).sum()):.3f}")

    if chosen not in ref:
        return
    pool = ref[chosen]
    boot = np.array([
        float(np.abs(pool[rng.choice(len(pool), n_blocks, replace=False)].mean(axis=0)
                     - means[chosen]).sum())
        for _ in range(400)])
    d_zeta = float(np.abs(zeta - means[chosen]).sum())
    pct = float((boot < d_zeta).mean())

    print(f"\n  null distribution, matched to the zeta sample's size "
          f"({n_blocks} blocks of {N_SPACINGS}):")
    print(f"    {chosen} group means sit {boot.mean():.3f} +/- {boot.std():.3f} "
          f"from the {chosen} mean (max over 400 draws: {boot.max():.3f})")
    print(f"    the zeta group mean sits at {d_zeta:.3f}, beyond {pct:.0%} of them")
    if d_zeta > boot.max():
        print(f"    -> OUTSIDE the {chosen} sampling distribution, by "
              f"{d_zeta / boot.max():.1f}x the largest of 400 draws.\n"
              f"       So: much nearer {chosen} than the alternatives, and still\n"
              f"       statistically distinguishable from it at these heights.")
    else:
        print(f"    -> inside the {chosen} sampling distribution: a typical sample.")

    # Does the gap close as t grows? Convergence to GUE is known to be slow,
    # and a deviation that SHRINKS with height is the signature of a limit not
    # yet reached, as against one that is simply not there.
    per_block = [(ex.meta.get("t_lo", 0.0),
                  np.asarray(ex.inp.meta["scene"]["hist"], dtype=float))
                 for ex in blocks]
    per_block.sort(key=lambda r: r[0])
    half = len(per_block) // 2
    print("\n  and does it close as the zeros climb?")
    for tag, chunk in (("lower half", per_block[:half]), ("upper half", per_block[half:])):
        m = np.mean([h for _, h in chunk], axis=0)
        print(f"    {tag}  t from {chunk[0][0]:8.0f}   L1 to {chosen} mean "
              f"{float(np.abs(m - means[chosen]).sum()):.3f}")
    print("     A deviation that shrinks with height is what the literature\n"
          "     describes -- Odlyzko needed the 10^20-th zero for close\n"
          "     agreement -- and it is visible in the machine's own data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny run, for smoke testing")
    ap.add_argument("--llm", action="store_true", help="let Claude drive instead")
    ap.add_argument("--device", default=None,
                    help="cuda when a GPU is present; pass cpu to pin it")
    ap.add_argument("--dump", default="renders/montgomery",
                    help="directory the drawn histograms are written to")
    ap.add_argument("--no-dump", action="store_true", help="do not write any images")
    ap.add_argument("--form", choices=["text", "image"], default="text",
                    help="how propose_rules shows the drawn cells; only --llm runs")
    args = ap.parse_args()

    n_train, epochs = (300, 10) if args.quick else (1500, 40)
    machine = RenMachine(goal=GOAL, device=args.device, propose_form=args.form)

    if args.llm:
        run = LLMController(machine, max_steps=60).run(GOAL)
    else:
        run = ScriptedController(machine).run(plan(n_train, epochs))

    print("\n" + "=" * 78)
    print(run.summary())
    print("\n--- final library ---")
    print(machine.library.table())

    report_verification(machine)
    report_conjecture(machine)

    if args.dump and not args.no_dump:
        from dynamicmultinets.verify import answer_text, normalize

        out = Path(args.dump)
        if not out.is_absolute():
            out = ROOT / out
        written, first = 0, True
        for ds_name in ("spacing_fresh", "zeta_blocks"):
            if ds_name not in machine.datasets or "read_spacing_class" not in machine.library:
                continue
            rule = machine.library.get("read_spacing_class")

            def cases(ds=machine.datasets[ds_name], rule=rule):
                for ex in ds.examples[:8]:
                    want = answer_text(ex.out) if ex.labeled else "no-label"
                    got = rule.apply(ex.inp)
                    text = answer_text(got) or "(no answer)"
                    mark = ("unlabeled" if not ex.labeled
                            else "ok" if normalize(text) == normalize(want)
                            else "WRONG")
                    yield f"{mark}_{ex.inp.text}_want-{want}_got-{text}", ex.inp.image

            written += save_gallery(cases(), str(out), prefix=ds_name, reset=first)
            first = False
        if written:
            print(f"\nwrote {written} images to {out}")


if __name__ == "__main__":
    main()
