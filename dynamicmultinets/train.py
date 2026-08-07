"""
Training a mapping rule, and growing it into an ensemble.

Two things happen here, and the second is the one that matters for the paper's
accuracy claim.

1. `train_rule` fits a RuleNet on an ExampleSet through the rule's codec. The
   loss is cross-entropy with softened inverse-frequency class weights, copied
   from detourNet.make_criterion for the same reason it exists there: the label
   distribution is wildly imbalanced (most slots of a padded expression are
   padding; most robot scenes need no detour), and unweighted training produces
   a net that always answers with the majority class and reports a fine
   accuracy while doing it. Full inverse frequency over-corrects, so the
   exponent is a dial (`weight_power`, default 0.5) rather than a switch.

2. `grow_ensemble` implements section 3's construction for getting a mapping
   rule close to 100%: train a base rule, find the cases it gets wrong, train a
   specialist on those, and compose by letting the specialist override the base
   where it claims competence. The specialist's gate is mined, not hand-written
   -- it is "this input resembles the ones the base failed on", measured in the
   base encoder's own feature space, so it transfers to inputs never seen.

Both report rather than print: a tool call returns the report to the controller,
which is what decides what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .dataset import ExampleSet, RuleDataset, encode_pair
from .rules import EnsembleRule, NeuralRule, Recipe, Rule, RuleLibrary
from .tapes import Content


@dataclass
class TrainReport:
    rule: str
    n_train: int = 0
    n_holdout: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    holdout_exact: float = 0.0        # every slot correct
    holdout_slot: float = 0.0         # per-slot accuracy (== exact when slots==1)
    best_epoch: int = 0               # the epoch whose weights were kept
    dropped: list[str] = field(default_factory=list)
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def summary(self) -> str:
        drop = f", {len(self.dropped)} dropped" if self.dropped else ""
        best = f" (kept epoch {self.best_epoch})" if self.best_epoch else ""
        return (f"trained {self.rule}: {self.n_train} train / {self.n_holdout} holdout"
                f"{drop}, {self.epochs} epochs{best}, loss {self.final_loss:.4f}, "
                f"holdout exact {self.holdout_exact:.3f}, slot {self.holdout_slot:.3f}")


def _criterion(targets: np.ndarray, num_classes: int, weight_power: float,
               label_smoothing: float, device: str):
    """Cross-entropy with count**(-weight_power) class weights.

    weight_power = 1.0 is full inverse frequency, 0.0 is uniform, 0.5 is the
    softened middle that detourNet settled on after full inverse frequency was
    measured to over-fire the rare classes.
    """
    import torch
    import torch.nn as nn

    counts = np.bincount(np.asarray(targets).reshape(-1), minlength=num_classes)
    w = np.where(counts > 0, np.maximum(counts, 1.0) ** (-weight_power), 0.0)
    if w[w > 0].size:
        w = w / w[w > 0].mean()
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def _seed_everything(seed: int) -> None:
    """Pin every source of randomness a training run draws on.

    `torch.manual_seed` alone is not enough on a GPU: cuDNN chooses its
    convolution algorithms by benchmarking and several of the fast ones
    accumulate in nondeterministic order, so the same data with the same seed
    lands on different weights every run. Measured here, one rule trained three
    times on identical data verified at 0.970, 0.978 and 0.983.

    That spread is invisible until a rule is checked against a threshold, and
    then it decides trust: a rule whose true accuracy sits near 0.95 is trusted
    or not depending on which run you happened to make, which is the one thing
    verification is supposed to rule out. Determinism costs a little speed and
    buys a number that means the same thing twice.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Process-global, like torch's own seeding: a rule is trained one at a time
    # and the whole point is that the next run agrees with this one.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_rule(
    rule: NeuralRule,
    example_set: ExampleSet,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 1e-3,
    holdout: float = 0.2,
    weight_power: float = 0.5,
    label_smoothing: float = 0.05,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> TrainReport:
    """Fit `rule`'s net on `example_set`; returns a report, prints nothing."""
    import torch
    from torch.utils.data import DataLoader

    _seed_everything(seed)
    train_set, hold_set = example_set.split(holdout)
    ds_train = RuleDataset(train_set, rule.codec)
    ds_hold = RuleDataset(hold_set, rule.codec)
    report = TrainReport(rule=rule.name, n_train=len(ds_train), n_holdout=len(ds_hold),
                         epochs=epochs, dropped=ds_train.dropped + ds_hold.dropped)
    if not len(ds_train):
        raise ValueError(
            f"no usable labeled examples for {rule.name!r}; "
            f"first {len(report.dropped)} problems: {report.dropped[:3]}"
        )

    device = rule.device
    net = rule.net.to(device)
    targets = np.stack([it[2] for it in ds_train.items])
    criterion = _criterion(targets, rule.codec.num_classes, weight_power,
                           label_smoothing, device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    from .nets import class_index_to_channels

    # Keep the best-scoring weights rather than the last ones. Holdout accuracy
    # on these tasks is not monotone -- a run that peaks at 0.98 can end at
    # 0.92 -- and a rule is about to be VERIFIED against a threshold, so
    # shipping whatever the last epoch happened to produce would make trust
    # depend on where training stopped.
    best_state, best_exact, best_epoch = None, -1.0, -1

    for epoch in range(epochs):
        net.train()
        total, n = 0.0, 0
        for ia, ib, y in loader:
            xa = class_index_to_channels(ia).to(device)
            xb = class_index_to_channels(ib).to(device)
            y = y.to(device).long()
            logits = net(xa, xb)
            loss = criterion(logits.reshape(-1, rule.codec.num_classes), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * y.size(0)
            n += y.size(0)
        report.final_loss = total / max(n, 1)
        exact, slot = _evaluate(net, ds_hold, rule.codec.num_classes, device)
        report.history.append((epoch, report.final_loss, exact))
        if exact > best_exact:
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_exact, best_epoch = exact, epoch
        if log:
            log(f"  epoch {epoch + 1:>3}/{epochs}  loss {report.final_loss:.4f}  "
                f"holdout exact {exact:.3f}")

    if best_state is not None and len(ds_hold):
        net.load_state_dict(best_state)
        report.best_epoch = best_epoch + 1
    report.holdout_exact, report.holdout_slot = _evaluate(
        net, ds_hold, rule.codec.num_classes, device)

    rule.recipe = Recipe(
        generator=example_set.generator,
        generator_params=dict(example_set.generator_params),
        oracle=example_set.oracle,
        n_examples=len(example_set),
        epochs=epochs,
        seed=seed,
        architecture=rule.recipe.architecture,
    )
    return report


def _evaluate(net, dataset, num_classes: int, device: str) -> tuple[float, float]:
    """(exact-match accuracy, per-slot accuracy) on a RuleDataset."""
    import torch
    from torch.utils.data import DataLoader

    from .nets import class_index_to_channels

    if not len(dataset):
        return 0.0, 0.0
    net.eval()
    exact_hits = slot_hits = slot_total = n = 0
    with torch.no_grad():
        for ia, ib, y in DataLoader(dataset, batch_size=64):
            xa = class_index_to_channels(ia).to(device)
            xb = class_index_to_channels(ib).to(device)
            pred = net(xa, xb).argmax(dim=-1).cpu()
            y = y.long()
            if pred.dim() == 1:
                pred, y = pred.unsqueeze(1), y.unsqueeze(1)
            match = pred == y
            exact_hits += int(match.all(dim=1).sum())
            slot_hits += int(match.sum())
            slot_total += int(match.numel())
            n += y.size(0)
    return exact_hits / max(n, 1), slot_hits / max(slot_total, 1)


# ---------------------------------------------------------------------------
# Hard-case mining and specialists (paper, section 3)
# ---------------------------------------------------------------------------
def find_failures(rule: Rule, example_set: ExampleSet) -> list[int]:
    """Indices of labeled examples the rule currently gets wrong."""
    from .verify import answer_text, normalize

    bad = []
    for i, ex in enumerate(example_set.examples):
        if not ex.labeled:
            continue
        got = rule.apply(ex.inp)
        if got is None or normalize(answer_text(got)) != normalize(answer_text(ex.out)):
            bad.append(i)
    return bad


def _feature_gate(rule: NeuralRule, hard: ExampleSet, quantile: float = 0.3
                  ) -> Callable[[Content], bool]:
    """Gate that fires on inputs resembling the base rule's failures.

    Built from the BASE rule's own encoder: embed every hard case, keep the
    centroid, and claim an input whose cosine similarity to it beats the
    threshold set by the hard cases themselves. Using the base's features is
    the point -- the specialist should take over exactly where the base's
    representation says "this looks like the thing I get wrong", which is a
    property of the base, not of the raw pixels.
    """
    import torch

    from .nets import class_index_to_channels

    net = rule.net
    net.eval()

    def embed(content: Content) -> "torch.Tensor":
        ia, ib = encode_pair(rule.codec, content)
        xa = class_index_to_channels(torch.from_numpy(ia.astype(np.int64))[None])
        with torch.no_grad():
            f = net.encoder(net._to_channels(xa.to(rule.device)))
        return torch.nn.functional.normalize(f, dim=1)[0]

    embeddings = torch.stack([embed(ex.inp) for ex in hard.examples])
    centroid = torch.nn.functional.normalize(embeddings.mean(dim=0, keepdim=True), dim=1)[0]
    sims = (embeddings @ centroid).cpu().numpy()
    threshold = float(np.quantile(sims, 1.0 - quantile)) if len(sims) else 1.0

    def gate(content: Content) -> bool:
        try:
            return float(embed(content) @ centroid) >= threshold
        except Exception:
            return False

    return gate


def grow_ensemble(
    library: RuleLibrary,
    base_name: str,
    example_set: ExampleSet,
    ensemble_name: str | None = None,
    epochs: int = 12,
    min_failures: int = 8,
    min_specialist_gain: float = 0.005,   # required held-out accuracy gain
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> tuple[Rule, TrainReport | None, str]:
    """Train a specialist on the base rule's failures and compose the two.

    Returns (rule_in_library, specialist_report, note). The base is returned
    unchanged in two cases: too few failures to learn from (a specialist fitted
    to three examples is a memorization of three examples, and it would then
    claim cases it knows nothing about), or an ensemble that did not beat the
    base rule by `min_specialist_gain` on held-out examples. `note` says which
    happened, in the words the controller should act on.

    PASS A SET THE BASE WAS NOT TRAINED ON. Failures are mined by running the
    base over `example_set`, so handing it the base's own training data measures
    how well the base memorised rather than where it is wrong: a rule that
    verifies at 0.61 on fresh scenes can show almost no failures on the data it
    was fitted to, and the construction then silently does nothing. The
    diagnostic in `no_failures_hint` reports that case rather than leaving it to
    look like "this rule has no hard cases".
    """
    base = library.get(base_name)
    if not isinstance(base, NeuralRule):
        raise TypeError("only a learned rule can be grown into an ensemble")

    # Failures are mined from one slice and the ensemble is judged on another.
    # Mining and judging on the same examples measures whether the specialist
    # memorised them, which it always can.
    fit_set, check_set = example_set.split(0.25)
    failures = find_failures(base, fit_set)
    if log:
        log(f"  {len(failures)}/{len(fit_set.labeled)} failures on {fit_set.name}")
    if len(failures) < min_failures:
        return base, None, no_failures_hint(base, failures, fit_set)

    hard = fit_set.subset(failures, name=f"{example_set.name}:hard")
    example_set = fit_set
    name = ensemble_name or f"{base_name}_ens"
    specialist = NeuralRule(
        f"{base_name}_spec", type(base.codec)(**_codec_kwargs(base.codec)),
        base.domain_in, f"specialist for {base_name} hard cases", device=base.device,
    )
    # Oversample: a specialist trained only on failures never sees a case it
    # should decline, so a slice of ordinary examples goes in with them.
    easy_idx = [i for i in range(len(example_set.examples))
                if i not in set(failures)][: len(failures)]
    # Sorted, so hard and easy cases interleave: train_rule holds out the TAIL,
    # and an unsorted concatenation would hold out nothing but easy cases.
    mixed = example_set.subset(sorted(failures + easy_idx),
                               name=f"{example_set.name}:mixed")
    mixed.generator, mixed.oracle = example_set.generator, example_set.oracle
    report = train_rule(specialist, mixed, epochs=epochs, seed=seed, log=log)

    library.add(specialist, replace=True)

    # The construction only makes sense if the specialist is actually better
    # where it claims competence. Composing by "prioritising the specialised
    # model's output" presupposes that; without the check, a specialist that
    # learned nothing from a hundred hard cases silently overrides a base rule
    # that was working, and the ensemble verifies WORSE than what it replaced.
    ens = EnsembleRule(name, base, description=f"{base_name} + specialist override")
    ens.add_specialist(_feature_gate(base, hard), specialist)

    # The section-3 claim -- base plus specialists beats the base -- is a claim,
    # so check it on examples neither of them was fitted to. A specialist that
    # memorised its hard cases and then claims unrelated ones makes the
    # composite WORSE, and the only way to see that is a held-out comparison.
    base_acc = accuracy_on(base, check_set)
    ens_acc = accuracy_on(ens, check_set)
    if log:
        log(f"  held-out check on {len(check_set.labeled)}: base {base_acc:.3f} "
            f"-> ensemble {ens_acc:.3f}")
    if ens_acc <= base_acc + min_specialist_gain:
        library.remove(specialist.name)
        if log:
            log("  the specialist did not improve the rule where it claims "
                "competence; discarded, base rule left alone")
        return base, report, (
            f"held-out check on {len(check_set.labeled)}: base {base_acc:.3f} "
            f"-> ensemble {ens_acc:.3f}, no gain")

    library.add(ens, replace=True)
    return ens, report, (
        f"held-out check on {len(check_set.labeled)}: base {base_acc:.3f} "
        f"-> ensemble {ens_acc:.3f}")


def no_failures_hint(base: Rule, failures: Sequence[int], fit_set: ExampleSet) -> str:
    """Why a rule showed no hard cases -- 'it has none' is rarely the reason.

    A base rule scores far better on data it was fitted to than on fresh data,
    so mining its training set finds nothing to specialise on however wrong the
    rule is in the field. Comparing the mined rate against what verification
    already measured says which of the two situations this is, and the two call
    for opposite responses: generate a fresh set, or leave the rule alone.
    """
    n = len(fit_set.labeled)
    mined = len(failures) / n if n else 0.0
    verified = base.stats.accuracy
    if base.stats.n_checked and (1.0 - verified) > 4.0 * mined + 0.02:
        return (f"only {len(failures)}/{n} failures on {fit_set.name}, but this "
                f"rule verifies at {verified:.3f} on fresh data -- {fit_set.name} "
                "looks like data the base was trained on, so this measures "
                "memorisation, not hard cases. Mine a freshly generated set "
                "instead.")
    return (f"only {len(failures)}/{n} failures on {fit_set.name}: too few to "
            "fit a specialist that would not simply memorise them")


def accuracy_on(rule: Rule, example_set: ExampleSet) -> float:
    """Fraction of labeled examples the rule gets right."""
    n = len(example_set.labeled)
    return (n - len(find_failures(rule, example_set))) / n if n else 0.0


def _codec_kwargs(codec) -> dict:
    """Rebuild kwargs for a codec of the same shape as an existing one."""
    cfg = dict(codec.config())
    cfg.pop("type", None)
    return cfg
