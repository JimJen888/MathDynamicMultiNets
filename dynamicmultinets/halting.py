"""
The statistical anytime algorithm for the halting problem (paper, section 5).

Undecidable on a Turing machine, and the architecture's answer is not to decide
it exactly but to bound the error: sample N halting programs, take a high order
statistic of their running times as a threshold T, and declare anything still
running at T to be non-halting. The error of that declaration is at most eps
with confidence at least 1 - delta, PROVIDED the per-step reasoning error sigma
is small enough -- which is the paper's actual contribution, since on this
machine the steps are neural mapping rules and are not error-free.

    N = ceil( 1/(2 lambda^2) * ln(1/delta) )
    k = ceil( N(1 - eps + lambda) / (1 - 2 N sigma (1 - eps)) )
    T = t_(k)                       the k-th smallest observed running time

with lambda < eps and sigma = o(min(1/t_(N), 1/N^2)). The sigma term is what
inflates k: the less reliable each reasoning step, the further out the order
statistic has to be taken to keep the same guarantee. `calibrate` checks that
condition and refuses rather than returning a threshold whose guarantee does
not hold.

N is capped at MAX_SAMPLES. The formula grows as 1/lambda^2, so a lambda that
looks innocuous asks for a sample no run can produce, and an uncapped N turns a
tightened parameter into a hang. When the cap binds, the calibration reports the
lambda the sample really supports (`effective_lambda`) instead of the one that
was requested -- capping the cost must not also inflate the claim.

"Running time" here is measured in RULE APPLICATIONS, so this applies directly
to derivations on this machine -- `halting_budget_for_library` calibrates the
threshold from proof lengths the library actually produced, and the decision
"this derivation will not terminate" becomes a step budget with a stated error
rate instead of an arbitrary max_depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class HaltingCalibration:
    eps: float
    lam: float
    delta: float
    sigma: float
    n_samples: int
    order_k: int
    threshold: float
    times: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (f"N={self.n_samples} samples, sigma={self.sigma:.2e}, "
                f"order statistic k={self.order_k} of {self.n_samples}, "
                f"threshold T={self.threshold:g} steps\n"
                f"  guarantee: P(wrongly declaring non-halting) <= {self.eps} "
                f"with confidence >= {1 - self.delta}")


# Ceiling on N. The formula grows as 1/lambda^2, so a lambda a caller might
# reasonably type (1e-3) asks for ~1.5 million sampled halting programs, and a
# smaller one asks for a number no run can ever produce. Capping N keeps the
# requirement inside what a machine can actually sample; `effective_lambda`
# reports the lambda the cap really buys, so the guarantee stays honest instead
# of being quietly claimed at the lambda that was requested.
MAX_SAMPLES = 100_000


def sample_size(lam: float, delta: float, max_n: int | None = MAX_SAMPLES) -> int:
    """N(lambda, delta) = ceil( ln(1/delta) / (2 lambda^2) ), capped at `max_n`.

    Pass `max_n=None` for the uncapped formula. When the cap binds, the sample
    no longer supports the requested lambda -- `effective_lambda(N, delta)` is
    the one it does support, and `calibrate` reports that rather than the
    requested one.
    """
    if not 0 < lam < 1 or not 0 < delta < 1:
        raise ValueError("lambda and delta must lie in (0, 1)")
    if max_n is not None and max_n < 1:
        raise ValueError("max_n must be a positive number of samples")
    n = math.ceil(math.log(1.0 / delta) / (2.0 * lam * lam))
    return n if max_n is None else min(n, max_n)


def effective_lambda(n_samples: int, delta: float) -> float:
    """The lambda that `n_samples` halting programs actually buy.

    N(lambda, delta) inverted: lambda = sqrt( ln(1/delta) / (2N) ). Used to
    report the true guarantee when the sample is capped or simply smaller than
    the requested lambda would need, and to pick a feasible lambda from a
    sample the machine already has.
    """
    if n_samples < 1:
        raise ValueError("need at least one sampled halting program")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    lam = math.sqrt(math.log(1.0 / delta) / (2.0 * n_samples))
    if lam >= 1.0:
        # lambda is a deviation bound on a CDF; at or above 1 it says nothing.
        raise ValueError(
            f"{n_samples} sample(s) support no lambda below 1 at delta="
            f"{delta}: at least {sample_size(0.999, delta, None)} are needed "
            "before any guarantee exists at all"
        )
    # sqrt then square is not exact, so the analytic inverse can land a hair
    # below the true root and ask for n+1 samples -- which turns "fit lambda to
    # what I have" into a refusal by one. Nudge up until the round trip closes.
    while lam < 1.0 and sample_size(lam, delta, None) > n_samples:
        lam = math.nextafter(lam, 1.0)
    return lam


def max_step_error(times: Sequence[float], n: int) -> float:
    """The sigma = o(min(1/t_(N), 1/N^2)) bound, as a hard ceiling.

    Returned as the largest sigma still admissible; the algorithm needs
    strictly less than this, so `calibrate` compares with <.
    """
    t_max = max(times) if times else 1.0
    return min(1.0 / max(t_max, 1.0), 1.0 / (n * n))


def calibrate(
    running_times: Sequence[float],
    eps: float = 0.05,
    lam: float = 0.02,
    delta: float = 0.05,
    sigma: float = 0.0,
    max_samples: int | None = MAX_SAMPLES,
) -> HaltingCalibration:
    """Compute the threshold T from observed running times of halting programs.

    `running_times` must contain at least N = N(lambda, delta) samples, capped
    at `max_samples`; extra samples are used (a larger sample only tightens the
    empirical CDF). Pass `sigma` as the per-step error rate of the reasoning
    rules involved -- for a chain of learned rules, 1 - min(confidence) over the
    rules is the honest estimate, and `sigma=0` recovers the classical
    error-free statement.

    When the cap binds, the sample cannot support the lambda that was asked
    for, and the returned calibration carries the lambda it DOES support
    (`effective_lambda`) rather than the requested one -- reporting the
    requested lambda would be claiming a guarantee that was never paid for.
    """
    if not 0 < lam < eps < 1:
        raise ValueError("need 0 < lambda < eps < 1")
    n_needed = sample_size(lam, delta, max_samples)
    times = sorted(float(t) for t in running_times)
    if len(times) < n_needed:
        # Say what WOULD work: N grows as 1/lambda^2, so a handful of samples is
        # never enough at the default lambda=0.02, and the bare number does not
        # make the size of the gap obvious.
        feasible = effective_lambda(len(times), delta) if times else float("inf")
        hint = (f"raise lambda above {feasible:.4f} (it must stay below eps={eps})"
                if feasible < eps else
                f"no lambda below eps={eps} works for {len(times)} samples; "
                f"sample more halting programs, or raise eps")
        raise ValueError(
            f"need at least N={n_needed} sampled halting programs for "
            f"(lambda={lam}, delta={delta}); got {len(times)}. To calibrate "
            f"from what you have, {hint}."
        )
    n = len(times)
    # The sample, not the request, is what the guarantee rests on. When the cap
    # bound N, `n` is smaller than the requested lambda needs, so quote the
    # lambda `n` actually buys; when it did not, this is a no-op (a sample big
    # enough for lambda buys a lambda at least as tight).
    lam = max(lam, effective_lambda(n, delta))
    if lam >= eps:
        raise ValueError(
            f"a sample of {n} halting programs only supports lambda="
            f"{lam:.4f}, which is not below eps={eps}. Sample more programs "
            f"(N={sample_size(eps, delta, None)} would support eps={eps}), "
            "or raise eps."
        )

    ceiling = max_step_error(times, n)
    if sigma >= ceiling:
        raise ValueError(
            f"per-step error sigma={sigma:.3e} is too large for this sample: it "
            f"must be o(min(1/t_(N), 1/N^2)) = o({ceiling:.3e}). Verify the rules "
            "in the chain to a higher accuracy, or shorten the derivation."
        )

    denom = 1.0 - 2.0 * n * sigma * (1.0 - eps)
    if denom <= 0:
        raise ValueError("sigma too large: the order-statistic index diverges")
    k = math.ceil(n * (1.0 - eps + lam) / denom)
    k = max(1, min(k, n))                    # the paper's [1, N] validity claim
    return HaltingCalibration(eps, lam, delta, sigma, n, k, times[k - 1], times)


def decide(calibration: HaltingCalibration, run: Callable[[int], bool]) -> bool:
    """Run a program under the threshold. True == "declare it halts".

    `run(budget)` executes the program for at most `budget` steps and returns
    whether it stopped. Declaring non-halting is the only fallible direction:
    if it stopped, it stopped.
    """
    return bool(run(int(math.ceil(calibration.threshold))))


def halting_budget_for_library(
    proof_lengths: Sequence[int],
    min_rule_confidence: float = 1.0,
    eps: float = 0.05,
    lam: float | None = None,
    delta: float = 0.05,
) -> HaltingCalibration:
    """Turn observed proof lengths into a principled search budget.

    The programs are derivations, the running time is the number of rule
    applications, and sigma is one minus the confidence of the least reliable
    rule that may appear in one. The resulting `threshold` is the `max_depth`
    to give `proof.search`, with an explicit error rate attached -- which is
    strictly more than an arbitrary constant tells you.

    `lam=None` (the default) fits lambda to the sample instead of demanding a
    sample for a fixed lambda. A library's proofs number in the dozens, and
    lambda=0.02 asks for 3745 of them, so a fixed lambda makes this raise on
    every library that exists. Fitting it reports a weaker but true guarantee
    rather than no guarantee at all; pass a lambda explicitly to insist on one.
    """
    sigma = max(0.0, 1.0 - float(min_rule_confidence))
    if lam is None:
        if not proof_lengths:
            raise ValueError(
                "no halting programs to calibrate from: solve at least one "
                "benchmark task before asking for a search budget"
            )
        lam = effective_lambda(len(proof_lengths), delta)
        if lam >= eps:
            # Fitting lambda cannot rescue a sample this small, and saying
            # "need 0 < lambda < eps" would leave the caller with no idea what
            # to change. Name the sample size that would work.
            raise ValueError(
                f"{len(proof_lengths)} proof(s) support at most lambda="
                f"{lam:.3f}, which is not below eps={eps}: no honest budget "
                f"can be calibrated from them. About "
                f"{sample_size(eps, delta, None)} solved derivations would "
                f"support eps={eps} at delta={delta} -- add benchmark tasks, "
                "or ask for a looser eps."
            )
    return calibrate(proof_lengths, eps=eps, lam=lam, delta=delta, sigma=sigma)
