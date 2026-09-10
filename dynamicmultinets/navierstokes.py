"""
The forced Navier-Stokes blowup construction, expressed as rules.

This module encodes the OpenAI preprint "Finite time blowup for Navier-Stokes"
in the only terms this machine has: mapping rules over cells, an oracle that
can check them, and a trust flag that says which of them the machine actually
believes. It is worth being exact about what that can and cannot mean.

WHAT THE PAPER CLAIMS. For every viscosity there is a force in C_c^inf, a
compact set K, and a smooth solution on R^3 x [0,1) starting from rest, with
support in K, uniformly bounded kinetic energy, and sup|u| -> infinity as
t -> 1 (Theorem 1.1). This is a FORCED breakdown -- alternatives (C) and (D) of
Fefferman's problem description. The unforced Cauchy problem, alternatives (A)
and (B), is a different statement and nothing here touches it; the force is
necessarily nonzero, since the energy identity (10.13) would otherwise give
u = 0.

HOW IT IS RE-PROVED HERE. Not by importing the paper's propositions. Every
mechanism the construction rests on is turned into a rule the machine FORMS
and then CHECKS against an oracle that decides instances by computing them:

  mechanism (paper)                rule                  oracle that decides it
  -------------------------------  --------------------  ----------------------
  energy/dissipation budget (3.5)  ns_energy_decide      numerical quadrature
  admissible cone (Lemma 4.5)      ns_cone_quadratic     the covariance weights
                                                         actually solved for
  exact heat exterior (Lemma A.6)  ns_heat_exterior      finite differences on
                                                         the field itself
  pulse growth then decay (7.4)    ns_pulse_envelope     integrating the
                                                         amplitude equation
  moment corrections (Lemma A.1)   ns_moment_exponents   the determinant of the
                                                         matrix, built
  correction cycle (9.6)           ns_cycle_closed       iterating the recursion

Each of those rules is declared UNVERIFIED. It states a claim -- that a cheap
test decides an expensive question -- and it becomes trusted only when it has
agreed with its oracle on data it was not formed from, with the base rate
printed beside the accuracy. That is the whole loop the architecture is built
around, and here it is doing real work: before verification the benchmark
tasks fail, and after it they are proved.

WHAT THE CHECKS DO NOT SEE. Agreement is only worth what a disagreement
would have cost, so `MUTATIONS` below runs each rule again with one thing
deliberately wrong and the run prints how often the oracle objects. Four
limits survive that audit and are not repaired by more data:

  * `lemma_7_4_pulse` and `pulse_by_integration` compute their rates from the
    same function. The rule tests the sign of that rate at two points, the
    oracle integrates it, so they are different computations over a SHARED
    MODEL -- an error in the rates themselves is invisible to this check. The
    oracle also drops the frame error E of (7.10), so what it confirms is the
    reference amplitude equation, not (7.5) in full.
  * `covariance_positive_weights` builds the reference covariance directions
    of (7.28) with the error term e_sigma dropped and the normalizations
    fixed at one. It exercises the mechanism at leading order; that the
    errors preserve positivity is part of Proposition 7.5 and is imported.
  * `cone_by_definition` shares the coordinate map (4.20) with the rule it
    checks. The covariance route does not, which is why that one is the
    primary oracle for the cone and this one is the second opinion.
  * `prop_9_6_decay` checks that the closed form of sigma_j matches the
    recursion. That a correction cycle gains 1/10 in the first place is the
    estimate of Proposition 9.6 and is imported.
  * `ns_core_scales` and `ns_core_energy` are definitions written here, not
    rules checked against a field. `energy_by_quadrature` decides the verdict
    from exponents it is handed; that those are the exponents of the
    constructed velocity is read off the paper and is not tested.

WHERE IT STOPS, AND WHY. Two gaps, and neither is closed by running longer.

  * Every oracle above decides an INSTANCE. The paper's statements are
    universal -- over all q as q goes to zero, over every dyadic band and
    label, over every correction stage j, over an infinite tower of profile
    coefficients. A rule verified on ten thousand instances is evidence of
    exactly the kind that already exists in abundance and is not a proof.
  * The estimates themselves -- Theorem 4.6, Propositions 5.5, 7.5, 9.6, 9.9,
    the localization and comparison of Section 10, the three appendices --
    are bounds on function spaces. No rule over tape cells can carry one, and
    the machine does not pretend to. They appear at the end of the run as the
    dependency skeleton, as rules that are exact-free and untrusted, so that
    a proof search reaching Theorem 1.1 finds them and reports the chain as
    unproved. Reaching that cell at confidence 1.0 would be the machine
    reading its own handwriting.

Every quantity below is a rational, carried as a `Fraction` and written on the
tape as `p/q`, so the exponent algebra is exact and a rule that computes it
cannot be wrong by rounding. The two exceptions are stated where they occur.

Install with `install_navier_stokes_rules(machine.library)`; the rules are kept
out of `prior.PRIOR_RULES` until then, so a machine built for another problem
does not carry them.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Callable

import numpy as np

from .dataset import Example
from .linarith import Lin, obligations, short, solve
from .provers import Judgement, prover
from .symalg import Poly
from .generators import generator
from .oracles import oracle
from .prior import PRIOR_RULES
from .rules import PythonRule, Rule, RuleLibrary
from .tapes import ABSTRACT, Content

# numpy renamed the trapezoid rule in 2.0 and this repo runs against 1.26.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

NS_RULES: dict[str, Callable[[], Rule]] = {}


def ns_rule(name: str):
    """Register a factory. Kept in a separate registry from the built-ins: a
    multiplication machine has no business carrying the cone condition."""

    def wrap(factory: Callable[[], Rule]) -> Callable[[], Rule]:
        NS_RULES[name] = factory
        return factory

    return wrap


# ---------------------------------------------------------------------------
# Rationals on the tape
# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:/\d+)?"


def fr(text: str) -> Fraction:
    return Fraction(text)


def sf(x: Fraction) -> str:
    """A rational as it is written on the tape: `97/200`, `-1/2`, `0`."""
    return str(Fraction(x))


def _fields(text: str, *names: str) -> dict[str, Fraction] | None:
    """Parse `a=1/2,b=-3` into rationals, requiring exactly `names` in order."""
    pattern = ",".join(rf"{re.escape(n)}=({_NUM})" for n in names)
    m = re.fullmatch(rf"\s*{pattern}\s*", text.replace(" ", ""))
    if not m:
        return None
    return {n: fr(g) for n, g in zip(names, m.groups())}


def _tagged(text: str, tag: str) -> Fraction | None:
    """Parse `<tag>,h=1/200`, the carrier used along the derivation chain."""
    m = re.fullmatch(rf"\s*{re.escape(tag)},h=({_NUM})\s*", text.replace(" ", ""))
    return fr(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# The concentrating core: Section 2.1, 3.1 and (10.20)-(10.23)
# ---------------------------------------------------------------------------
@ns_rule("ns_h_from_integer")
def make_ns_h_from_integer() -> PythonRule:
    """`200` -> `h=1/200`. The one rule that turns a bare integer into the
    construction's only free exponent, so a DRAWN integer could start the
    chain once something can read it. Nothing here can; see the observed task
    in the example run."""

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*(\d+)\s*", c.text)
        if not m or int(m.group(1)) < 2:
            return None
        return Content.abstract(f"h=1/{m.group(1)}", derivation="h from denominator")

    return PythonRule("ns_h_from_integer", fn, ABSTRACT, ABSTRACT,
                      description="an integer n names the exponent h=1/n",
                      source="n->h=1/n")


@ns_rule("ns_exponents")
def make_ns_exponents() -> PythonRule:
    """`h=1/200` -> `A=101/200,D=99/200,h=1/200`.

    The two exponents of (4.1): the tangential velocity grows like tau^-A with
    A = 1/2+h, and the axial length shrinks like tau^D with D = 1/2-h. The
    identity A+D = 1 is what makes every transport product in the residual
    carry the same power of q, and it is checked here rather than assumed.

    Declines outside 0 < h < 1/2, where the similarity coordinates of (4.1)
    stop being uniquely solvable. Whether h also clears the tighter 1/100 of
    Theorem 4.6 is left to the rule that needs it.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "h")
        if f is None:
            return None
        h = f["h"]
        if not (0 < h < Fraction(1, 2)):
            return None
        A = Fraction(1, 2) + h
        D = Fraction(1, 2) - h
        if A + D != 1:                      # the identity the whole scaling rests on
            return None
        return Content.abstract(f"A={sf(A)},D={sf(D)},h={sf(h)}",
                                derivation="A=1/2+h, D=1/2-h")

    return PythonRule("ns_exponents", fn, ABSTRACT, ABSTRACT,
                      description="h -> the similarity exponents A, D of (4.1)",
                      source="h=H->A,D")


@ns_rule("ns_core_scales")
def make_ns_core_scales() -> PythonRule:
    """`A=..,D=..,h=..` -> `vol=..,swirl=..,radial=..,h=..`.

    Exponents of tau for the core of Section 3.1: radial width tau^(1/2), axial
    height tau^D, so the volume exponent is 1 + D = 3/2 - h. The swirl and
    axial speeds go as tau^-A and the radial speed as tau^(-1/2), which is the
    asymmetry that makes the radial Reynolds number stay bounded while the
    angular one diverges.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "A", "D", "h")
        if f is None or f["A"] + f["D"] != 1:
            return None
        vol = 2 * Fraction(1, 2) + f["D"]           # ell_r^2 * ell_z
        swirl = -f["A"]
        radial = -Fraction(1, 2)
        return Content.abstract(
            f"vol={sf(vol)},swirl={sf(swirl)},radial={sf(radial)},h={sf(f['h'])}",
            derivation="core volume and velocity exponents")

    return PythonRule("ns_core_scales", fn, ABSTRACT, ABSTRACT,
                      description="A,D -> core volume and velocity exponents in tau",
                      source="A,D->vol,swirl,radial")


@ns_rule("ns_core_energy")
def make_ns_core_energy() -> PythonRule:
    """`vol=..,swirl=..,radial=..,h=..` -> `Ecore=..,Dcore=..,swirl=..,h=..`.

    The two numbers of Section 3.5. Kinetic energy is volume times squared
    velocity, so its exponent is vol + 2*swirl = 1/2 - 3h. The integrated
    squared radial derivative divides each velocity by the radial length, so
    its exponent is vol + 2*(swirl - 1/2) = -1/2 - 3h.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "vol", "swirl", "radial", "h")
        if f is None:
            return None
        e = f["vol"] + 2 * f["swirl"]
        d = f["vol"] + 2 * (f["swirl"] - Fraction(1, 2))
        return Content.abstract(
            f"Ecore={sf(e)},Dcore={sf(d)},swirl={sf(f['swirl'])},h={sf(f['h'])}",
            derivation="E=vol+2*swirl, D=vol+2*(swirl-1/2)")

    return PythonRule("ns_core_energy", fn, ABSTRACT, ABSTRACT,
                      description="core exponents -> energy and dissipation exponents",
                      source="vol,swirl->Ecore,Dcore")


@ns_rule("energy_budget_3_5")
def make_energy_budget_3_5() -> PythonRule:
    """`Ecore=..,Dcore=..,swirl=..,h=..` -> the verdict of Section 3.5.

    Three separate conditions, and the rule names which one fails:

      Ecore > 0    the kinetic energy of the core vanishes as tau -> 0;
      Dcore > -1   the dissipation integral converges at the singular time;
      swirl < 0    the speed on the growth path actually diverges.

    With Ecore = 1/2-3h and Dcore = -1/2-3h, the first two are both h < 1/6,
    which is the inequality the paper cites; the third is A > 0.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "Ecore", "Dcore", "swirl", "h")
        if f is None:
            return None
        if f["Ecore"] <= 0:
            verdict = "energy_does_not_vanish"
        elif f["Dcore"] <= -1:
            verdict = "dissipation_not_integrable"
        elif f["swirl"] >= 0:
            verdict = "velocity_stays_bounded"
        else:
            verdict = "bounded_energy_unbounded_velocity"
        return Content.abstract(f"{verdict},h={sf(f['h'])}",
                                derivation="sign tests on the tau exponents")

    return PythonRule("energy_budget_3_5", fn, ABSTRACT, ABSTRACT,
                      description="energy/dissipation/velocity exponents -> verdict",
                      source="Ecore,Dcore,swirl->verdict",
                      exact=False, trusted=False)


@ns_rule("ns_reynolds")
def make_ns_reynolds() -> PythonRule:
    """`h=1/200` -> the two Reynolds exponents of Section 2.1.

    At viscosity one and radial length tau^(1/2), the angular Reynolds number
    is tau^(-1/2-h) * tau^(1/2) = tau^-h and diverges, while the radial one is
    tau^(-1/2) * tau^(1/2) = O(1) and keeps viscosity in the leading balance.
    Both facts are used: the first is why the pulses can grow, the second is
    why the core does not simply diffuse.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "h")
        if f is None or f["h"] <= 0:
            return None
        theta = -f["h"]
        radial = Fraction(0)
        state = ("swirl_reynolds_diverges_radial_bounded"
                 if theta < 0 and radial == 0 else "reynolds_balance_fails")
        return Content.abstract(f"{state},Re_theta={sf(theta)},Re_r={sf(radial)}",
                                derivation="Re = |u| ell_r / nu at nu=1")

    return PythonRule("ns_reynolds", fn, ABSTRACT, ABSTRACT,
                      description="h -> angular and radial Reynolds exponents",
                      source="h=H->Re_theta,Re_r")


# ---------------------------------------------------------------------------
# The oscillations: Section 3.3, 6.1 and 7.2
# ---------------------------------------------------------------------------
@ns_rule("ns_wave_balance")
def make_ns_wave_balance() -> PythonRule:
    """`h=1/200` -> whether the pulse amplitude carries the required stress.

    Section 3.3 fixes A_wave = q^(-1/2-h/2) and ell_wave = q^(1/2+h/2), and
    then needs three separate identities to hold at once:

      2*A_wave  = swirl - 1/2      the squared amplitude is the stress scale;
      2*A_wave - 1/2 = -3/2 - h    its radial divergence matches the leading
                                   time derivative, transport and viscosity;
      A_wave - swirl = ell_wave - 1/2   both ratios are the same power q^(h/2).

    All three are exponent identities over the rationals, so this rule decides
    them rather than estimating them.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "h")
        if f is None or f["h"] <= 0:
            return None
        h = f["h"]
        swirl = -(Fraction(1, 2) + h)
        amp = -Fraction(1, 2) - h / 2
        wave = Fraction(1, 2) + h / 2
        ok = (2 * amp == swirl - Fraction(1, 2)
              and 2 * amp - Fraction(1, 2) == -Fraction(3, 2) - h
              and amp - swirl == wave - Fraction(1, 2))
        verdict = "wave_flux_matches_stress" if ok else "wave_flux_mismatch"
        return Content.abstract(f"{verdict},Awave={sf(amp)},ellwave={sf(wave)}",
                                derivation="pulse amplitude vs annular stress")

    return PythonRule("ns_wave_balance", fn, ABSTRACT, ABSTRACT,
                      description="h -> pulse amplitude, wavelength and the stress match",
                      source="h=H->Awave,ellwave")


@ns_rule("ns_band_cancel")
def make_ns_band_cancel() -> PythonRule:
    """`h=1/200` -> whether the band scale cancels out of the summed stress.

    Equation (7.30) sums the local covariances over dyadic bands and needs the
    band scale Q to disappear, leaving the physical stress q^(-A-1/2) T_0. The
    exponent of Q is -2A + h + (A + 1/2), and it is zero exactly because
    A = 1/2 + h. If it were not, the leading stress would depend on which band
    supplied it.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "h")
        if f is None or f["h"] <= 0:
            return None
        A = Fraction(1, 2) + f["h"]
        power = -2 * A + f["h"] + (A + Fraction(1, 2))
        verdict = "Q_powers_cancel" if power == 0 else "band_scale_leaks"
        return Content.abstract(f"{verdict},Qpower={sf(power)}",
                                derivation="exponent of Q in (7.30)")

    return PythonRule("ns_band_cancel", fn, ABSTRACT, ABSTRACT,
                      description="h -> does the dyadic band scale cancel in (7.30)",
                      source="h=H->Qpower")


@prover("carrier_balance_by_factorisation",
        "proves the viscous coefficient stays in [1, 4] for EVERY eps in "
        "(0, 1], by splitting on the carrier frequency and factorising the "
        "resulting quadratic")
def _prove_carrier_balance(library, rule) -> Judgement:
    """Section 7.2's balance, for every eps rather than the ones sampled.

    The rule computes k = ceil(eps^-1/2) and reports whether eps*k^2 lands
    in [1, 4]. That is a pointwise computation, and the claim behind it is
    uniform: the viscosity must neither drop out of the amplitude equation
    nor swamp the shear amplification, at ANY eps the construction picks.
    A continuum is not something instances reach, and it does not have to
    be, because the argument is three lines.

    The rule enforces the two conditions that define the ceiling before it
    answers, so any k it emits satisfies

        eps * k^2 >= 1        and        eps * (k-1)^2 < 1  (for k >= 2).

    The lower bound is the first of those, immediately. For the upper
    bound, the second gives eps < 1/(k-1)^2, so eps*k^2 < k^2/(k-1)^2, and

        4 (k-1)^2 - k^2 = (3k - 2)(k - 2),

    which is an identity, checked here by expansion rather than asserted.
    Both factors are non-negative linear forms in k for k >= 2, which
    linear arithmetic decides, so k^2/(k-1)^2 <= 4 for every such k. That
    leaves k = 1, where the rule's own condition forces eps >= 1 and the
    domain forces eps <= 1, so eps = 1 and the coefficient is exactly 1.

    Nothing in that is sampled and no eps is chosen anywhere in it.
    """
    if rule.name != "ns_carrier_frequency":
        return Judgement(False, obstruction=(
            f"{rule.name} is not the carrier frequency rule"))

    # The factorisation, expanded rather than quoted.
    k = Poly.sym("k")
    left = Poly.constant(4) * (k - 1) * (k - 1) - k * k
    right = (Poly.constant(3) * k - 2) * (k - 2)
    if not (left - right).is_zero:
        return Judgement(False, obstruction=(
            f"4(k-1)^2 - k^2 does not expand to (3k-2)(k-2); the difference "
            f"is {left - right}"))

    # Both factors non-negative for every integer k >= 2.
    with obligations() as obs:
        short(Lin.var("m") * 3 + 4, Lin(Fraction(0)), "3k - 2 at k = m + 2")
        short(Lin.var("m"), Lin(Fraction(0)), "k - 2 at k = m + 2")
    region = solve(obs)
    if not region.feasible:
        return Judgement(False, obstruction=(
            "a factor changes sign on k >= 2: " + "; ".join(region.blocking[:2])))

    # k = 1 is the one case the algebra does not cover, and the rule's own
    # condition pins it: eps * 1 >= 1 with eps <= 1 leaves eps = 1.
    at_one = rule.apply(Content.abstract("eps=1"))
    if at_one is None or "k=1" not in at_one.text or "epsk2=1" not in at_one.text:
        return Judgement(False, obstruction=(
            f"at eps = 1 the rule says {at_one.text if at_one else 'nothing'}, "
            f"and the argument needs k = 1 with coefficient exactly 1"))

    # The algebra bounds the COEFFICIENT. It says nothing about the window
    # the rule compares it against, and a rule that had narrowed that
    # window would still be sitting on a true theorem while giving the
    # wrong verdict. The coefficient is 4*eps on the whole of [1/4, 1), so
    # its achievable values fill [1, 4); probing the rule at both ends of
    # that and in between confirms its window covers what the theorem
    # allows, which is the other half of deciding the rule.
    span = [Fraction(1), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4),
            Fraction(9, 10), Fraction(10 ** 6 - 1, 10 ** 6)]
    for eps in span:
        got = rule.apply(Content.abstract(f"eps={sf(eps)}"))
        if got is None:
            return Judgement(False, obstruction=(
                f"the rule declines eps = {sf(eps)}, which is in its domain"))
        coefficient = Fraction(got.text.split("epsk2=")[1])
        if not (1 <= coefficient <= 4):
            return Judgement(False, obstruction=(
                f"at eps = {sf(eps)} the coefficient is {sf(coefficient)}, "
                f"outside the interval the argument proves"))
        if not got.text.startswith("viscosity_stays_in_pulse_equation"):
            return Judgement(False, obstruction=(
                f"at eps = {sf(eps)} the coefficient is {sf(coefficient)}, "
                f"which the argument allows, and the rule rejects it -- its "
                f"acceptance window is narrower than the theorem"))

    return Judgement(
        True,
        statement=("for every eps in (0, 1] the carrier frequency "
                   "k = ceil(eps^-1/2) gives 1 <= eps*k^2 <= 4, so the "
                   "viscous coefficient of Section 7.2 neither vanishes nor "
                   "swamps the amplitude equation at any eps"),
        covers=("every rational eps in (0, 1], which is the rule's whole "
                "domain, and every carrier frequency it can produce"),
        whole_domain=True,
        detail=[
            "lower bound: eps*k^2 >= 1 is one of the two conditions the "
            "rule enforces before answering",
            "upper bound: 4(k-1)^2 - k^2 = (3k-2)(k-2), expanded here as a "
            "polynomial identity, with both factors non-negative for k >= 2",
            "k = 1 is the remaining case and pins eps = 1 exactly",
            "the coefficient is 4*eps across [1/4, 1), so its achievable "
            "values fill [1, 4) and the rule's window is checked against "
            "both ends of that rather than against the midpoint",
            "a consequence worth stating: the rule's viscous_balance_lost "
            "branch is now known to be unreachable, so no number of "
            "instances could ever have exercised it",
        ])


@ns_rule("ns_carrier_frequency")
def make_ns_carrier_frequency() -> PythonRule:
    """`eps=1/64` -> whether viscosity stays in the leading pulse equation.

    The carrier frequency is k = ceil(eps^(-1/2)), and Section 7.2 needs the
    viscous coefficient eps*k^2 to stay of order one -- neither dropping out of
    the amplitude equation nor swamping the shear amplification. The paper's
    bound is 1 <= eps*k^2 <= 4 for eps <= 1, and that is decidable exactly:
    k is an integer, so eps*k^2 is a rational.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "eps")
        if f is None or not (0 < f["eps"] <= 1):
            return None
        eps = f["eps"]
        k = math.ceil(math.sqrt(1.0 / float(eps)))
        # ceil of an irrational square root, then confirmed exactly:
        while Fraction(k * k) * eps < 1:
            k += 1
        while k > 1 and Fraction((k - 1) ** 2) * eps >= 1:
            k -= 1
        # The two conditions that DEFINE the ceiling, enforced rather than
        # left to the loop above. Stating them here is what lets the prover
        # argue from them: it reasons about any k with these properties,
        # and every k this rule emits has them by the guard, so there is no
        # step where a claim about the loop has to be taken on trust.
        if not (Fraction(k * k) * eps >= 1
                and (k == 1 or Fraction((k - 1) ** 2) * eps < 1)):
            return None
        damping = eps * k * k
        verdict = ("viscosity_stays_in_pulse_equation"
                   if 1 <= damping <= 4 else "viscous_balance_lost")
        return Content.abstract(f"{verdict},k={k},epsk2={sf(damping)}",
                                derivation="k=ceil(eps^-1/2), eps k^2 in [1,4]")

    return PythonRule("ns_carrier_frequency", fn, ABSTRACT, ABSTRACT,
                      description="eps -> carrier frequency and the viscous coefficient",
                      source="eps=E->k,epsk2")


# ---------------------------------------------------------------------------
# The admissible stress cone: Lemma 4.5
# ---------------------------------------------------------------------------
def _cone_coordinates(a: Fraction, bs: Fraction, p1: Fraction, p2: Fraction):
    """(4.20): the shear direction and the stress components along it."""
    ts = -bs / a
    vs = a * (1 + ts * ts)
    Pc = p1 + ts * p2
    Jc = p2 - ts * p1
    return ts, vs, Pc, Jc


@ns_rule("lemma_4_5_cone")
def make_lemma_4_5_cone() -> PythonRule:
    """`cone(a=..,bs=..,p1=..,p2=..)` -> `admissible` or `not_admissible`.

    The quadratic form (4.22) of the admissible stress cone: with a > 0,

        v_s > 2,   P_c > v_s,   (v_s - 2) J_c^2 < 2 (P_c - v_s)^2.

    Every quantity is a rational function of the inputs, so over rational data
    this rule DECIDES the condition; there is no threshold and no rounding.
    That is what makes it worth checking the paper's other formulation against.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text.replace("cone(", "").replace(")", ""),
                    "a", "bs", "p1", "p2")
        if f is None or f["a"] <= 0:
            return None
        ts, vs, Pc, Jc = _cone_coordinates(f["a"], f["bs"], f["p1"], f["p2"])
        ok = (vs > 2 and Pc > vs and (vs - 2) * Jc * Jc < 2 * (Pc - vs) ** 2)
        return Content.abstract("admissible" if ok else "not_admissible",
                                derivation="quadratic cone test (4.22)")

    return PythonRule("lemma_4_5_cone", fn, ABSTRACT, ABSTRACT,
                      description="decide the admissible stress cone by (4.22)",
                      source="cone(a,bs,p1,p2)->admissible|not_admissible",
                      exact=False, trusted=False)


@ns_rule("lemma_4_5_relaxed")
def make_lemma_4_5_relaxed() -> PythonRule:
    """The relaxed cone condition (4.21): P_c > 2 and v_s < U(P_c, J_c).

    This is the weaker condition the profile construction carries while the
    axis is being joined to the exterior; the admissible one is imposed later,
    where the viscous waves need it.

    `exact=False`, and the reason is worth stating. U contains a square root,
    so this test is evaluated in floating point and can misclassify data close
    enough to the boundary. It is still TRUSTED: the machine may use it, and
    the flag records that it can be wrong. That is the opposite pairing from
    the caption copier, which is exact and untrusted.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text.replace("cone(", "").replace(")", ""),
                    "a", "bs", "p1", "p2")
        if f is None or f["a"] <= 0:
            return None
        _, vs, Pc, Jc = _cone_coordinates(f["a"], f["bs"], f["p1"], f["p2"])
        if Pc <= 2:
            return Content.abstract("outside_relaxed_cone")
        pc, jc = float(Pc), float(Jc)
        U = pc + jc * jc / 4 - abs(jc) * math.sqrt((pc - 2) / 2 + jc * jc / 16)
        ok = float(vs) < U
        return Content.abstract("relaxed_cone" if ok else "outside_relaxed_cone",
                                derivation="relaxed cone test (4.21)")

    return PythonRule("lemma_4_5_relaxed", fn, ABSTRACT, ABSTRACT,
                      description="decide the relaxed cone (4.21) in floating point",
                      source="cone(a,bs,p1,p2)->relaxed_cone|outside_relaxed_cone",
                      exact=False, trusted=True)


# ---------------------------------------------------------------------------
# Pulse growth and decay: Lemma 7.4 and (7.22)
# ---------------------------------------------------------------------------
def _pulse_rates(lam0: float, ustar: float, damp: float, slot: float, v: float):
    """The two competing rates at pulse time v, in the frame of (7.10).

    The shear amplifies at lambda(v) = lambda0 / sqrt(1+s^2) while viscosity
    damps at d_ref(v) = damp * lambda0 (1+s^2) / (1+ustar^2)^{3/2}, where the
    phase parameter s runs from ustar/2 to 3*ustar/2 across the slot. The two
    are equal at s = ustar when damp = 1, which is what makes the envelope
    turn over in the middle of the slot rather than at its ends.
    """
    s = ustar / 2 + ustar * v / slot
    lam = lam0 / math.sqrt(1 + s * s)
    d = damp * lam0 * (1 + s * s) / (1 + ustar * ustar) ** 1.5
    return lam, d


@ns_rule("lemma_7_4_pulse")
def make_lemma_7_4_pulse() -> PythonRule:
    """`pulse(lambda0=..,ustar=..,damp=..,slot=..)` -> the envelope's shape.

    The claim is Lemma 7.4 in its cheapest form: you can tell whether a pulse
    grows and then decays by looking at the NET rate at the two ends of its
    slot, without solving anything. Amplification exceeds damping at the start
    and damping exceeds amplification at the end, so the amplitude turns over
    once in between and the temporal cutoff acts only where it is already
    exponentially small.

    That is a claim about the solution of a differential equation made from
    two evaluations of its coefficients, so it is declared unverified and is
    checked against `pulse_by_integration`, which actually integrates the
    system. The damping multiplier is what makes the check informative: away
    from one it produces pulses that only grow or only decay, and the rule has
    to get those right too.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text.replace("pulse(", "").replace(")", ""),
                    "lambda0", "ustar", "damp", "slot")
        if f is None:
            return None
        lam0, ustar = float(f["lambda0"]), float(f["ustar"])
        damp, slot = float(f["damp"]), float(f["slot"])
        if min(lam0, ustar, damp, slot) <= 0:
            return None
        start = (lambda p: p[0] - p[1])(_pulse_rates(lam0, ustar, damp, slot, 0.0))
        end = (lambda p: p[0] - p[1])(_pulse_rates(lam0, ustar, damp, slot, slot))
        if start > 0 and end < 0:
            verdict = "grows_then_decays"
        elif start <= 0:
            verdict = "decays_throughout"
        else:
            verdict = "grows_throughout"
        return Content.abstract(verdict, derivation="net rate at both slot ends")

    return PythonRule("lemma_7_4_pulse", fn, ABSTRACT, ABSTRACT,
                      description="predict the pulse envelope from its end rates",
                      source="pulse(lambda0,ustar,damp,slot)->envelope",
                      exact=False, trusted=False)


# ---------------------------------------------------------------------------
# The exact exterior: Lemma A.6
# ---------------------------------------------------------------------------
def _heat_factor(Z: float, h: float, nodes: int = 6001) -> float:
    """H(Z) of (A.32), quadrature done on a substitution that removes the
    fractional power at the origin.

    H(Z) = (1/Gamma(1+h)) * integral e^-v v^h (1+Zv)^-h dv. Gauss-Laguerre is
    the obvious choice and the wrong one: its nodes are built for a
    polynomial times e^-v, and v^h with h not an integer is not smooth at
    zero, so the error shows up in the fifth digit and the heat identity
    fails a tolerance it should pass. Substituting v = w^p with p = 1/(1+h)
    makes v^h dv = p dw exactly, leaving a smooth integrand for Simpson.
    """
    p = 1.0 / (1.0 + h)
    upper = 60.0 ** (1.0 / p)
    w = np.linspace(0.0, upper, nodes)
    v = w ** p
    integrand = np.exp(-v) * (1.0 + Z * v) ** (-h) * p
    # composite Simpson on an odd number of samples
    step = upper / (nodes - 1)
    weights = np.ones(nodes)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    val = float(np.sum(weights * integrand) * step / 3.0)
    return val / math.gamma(1.0 + h)


def _swirl_field(r: float, tau: float, h: float, A: float) -> float:
    """K(r,tau) = s^-A H(2 tau / s), s = r^2/2, as in (4.29) and (A.33).

    The exponent A is passed separately from h on purpose. The field solves
    the radial swirl heat equation only when A = 1/2 + h, and giving the two
    independently is what lets an oracle discover that rather than be told it.
    """
    s = r * r / 2.0
    return s ** (-A) * _heat_factor(2.0 * tau / s, h)


@ns_rule("lemma_A6_heat")
def make_lemma_A6_heat() -> PythonRule:
    """`heat(h=..,A=..,r=..,tau=..)` -> does that field solve the exterior?

    Beyond the annulus the flow is purely azimuthal and the construction needs
    its residual to be exactly zero, which happens when the swirl solves

        -d_tau K = d_rr K + r^-1 d_r K - r^-2 K.

    Lemma A.6 says the similarity field s^-A H(2 tau / s) does that precisely
    when A = 1/2 + h, because that is when the hypergeometric identity (A.37)
    closes. The rule makes the cheap test -- compare two rationals -- and
    `heat_by_finite_differences` checks it by differencing the actual field.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text.replace("heat(", "").replace(")", ""),
                    "h", "A", "r", "tau")
        if f is None or f["h"] <= 0 or f["r"] <= 0 or f["tau"] < 0:
            return None
        ok = f["A"] == Fraction(1, 2) + f["h"]
        return Content.abstract(
            "solves_radial_swirl_heat" if ok else "leaves_exterior_residual",
            derivation="A = 1/2 + h closes the identity (A.37)")

    return PythonRule("lemma_A6_heat", fn, ABSTRACT, ABSTRACT,
                      description="decide the exact heat exterior from its exponent",
                      source="heat(h,A,r,tau)->solves|leaves_residual",
                      exact=False, trusted=False)


# ---------------------------------------------------------------------------
# The correction cycle: Proposition 9.6
# ---------------------------------------------------------------------------
@ns_rule("ns_cycle_step")
def make_ns_cycle_step() -> PythonRule:
    """`sigma=1/5` -> `sigma=3/10`. One turn of the correction cycle.

    Each cycle cancels the supported harmonics, corrects the averaged stress,
    removes the auxiliary mean and restores the three compatibility defects,
    and the net effect on the residual exponent is sigma -> sigma + 1/10.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "sigma")
        if f is None:
            return None
        return Content.abstract(f"sigma={sf(f['sigma'] + Fraction(1, 10))}",
                                derivation="one correction cycle")

    return PythonRule("ns_cycle_step", fn, ABSTRACT, ABSTRACT,
                      description="one correction cycle: sigma -> sigma + 1/10",
                      source="sigma=S->sigma")


@ns_rule("prop_9_6_decay")
def make_prop_9_6_decay() -> PythonRule:
    """`stage=25` -> `sigma=27/10`. The closed form sigma_j = 1/5 + j/10.

    Iterating `ns_cycle_step` from sigma_0 = 1/5 must give the same value, and
    the two are checked against each other rather than assumed equal: that is
    the whole content of the claim sigma_j -> infinity, which is what makes the
    residual flat at the singular time.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(r"\s*stage=(\d+)\s*", c.text.replace(" ", ""))
        if not m:
            return None
        j = int(m.group(1))
        return Content.abstract(f"sigma={sf(Fraction(1, 5) + Fraction(j, 10))}",
                                derivation="sigma_j = 1/5 + j/10")

    return PythonRule("prop_9_6_decay", fn, ABSTRACT, ABSTRACT,
                      description="closed form of the stage decay parameter (9.8)",
                      source="stage=J->sigma", exact=False, trusted=False)


@ns_rule("ns_budget")
def make_ns_budget() -> PythonRule:
    """`kappa=1/100000,B=7/10` -> whether a cycle really gains 1/10.

    The radial derivative loss kappa_s is charged against every gain in
    Proposition 9.6, and the cycle closes only if four groups of inequalities
    survive it:

        min{1/2-3k, 1/2-k, B-k, 2/5}        >= 2/5
        min{1/2-4k, 2/5-k, 1/2-2k, B-3k}    >= 2/5 - k
        1/2 - 2k > 1/10   and   17/100 > 1/10
        9/10 - 4k > 1/10

    With kappa_s = 10^-5 and B >= 7/10 they all hold, and the margins are the
    reason kappa_s was taken that small. Exact rational comparisons.
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "kappa", "B")
        if f is None or f["kappa"] < 0:
            return None
        k, B = f["kappa"], f["B"]
        half, two5 = Fraction(1, 2), Fraction(2, 5)
        first = min(half - 3 * k, half - k, B - k, two5) >= two5
        second = min(half - 4 * k, two5 - k, half - 2 * k, B - 3 * k) >= two5 - k
        third = (half - 2 * k > Fraction(1, 10)
                 and min(Fraction(17, 100), 1 - 4 * k) > Fraction(1, 10))
        fourth = Fraction(9, 10) - 4 * k > Fraction(1, 10)
        ok = first and second and third and fourth
        return Content.abstract(
            "cycle_gain_at_least_1/10" if ok else "cycle_budget_fails",
            derivation="the four inequality groups closing Proposition 9.6")

    return PythonRule("ns_budget", fn, ABSTRACT, ABSTRACT,
                      description="kappa_s and B -> does a correction cycle gain 1/10",
                      source="kappa=K,B=Bv->cycle_gain")


# ---------------------------------------------------------------------------
# Finite-dimensional moment corrections: Lemma A.1
# ---------------------------------------------------------------------------
@ns_rule("lemma_A1_moments")
def make_lemma_A1_moments() -> PythonRule:
    """`exponents=1/2;-1/2;-3/2` -> whether the moment matrix is invertible.

    Lemma A.1: bumps with ordered disjoint supports and DISTINCT power weights
    give an invertible moment matrix, because a nonzero combination of m
    distinct powers has at most m-1 positive zeros. The hypothesis the
    construction has to keep checking is therefore just distinctness, and this
    rule checks exactly that. The determinant itself is the oracle's job.
    """

    def fn(c: Content) -> Content | None:
        m = re.fullmatch(rf"\s*exponents=((?:{_NUM})(?:;{_NUM})*)\s*",
                         c.text.replace(" ", ""))
        if not m:
            return None
        vals = [fr(v) for v in m.group(1).split(";")]
        if len(vals) < 2:
            return None
        ok = len(set(vals)) == len(vals)
        return Content.abstract(
            "moment_matrix_invertible" if ok else "moment_matrix_singular",
            derivation="distinct powers (Lemma A.1)")

    return PythonRule("lemma_A1_moments", fn, ABSTRACT, ABSTRACT,
                      description="distinct radial weights -> invertible moment matrix",
                      source="exponents=E->invertible|singular",
                      exact=False, trusted=False)


# ---------------------------------------------------------------------------
# Viscosity: (10.22)-(10.23)
# ---------------------------------------------------------------------------
@ns_rule("ns_viscosity_rescale")
def make_ns_viscosity_rescale() -> PythonRule:
    """`nu=3` -> whether the rescaling carries the solution to that viscosity.

    Under u_nu(x,t) = sqrt(nu) u(x/sqrt(nu), t), p_nu = nu p, the four terms of
    the momentum equation must all pick up the same power of nu, or the
    rescaled fields solve a different equation. In exponents, with u ~ nu^(1/2),
    x ~ nu^(1/2) and hence grad ~ nu^(-1/2):

        d_t u      1/2
        (u.grad)u  1/2 + 1/2 - 1/2
        nu lap u   1 + 1/2 - 1
        grad p     1 - 1/2

    All four are 1/2, so the force scales by sqrt(nu) and the singular time is
    unchanged. The energy picks up nu^(5/2) from nu * nu^(3/2).
    """

    def fn(c: Content) -> Content | None:
        f = _fields(c.text, "nu")
        if f is None or f["nu"] <= 0:
            return None
        u, x = Fraction(1, 2), Fraction(1, 2)
        grad = -x
        terms = [u, u + u + grad, 1 + u + 2 * grad, 2 * u - x]
        if len(set(terms)) != 1:
            return Content.abstract("rescaling_breaks_the_equation")
        energy = 2 * u + 3 * x
        return Content.abstract(
            f"force_scales_by_nu^{sf(terms[0])},energy_scales_by_nu^{sf(energy)},"
            f"singular_time_unchanged",
            derivation="term-by-term exponents of (10.22)")

    return PythonRule("ns_viscosity_rescale", fn, ABSTRACT, ABSTRACT,
                      description="nu -> the exponents of the viscosity rescaling",
                      source="nu=N->force,energy,time")


# ---------------------------------------------------------------------------
# The estimates, as rules the machine can apply and cannot justify
# ---------------------------------------------------------------------------
# Every result of the paper is a rule here, named for the result. The ones
# above decide instances and are checked against an oracle that computes the
# answer another way. The ones below are bounds on function spaces: a rule can
# state the step, and nothing this machine has can establish it. They are
# therefore `exact=False` -- an imported step can be wrong -- and untrusted,
# which is what makes proof search refuse to reach Theorem 1.1 with them.
#
# Their confidence is the Laplace prior for a rule nobody has checked, 1/2. No
# measurement stands behind it and none is claimed. Chained, the thirteen-step
# derivation comes out near 2^-9, and that number is the point: it is the
# machine reporting that the conclusion is exactly as good as the imports.


def _step(name: str, premise: str, conclusion: str, description: str,
          needs_small_h: bool = False) -> PythonRule:
    """One named result, as a rule from its hypothesis cell to its conclusion."""

    def fn(c: Content) -> Content | None:
        h = _tagged(c.text, premise)
        if h is None:
            return None
        if needs_small_h and not (0 < h < Fraction(1, 100)):
            return None
        return Content.abstract(f"{conclusion},h={sf(h)}",
                                derivation=f"imported: {description}")

    return PythonRule(name, fn, ABSTRACT, ABSTRACT, description=description,
                      source=f"{premise}->{conclusion}",
                      exact=False, trusted=False)


_IMPORTED = [
    ("thm_4_6_profiles", "bounded_energy_unbounded_velocity", "leading_profiles",
     "Theorem 4.6: profiles with a regular axis, an exact heat exterior, the "
     "four moment identities, and an annular stress inside the cone", True),
    ("prop_5_5_background", "leading_profiles", "background_annular_stress",
     "Proposition 5.5: the order-by-order corrections sum to a smooth "
     "divergence-free background whose residual is an annular stress "
     "divergence plus a flat remainder", False),
    ("prop_7_5_stress", "increment_split", "stress_realized_by_waves",
     "Proposition 7.5: two pulse families carry the stress with positive "
     "squared amplitudes", False),
    ("prop_9_9_summation", "residual_flat_at_singularity", "local_field_theorem_3_1",
     "Proposition 9.9: the corrections sum under shrinking cutoffs to the "
     "local field of Theorem 3.1, residual flat to every order", False),
    ("prop_10_1_localize", "local_field_theorem_3_1", "localized_fields",
     "Proposition 10.1: cutting the vector potential before taking the curl "
     "gives compact support and zero initial data, incompressibility intact",
     False),
    ("lemma_10_3_force", "localized_fields", "compact_smooth_force",
     "Lemmas 10.2 and 10.3: every derivative of the residual has a compatible "
     "limit at t=1, realized by a force in C_c^inf", False),
    ("lemma_10_4_energy", "compact_smooth_force", "uniform_energy_bound",
     "Lemma 10.4: the equation itself bounds the kinetic energy and the "
     "integrated dissipation up to the singular time", False),
    ("lemma_10_5_unique", "uniform_energy_bound", "no_global_smooth_competitor",
     "Lemma 10.5: any smooth bounded-energy solution with the same force and "
     "datum agrees with this one on every shorter interval", False),
]

for _name, _premise, _conclusion, _description, _small in _IMPORTED:
    def _factory(n=_name, p=_premise, c=_conclusion, d=_description, sm=_small):
        return _step(n, p, c, d, sm)
    ns_rule(_name)(_factory)


@ns_rule("thm_1_1_blowup")
def make_thm_1_1_blowup() -> PythonRule:
    """Theorem 1.1: the growth path lies where the cutoffs equal one, so the
    localized velocity is unbounded at t=1 while its energy stays bounded, and
    no global smooth solution with that force and datum can exist."""

    def fn(c: Content) -> Content | None:
        if _tagged(c.text, "no_global_smooth_competitor") is None:
            return None
        return Content.abstract("theorem_1_1_forced_blowup",
                                derivation="imported: Theorem 1.1")

    return PythonRule(
        "thm_1_1_blowup", fn, ABSTRACT, ABSTRACT,
        description="Theorem 1.1: forced finite-time blowup at bounded energy",
        source="no_global_smooth_competitor->theorem_1_1_forced_blowup",
        exact=False, trusted=False)


IMPORTED_NAMES = tuple(n for n, *_ in _IMPORTED) + ("thm_1_1_blowup",)


# ---------------------------------------------------------------------------
# Generators: the experiments the machine runs on itself
# ---------------------------------------------------------------------------
@generator("ns_cone_points",
           "Shear and integrated-stress data (a, b_s, p_s) for the cone "
           "tests. Half the points are generic and half are pushed against "
           "the cone boundary, because a check made only of easy points "
           "cannot see a wrong constant in the inequality.",
           {"margin": "how far the quadratic test must stay from equality",
            "stress": "fraction of points placed near the boundary"})
def ns_cone_points(n: int, rng: np.random.Generator, margin: float = 1e-3,
                   stress: float = 0.5):
    out: list[Example] = []
    want_near = int(n * stress)
    near = generic = 0
    tries = 0
    while len(out) < n and tries < 40_000 * n:
        tries += 1
        a = Fraction(int(rng.integers(1, 40)), 10)
        bs = Fraction(int(rng.integers(-30, 31)), 10)
        p1 = Fraction(int(rng.integers(-20, 80)), 10)
        p2 = Fraction(int(rng.integers(-40, 41)), 10)
        _, vs, Pc, Jc = _cone_coordinates(a, bs, p1, p2)
        gap = 2 * (Pc - vs) ** 2 - (vs - 2) * Jc * Jc
        smallest = min(abs(float(gap)), abs(float(vs - 2)), abs(float(Pc - vs)))
        # Both routes are numerical somewhere -- one takes a square root, the
        # other solves a 2x2 system -- so data exactly on a boundary would
        # measure rounding rather than the lemma. The margin is 1e-3 and the
        # covariance route was checked against the exact rational test down
        # to 1e-4, so this is inside where they are known to agree.
        if smallest < margin:
            continue
        # "Near" means the quadratic clause is within a factor of two of
        # deciding the point, which is the band a wrong constant would move.
        denominator = 2 * (Pc - vs) ** 2
        ratio = (float((vs - 2) * Jc * Jc / denominator)
                 if vs > 2 and denominator != 0 else None)
        is_near = ratio is not None and 0.5 <= ratio <= 2.0
        if is_near and near < want_near:
            near += 1
        elif not is_near and generic < n - want_near:
            generic += 1
        else:
            continue
        out.append(Example(inp=Content.abstract(
            f"cone(a={sf(a)},bs={sf(bs)},p1={sf(p1)},p2={sf(p2)})")))
    return out


@generator("ns_energy_cells",
           "Candidate energy, dissipation and velocity exponents for the "
           "Section 3.5 verdict. Sampled around the true values 1/2-3h and "
           "-1/2-3h so that both verdicts occur.",
           {"spread": "half-width of the sampling window, in tenths"})
def ns_energy_cells(n: int, rng: np.random.Generator, spread: int = 12):
    out: list[Example] = []
    margin = Fraction(3, 10)

    def away(x: Fraction, boundary: Fraction) -> bool:
        return abs(x - boundary) >= margin

    while len(out) < n:
        e = Fraction(int(rng.integers(-spread, spread + 1)), 10)
        d = Fraction(int(rng.integers(-spread, spread + 1)), 10) - Fraction(1, 2)
        s = Fraction(int(rng.integers(-spread, spread + 1)), 10)
        # A numeric route decides a limit by evaluating it somewhere, so it
        # cannot resolve an exponent sitting on the boundary. Keep the data
        # off all three, and say so rather than tuning a threshold until the
        # accuracy looks good.
        if not (away(e, Fraction(0)) and away(d, Fraction(-1))
                and away(s, Fraction(0))):
            continue
        h = Fraction(1, int(rng.integers(101, 1000)))
        out.append(Example(inp=Content.abstract(
            f"Ecore={sf(e)},Dcore={sf(d)},swirl={sf(s)},h={sf(h)}")))
    return out


@generator("ns_cycle_stages",
           "Correction-cycle stage indices, for the decay parameter sigma_j. "
           "The index range is wider than any run's instance count so that "
           "asking for more instances asks about more stages.",
           {"jmax": "largest stage index"})
def ns_cycle_stages(n: int, rng: np.random.Generator, jmax: int = 4000):
    # The old default of 60 capped the set at 61 questions however many
    # examples were drawn, so confidence past that point was counting
    # repeats. The recursion sigma_{j+1} = sigma_j + 1/10 is claimed for
    # every stage, and nothing about the check gets easier at large j: the
    # oracle iterates the recursion and the rule uses the closed form, so a
    # far stage is a longer iteration and a stronger test, not a weaker one.
    return [Example(inp=Content.abstract(f"stage={int(rng.integers(0, jmax + 1))}"))
            for _ in range(n)]


@generator("ns_moment_families",
           "Tuples of radial power weights for the finite moment corrections "
           "of Lemma A.1, with repeats deliberately included.",
           {"size": "how many weights per tuple",
            "reach": "largest weight, in halves",
            "collide": "fraction of tuples given a deliberate repeat"})
def ns_moment_families(n: int, rng: np.random.Generator, size: int = 3,
                       reach: int = 20, collide: float = 0.5):
    # Weights out to +-10 rather than +-3. Lemma A.1 asks for the moment
    # matrix of a family of radial powers to be invertible when the powers
    # are distinct, and a wider spread is a worse-conditioned matrix, so
    # the extra instances are harder rather than merely more numerous.
    out: list[Example] = []
    for _ in range(n):
        vals = [Fraction(int(rng.integers(-reach, reach + 1)), 2)
                for _ in range(size)]
        # Half the tuples are given a repeat on purpose. Widening the weight
        # range made accidental repeats rare, and repeats are the only cases
        # the "distinctness ignored" mutant gets wrong, so leaving the rate to
        # chance would have bought distinct instances by spending the contrast
        # that makes them worth having.
        if rng.random() < collide and size >= 2:
            vals[-1] = vals[0]
        out.append(Example(inp=Content.abstract(
            "exponents=" + ";".join(sf(v) for v in vals))))
    return out


@generator("ns_pulse_params",
           "Pulse parameters (growth rate, phase turn, damping multiplier, "
           "slot length) for the amplitude equation. The damping multiplier "
           "is swept across one so that all three envelope shapes occur.",
           {"damp_lo": "smallest damping multiplier, in tenths",
            "damp_hi": "largest damping multiplier, in tenths"})
def ns_pulse_params(n: int, rng: np.random.Generator,
                    damp_lo: int = 3, damp_hi: int = 25):
    out: list[Example] = []
    for _ in range(n):
        lam0 = Fraction(int(rng.integers(5, 40)), 10)
        ustar = Fraction(int(rng.integers(5, 30)), 10)
        damp = Fraction(int(rng.integers(damp_lo, damp_hi + 1)), 10)
        slot = Fraction(int(rng.integers(20, 120)), 1)
        out.append(Example(inp=Content.abstract(
            f"pulse(lambda0={sf(lam0)},ustar={sf(ustar)},"
            f"damp={sf(damp)},slot={sf(slot)})")))
    return out


@generator("ns_heat_points",
           "Exterior swirl fields s^-A H(2 tau/s): a decay exponent A, the "
           "profile parameter h, and a point (r, tau) to test them at. Half "
           "the exponents are the matching one, half are not.",
           {"mismatch": "fraction of exponents deliberately wrong"})
def ns_heat_points(n: int, rng: np.random.Generator, mismatch: float = 0.5):
    out: list[Example] = []
    for _ in range(n):
        # Theorem 4.6 fixes h in (0, 1/100); sampling outside it would test a
        # field the construction never uses, and would blame the quadrature
        # for the disagreement.
        h = Fraction(1, int(rng.integers(101, 2000)))
        A = Fraction(1, 2) + h
        if rng.random() < mismatch:
            # Across decades, not at one size. Differencing the field leaves a
            # noise floor near 4e-8 in relative residual and responds linearly
            # to the mismatch, so 1e-5 is the smallest miss it resolves with
            # room to spare; a generator that only ever missed by 1/20 would
            # never find out whether the rule needs the exponent exactly.
            size = int(rng.choice([100_000, 10_000, 1_000, 100, 20]))
            A += Fraction(int(rng.choice([-1, 1])), size)
        r = Fraction(int(rng.integers(10, 60)), 10)
        tau = Fraction(int(rng.integers(1, 40)), 100)
        out.append(Example(inp=Content.abstract(
            f"heat(h={sf(h)},A={sf(A)},r={sf(r)},tau={sf(tau)})")))
    return out


# ---------------------------------------------------------------------------
# Oracles: the second route
# ---------------------------------------------------------------------------
@oracle("cone_by_definition",
        "the admissible stress cone read straight off its definition (4.21), "
        "evaluated in floating point",
        "definitional", ("admissible", "not_admissible"))
def _cone_by_definition(ex: Example) -> Content | None:
    """The paper defines the cone through

        U(P_c, J_c) = P_c + J_c^2/4 - |J_c| sqrt((P_c-2)/2 + J_c^2/16),

    admissible meaning P_c > 2, v_s < U and v_s > 2. Lemma 4.5 proves that
    equivalent to the quadratic test, and this oracle is what makes that
    equivalence a measurement instead of a citation. It carries a square root,
    so it is the floating-point route; the generator keeps its data away from
    the boundary where the two could differ on rounding alone.
    """
    f = _fields(ex.inp.text.replace("cone(", "").replace(")", ""),
                "a", "bs", "p1", "p2")
    if f is None or f["a"] <= 0:
        return None
    _, vs, Pc, Jc = _cone_coordinates(f["a"], f["bs"], f["p1"], f["p2"])
    pc, jc, v = float(Pc), float(Jc), float(vs)
    if pc <= 2 or v <= 2:
        return Content.abstract("not_admissible")
    U = pc + jc * jc / 4 - abs(jc) * math.sqrt((pc - 2) / 2 + jc * jc / 16)
    return Content.abstract("admissible" if v < U else "not_admissible")


@oracle("energy_by_quadrature",
        "the Section 3.5 verdict decided by evaluating the fields numerically "
        "instead of comparing exponents",
        "measured")
def _energy_by_quadrature(ex: Example) -> Content | None:
    """A numeric route to the same verdict.

    The energy and velocity tests are evaluations at a small remaining time.
    The dissipation test is the integral of tau^Dcore over (0, 1], computed
    after the substitution tau = s^2, which turns it into 2 s^(2*Dcore+1) and
    removes the singularity for every exponent the test accepts. Being
    numerical it is evidence, not a decision: a Dcore just below -1 integrates
    to something large rather than to infinity, which is why the generator
    keeps its exponents on a grid of tenths.
    """
    f = _fields(ex.inp.text, "Ecore", "Dcore", "swirl", "h")
    if f is None:
        return None
    e, d, s = float(f["Ecore"]), float(f["Dcore"]), float(f["swirl"])
    tiny = 1e-40
    if tiny ** e >= 1e-6:                       # the energy has not vanished
        return Content.abstract(f"energy_does_not_vanish,h={sf(f['h'])}")

    def piece(lo: float, hi: float) -> float:
        """Integral of tau^d over [lo, hi], after tau = sigma^2."""
        grid = np.linspace(math.sqrt(lo), math.sqrt(hi), 200_001)
        return float(_trapezoid(2.0 * grid ** (2 * d + 1), grid))

    # Convergence is a statement about the tail, so test the tail: cut the
    # remaining interval into successive decades and ask whether their
    # contributions shrink. Comparing two truncated integrals does not do
    # this -- for an exponent just above -1 the difference between them is
    # still large, and the integral still converges.
    first = piece(1e-8, 1e-4)
    second = piece(1e-12, 1e-8)
    if not np.isfinite(second) or abs(second) > 0.5 * abs(first) + 1e-12:
        return Content.abstract(f"dissipation_not_integrable,h={sf(f['h'])}")
    if tiny ** s <= 1e6:                        # the speed has not diverged
        return Content.abstract(f"velocity_stays_bounded,h={sf(f['h'])}")
    return Content.abstract(f"bounded_energy_unbounded_velocity,h={sf(f['h'])}")


@oracle("sigma_by_iteration",
        "the stage decay parameter obtained by iterating the correction cycle "
        "rather than by its closed form",
        "definitional")
def _sigma_by_iteration(ex: Example) -> Content | None:
    m = re.fullmatch(r"\s*stage=(\d+)\s*", ex.inp.text.replace(" ", ""))
    if not m:
        return None
    sigma = Fraction(1, 5)
    for _ in range(int(m.group(1))):
        sigma += Fraction(1, 10)
    return Content.abstract(f"sigma={sf(sigma)}")


@oracle("moment_matrix_by_determinant",
        "invertibility of a moment matrix decided by building it from actual "
        "bumps and computing its determinant",
        "measured", ("moment_matrix_invertible", "moment_matrix_singular"))
def _moment_matrix_by_determinant(ex: Example) -> Content | None:
    """Lemma A.1 says distinct powers suffice. This builds the matrix

        B_ij = integral x^alpha_i beta_j(x) dx

    for bumps on ordered disjoint intervals and reports whether its
    determinant is nonzero. Numerical, so it is a check on instances rather
    than a proof of the lemma, and the tolerance is stated: 1e-9 relative to
    the product of the row norms.
    """
    m = re.fullmatch(rf"\s*exponents=((?:{_NUM})(?:;{_NUM})*)\s*",
                     ex.inp.text.replace(" ", ""))
    if not m:
        return None
    alphas = [float(fr(v)) for v in m.group(1).split(";")]
    k = len(alphas)
    if k < 2:
        return None
    centres = [1.0 + 0.7 * j for j in range(k)]
    grid = np.linspace(0.2, centres[-1] + 1.0, 20_001)
    bumps = [np.exp(-((grid - c) ** 2) / (2 * 0.08 ** 2)) for c in centres]
    B = np.array([[float(_trapezoid(grid ** a * b, grid)) for b in bumps]
                  for a in alphas])
    scale = float(np.prod([np.linalg.norm(row) for row in B]))
    ok = scale > 0 and abs(float(np.linalg.det(B))) > 1e-9 * scale
    return Content.abstract("moment_matrix_invertible" if ok
                            else "moment_matrix_singular")


@oracle("covariance_positive_weights",
        "the admissible cone decided by actually constructing the two wave "
        "families and solving for their squared amplitudes",
        "definitional", ("admissible", "not_admissible"))
def _covariance_positive_weights(ex: Example) -> Content | None:
    """Proposition 7.5 done rather than quoted.

    The cone condition exists for one reason: the stress has to be a
    NONNEGATIVE combination of the momentum fluxes two pulse families can
    carry. So this oracle builds those fluxes and solves for the weights.
    From the shear (a, -b_s) it forms the tangential frame of Section 7.1,

        g0 = (-a, b_s),  N = g0/|g0|,  K = N perp,
        lambda0^2 = -2 N_theta (2 N_theta + |g0|),  c0 = lambda0/(2 N_theta),

    picks the smallest u* on a grid that leaves the frozen-frame margin (7.1),
    forms the two covariance directions -A_c N -/+ u* K with
    A_c = -c0 sqrt(1+u*^2) > 0, and solves the 2x2 system for the squared
    amplitudes of the target stress T = p_s - (a, -b_s). Admissible means both
    weights come out strictly positive, which is what having real amplitudes
    means.

    Nothing in this computation evaluates (4.22), and nothing in it evaluates
    U(P_c, J_c) either. It is the mechanism, and the two algebraic tests are
    claims about it.
    """
    f = _fields(ex.inp.text.replace("cone(", "").replace(")", ""),
                "a", "bs", "p1", "p2")
    if f is None or f["a"] <= 0:
        return None
    a, bs = float(f["a"]), float(f["bs"])
    g0 = np.array([-a, bs])
    ng0 = float(np.linalg.norm(g0))
    if ng0 == 0:
        return None
    N = g0 / ng0
    K = np.array([-N[1], N[0]])
    disc = -2 * N[0] * (2 * N[0] + ng0)
    if disc <= 0:                       # lambda0 imaginary: no growing pulse
        return Content.abstract("not_admissible")
    lam0 = math.sqrt(disc)
    c0 = lam0 / (2 * N[0])              # negative, since N_theta < 0 for a > 0
    T = np.array([float(f["p1"]) - a, float(f["p2"]) + bs])
    TN, TK = float(T @ N), float(T @ K)
    if TN >= 0:
        return Content.abstract("not_admissible")
    for ustar in np.linspace(0.05, 60.0, 4000):
        Ac = -c0 * math.sqrt(1 + ustar * ustar)
        cols = np.stack([-Ac * N - ustar * K, -Ac * N + ustar * K], axis=1)
        if abs(float(np.linalg.det(cols))) < 1e-12:
            continue
        y = np.linalg.solve(cols, T)
        if y[0] > 0 and y[1] > 0:
            return Content.abstract("admissible")
    return Content.abstract("not_admissible")


@oracle("pulse_by_integration",
        "the pulse envelope obtained by integrating the amplitude equation "
        "across its slot",
        "definitional",
        ("grows_then_decays", "grows_throughout", "decays_throughout"))
def _pulse_by_integration(ex: Example) -> Content | None:
    """Lemma 7.4 on one pulse, by solving it.

    Integrates the two-dimensional system of (7.17) with the reference rates
    of (7.10), z' = (diag(lambda,-lambda) - d) z, from z = (1, 0) across the
    slot, and classifies the resulting amplitude: interior maximum with both
    ends below it, or monotone in one direction. Runge-Kutta with a fixed
    step, so it is evidence about this pulse and not a theorem about pulses.
    """
    f = _fields(ex.inp.text.replace("pulse(", "").replace(")", ""),
                "lambda0", "ustar", "damp", "slot")
    if f is None:
        return None
    lam0, ustar = float(f["lambda0"]), float(f["ustar"])
    damp, slot = float(f["damp"]), float(f["slot"])
    if min(lam0, ustar, damp, slot) <= 0:
        return None

    def rhs(v: float, z: np.ndarray) -> np.ndarray:
        lam, d = _pulse_rates(lam0, ustar, damp, slot, v)
        return np.array([(lam - d) * z[0], (-lam - d) * z[1]])

    steps = 4000
    dt = slot / steps
    z = np.array([1.0, 0.0])
    logs = [0.0]
    for i in range(steps):
        v = i * dt
        k1 = rhs(v, z)
        k2 = rhs(v + dt / 2, z + dt / 2 * k1)
        k3 = rhs(v + dt / 2, z + dt / 2 * k2)
        k4 = rhs(v + dt, z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        mag = float(np.linalg.norm(z))
        logs.append(math.log(mag) if mag > 0 else -math.inf)
    peak = int(np.argmax(logs))
    if 0 < peak < steps and logs[peak] > logs[0] and logs[peak] > logs[-1]:
        return Content.abstract("grows_then_decays")
    if logs[-1] > logs[0]:
        return Content.abstract("grows_throughout")
    return Content.abstract("decays_throughout")


@oracle("heat_by_finite_differences",
        "whether an exterior swirl field satisfies the radial heat equation, "
        "decided by differencing the field itself",
        "definitional",
        ("solves_radial_swirl_heat", "leaves_exterior_residual"))
def _heat_by_finite_differences(ex: Example) -> Content | None:
    """Lemma A.6 checked at a point, with no identity assumed.

    Evaluates K(r, tau) = s^-A H(2 tau/s) on a small stencil and forms the
    residual of -d_tau K = d_rr K + r^-1 d_r K - r^-2 K by central
    differences. The verdict compares that residual with the size of the
    individual terms, so a field that misses by a fixed exponent is caught
    while ordinary truncation error is not mistaken for one.
    """
    f = _fields(ex.inp.text.replace("heat(", "").replace(")", ""),
                "h", "A", "r", "tau")
    if f is None or f["r"] <= 0 or f["h"] <= 0:
        return None
    h, A = float(f["h"]), float(f["A"])
    r0, tau0 = float(f["r"]), float(f["tau"])

    def K(rr: float, tt: float) -> float:
        return _swirl_field(rr, tt, h, A)

    def relative_residual(r: float, tau: float) -> float:
        dr, dt = 1e-4 * r, 1e-4 * max(tau, 1e-3)
        k0 = K(r, tau)
        d_r = (K(r + dr, tau) - K(r - dr, tau)) / (2 * dr)
        d_rr = (K(r + dr, tau) - 2 * k0 + K(r - dr, tau)) / (dr * dr)
        d_tau = (K(r, tau + dt) - K(r, tau - dt)) / (2 * dt)
        residual = -d_tau - (d_rr + d_r / r - k0 / (r * r))
        scale = max(abs(d_tau), abs(d_rr), abs(d_r / r), abs(k0 / (r * r)))
        return abs(residual) / scale if scale > 0 else 0.0

    # One point is not enough. A field with the wrong exponent can still have
    # zero residual on a curve, and testing there would call it a solution.
    worst = max(relative_residual(r0 * f_r, tau0 * f_t)
                for f_r, f_t in ((1.0, 1.0), (1.7, 1.0), (1.0, 2.3), (0.6, 0.4)))
    return Content.abstract("solves_radial_swirl_heat" if worst <= 1e-6
                            else "leaves_exterior_residual")


# ---------------------------------------------------------------------------
# Mutations: what each check can and cannot see
# ---------------------------------------------------------------------------
# A rule agreeing with its oracle is worth nothing until you know the oracle
# could have disagreed. Each entry below is the same rule with one thing
# deliberately wrong. Running them against the same data measures the only
# quantity that matters here: how far off a rule can be before the check
# notices. A mutation that survives marks a claim the run does not establish,
# whatever the accuracy column says.


def _mutant_energy(c: Content) -> Content | None:
    """Dissipation is called integrable down to -3/2 instead of -1."""
    f = _fields(c.text, "Ecore", "Dcore", "swirl", "h")
    if f is None:
        return None
    if f["Ecore"] <= 0:
        v = "energy_does_not_vanish"
    elif f["Dcore"] <= Fraction(-3, 2):
        v = "dissipation_not_integrable"
    elif f["swirl"] >= 0:
        v = "velocity_stays_bounded"
    else:
        v = "bounded_energy_unbounded_velocity"
    return Content.abstract(f"{v},h={sf(f['h'])}")


def _mutant_cone(c: Content) -> Content | None:
    """The quadratic clause of (4.22) dropped, keeping only P_c > v_s > 2."""
    f = _fields(c.text.replace("cone(", "").replace(")", ""),
                "a", "bs", "p1", "p2")
    if f is None or f["a"] <= 0:
        return None
    _, vs, Pc, _ = _cone_coordinates(f["a"], f["bs"], f["p1"], f["p2"])
    return Content.abstract("admissible" if (vs > 2 and Pc > vs)
                            else "not_admissible")


def _mutant_pulse(c: Content) -> Content | None:
    """The closing rate read at the middle of the slot instead of its end."""
    f = _fields(c.text.replace("pulse(", "").replace(")", ""),
                "lambda0", "ustar", "damp", "slot")
    if f is None:
        return None
    lam0, ustar = float(f["lambda0"]), float(f["ustar"])
    damp, slot = float(f["damp"]), float(f["slot"])
    if min(lam0, ustar, damp, slot) <= 0:
        return None
    a0 = (lambda p: p[0] - p[1])(_pulse_rates(lam0, ustar, damp, slot, 0.0))
    a1 = (lambda p: p[0] - p[1])(_pulse_rates(lam0, ustar, damp, slot, slot / 2))
    if a0 > 0 and a1 < 0:
        return Content.abstract("grows_then_decays")
    return Content.abstract("decays_throughout" if a0 <= 0 else "grows_throughout")


def _mutant_heat(c: Content) -> Content | None:
    """The exterior exponent allowed to miss 1/2+h by up to 1/40."""
    f = _fields(c.text.replace("heat(", "").replace(")", ""),
                "h", "A", "r", "tau")
    if f is None:
        return None
    ok = abs(f["A"] - (Fraction(1, 2) + f["h"])) <= Fraction(1, 40)
    return Content.abstract("solves_radial_swirl_heat" if ok
                            else "leaves_exterior_residual")


def _mutant_moments(c: Content) -> Content | None:
    """Distinctness ignored: every family called invertible."""
    return Content.abstract("moment_matrix_invertible")


def _mutant_decay(c: Content) -> Content | None:
    """The cycle gain taken as 1/9 rather than 1/10."""
    m = re.fullmatch(r"\s*stage=(\d+)\s*", c.text.replace(" ", ""))
    if not m:
        return None
    j = int(m.group(1))
    return Content.abstract(f"sigma={sf(Fraction(1, 5) + Fraction(j, 9))}")


MUTATIONS: dict[str, tuple[str, Callable[[Content], "Content | None"]]] = {
    "energy_budget_3_5": ("dissipation threshold -1 -> -3/2", _mutant_energy),
    "lemma_4_5_cone": ("quadratic clause dropped", _mutant_cone),
    "lemma_7_4_pulse": ("closing rate read mid-slot", _mutant_pulse),
    "lemma_A6_heat": ("exponent allowed to miss by 1/40", _mutant_heat),
    "lemma_A1_moments": ("distinctness ignored", _mutant_moments),
    "prop_9_6_decay": ("cycle gain 1/10 -> 1/9", _mutant_decay),
}


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
def install_navier_stokes_rules(library: RuleLibrary) -> RuleLibrary:
    """Add the construction's rules to one library.

    Deliberately not registered in `prior.PRIOR_RULES`. That registry is
    what every new machine installs from, so putting these there would give a
    multiplication machine built later in the same process the admissible
    stress cone, and quietly change its objective and its benchmark.
    """
    for factory in NS_RULES.values():
        library.add(factory(), replace=True)
    return library


def register_for_reload() -> None:
    """Let `RuleLibrary.load` rebuild these rules from a saved manifest.

    A saved library records a PythonRule by name and rebuilds it from
    `PRIOR_RULES`, so a manifest written after `install_navier_stokes_rules`
    reloads without them unless this is called first. It is separate, and
    explicit, because it has the global effect the installer avoids.
    """
    for name, factory in NS_RULES.items():
        PRIOR_RULES.setdefault(name, factory)


def checked_rules() -> list[str]:
    """The rules that compute, as opposed to the ones that cite."""
    return [n for n in NS_RULES if n not in IMPORTED_NAMES]


#: The rules that state a claim and must earn their trust from an oracle.
CLAIM_RULES = {
    "energy_budget_3_5": "energy_by_quadrature",
    "lemma_4_5_cone": "covariance_positive_weights",
    "lemma_7_4_pulse": "pulse_by_integration",
    "lemma_A6_heat": "heat_by_finite_differences",
    "lemma_A1_moments": "moment_matrix_by_determinant",
    "prop_9_6_decay": "sigma_by_iteration",
}

#: Generator for each claim rule's data, with the parameters the run uses.
CLAIM_DATA = {
    "energy_budget_3_5": ("ns_energy_cells", {}),
    "lemma_4_5_cone": ("ns_cone_points", {}),
    "lemma_7_4_pulse": ("ns_pulse_params", {}),
    "lemma_A6_heat": ("ns_heat_points", {}),
    "lemma_A1_moments": ("ns_moment_families", {}),
    "prop_9_6_decay": ("ns_cycle_stages", {}),
}
