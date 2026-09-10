"""
Provers: the third way a rule can earn its trust.

There were two before this, and they answer different questions.

    an ORACLE decides one instance by a route the rule does not use, and
    agreeing with it on many instances measures the rule

    a REFERENCE CHAIN does the same with other rules instead of an oracle

Both are sampling. They establish a rule on the questions it was asked and
say nothing about the ones it was not, which is exactly right for a rule
that reads pixels or integrates a field, and exactly wrong for a rule whose
content is a statement about every case at once. The Navier-Stokes run says
so in as many words: its imported steps stay untrusted however many
instances agree, because each asserts something uniform and no number of
instances reaches a statement of that shape.

A PROVER is the case where something does reach it. It takes the rule and
decides its answer on the rule's whole input domain, by an argument rather
than by cases -- and when it succeeds the rule is not "accurate to four
digits", it is right, and `exact` is the flag that says so.

This is a narrow door and it is meant to be. A prover has to be a decision
procedure for a fragment the rule actually lives in; `linarith` is one such
fragment, and the reason the correction cycle of Proposition 9.6 can go
through it is that the cycle is rational arithmetic on an integer stage
index and nothing else. The estimates the cycle is built from are not in
any such fragment, they stay imported, and a prover that claimed otherwise
would be the most expensive kind of mistake this package can make.

So the contract is deliberately strict:

  * `covers` must say what domain was decided, in words, and the rule is
    marked exact only if that is the rule's whole domain.
  * a prover returns `established=False` with a reason rather than raising,
    because "this could not be decided" is a result and gets printed.
  * `statement` is what was proved, and it is recorded on the rule, so a
    later reader can see the difference between a rule at 0.9992 over 1200
    instances and a rule that is simply right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Judgement:
    """What a prover concluded."""

    established: bool
    #: What was proved, or would have been. One sentence, no hedging.
    statement: str = ""
    #: The domain the argument covers, in words.
    covers: str = ""
    #: Whether `covers` is the rule's entire input domain. A prover that
    #: decides only part of one has still done something worth printing,
    #: and the rule does not become exact on the strength of it.
    whole_domain: bool = False
    #: How it was decided: the inequalities, the interval, the case count.
    detail: list[str] = field(default_factory=list)
    #: Why not, when it was not.
    obstruction: str = ""


@dataclass
class Prover:
    name: str
    description: str
    fn: Callable[..., Judgement]

    def __call__(self, *args: Any, **kwargs: Any) -> Judgement:
        return self.fn(*args, **kwargs)


PROVERS: dict[str, Prover] = {}


def prover(name: str, description: str):
    """Register a decision procedure under a name the machine can call."""

    def wrap(fn: Callable[..., Judgement]) -> Callable[..., Judgement]:
        PROVERS[name] = Prover(name, description, fn)
        return fn

    return wrap


def get(name: str) -> Prover:
    if name not in PROVERS:
        raise KeyError(f"no prover {name!r}; known: {sorted(PROVERS)}")
    return PROVERS[name]
