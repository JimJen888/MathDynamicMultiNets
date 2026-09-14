"""
Experiment 1l: why the remaining five are not the same job as the first six.

Six of the eleven have been decomposed here, every derived step validating
against a registered rule. Asked to finish the rest in reasonable time, I
read the proofs of Propositions 7.5 and 9.5 and stopped, because they are
a different kind of text and the difference is measurable.

MEASURE IT, DO NOT ASSERT IT. For each proof, count two things in its own
words: references to the paper's own numbered results and equations, and
named theorems of public mathematics. The proofs this repository has
decomposed lean on both. The ones it has not lean only on the first.

    proof                       lines   internal refs   named theorems
    Lemma 10.4 (energy)            81              20                2
    Lemma 10.5 (comparison)       123              25                3
    Prop 10.1 (localize)           43               6                1
    Prop 9.9 (summation)          111              22                1
    Prop 9.5 (initialize)          96              27                0
    Prop 7.5 (stress)              87              14                0

Zero, in both. Proposition 9.5's proof reasons about residual orders in
M-spaces, a temporal mean update, a five-equation correction and a
covariance assembled two sections earlier. Proposition 7.5's reasons about
slow boxes, band tori, homogeneous pulses and an envelope class. None of
that is public mathematics; all of it is defined across Sections 4 to 8.

WHY THAT CHANGES THE JOB. When a Section 10 proof says "by Cauchy-Schwarz"
the leaf is shared knowledge, and registering it as a rule commits nothing
a reader cannot check against any textbook. When a Section 9 proof says
"the tangential increment is M_{H0}", the leaf is a statement in a private
apparatus. Registering that as a rule would mean writing my own paraphrase
of notation defined a hundred pages earlier, and the decomposition would
then validate because I had written both halves of it.

That failure mode is worth naming precisely, because it would not look
like a failure. Every step would pass, the audit would report MODULO
ASSUMPTIONS with a tidy list, and the list would be my paraphrases rather
than mathematics anyone holds. The count would go from six to eleven and
would mean less than six does.

SO THE HONEST STOPPING POINT. Decomposing the front half is possible and
is not a different method; it is the same method with a prerequisite,
which is to encode the construction's apparatus first. That is Sections 4
through 8 of a 166-page paper, and it is a project rather than a turn.

Run: python examples/run_apparatus.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Line spans of each proof in the extracted text, and whether this
#: repository has decomposed it.
PROOFS = [
    ("Lemma 10.4 (energy)", 6236, 6317, True),
    ("Lemma 10.5 (comparison)", 6317, 6440, True),
    ("Prop 10.1 (localize)", 6109, 6152, True),
    ("Prop 9.9 (summation)", 5949, 6060, True),
    ("Prop 9.5 (initialize)", 5564, 5660, False),
    ("Prop 7.5 (stress)", 4313, 4400, False),
]

INTERNAL = re.compile(r"\(\d+\.\d+\)|Lemma \d|Proposition \d|Theorem \d|"
                      r"Definition \d|Section \d")
NAMED = re.compile(r"Cauchy|Schwarz|Gronwall|Sobolev|Young|Hoelder|Hölder|"
                   r"Riesz|Weierstrass|Borel|Leibniz|Fubini|Minkowski|"
                   r"Calderon|Calderón|fundamental theorem|mean value")


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print("=" * 78)
    print("Why the remaining five are a different job")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    if source is None or not source.exists():
        print("\n  (pass the extracted paper text as an argument to recompute")
        print("   the table; the numbers above were produced this way)")
        return

    lines = source.read_text().splitlines()
    print(f"\n  {'proof':26}{'lines':>7}{'internal':>10}{'named':>8}"
          f"{'decomposed':>12}")
    for label, a, b, done in PROOFS:
        body = lines[a:b]
        internal = sum(len(INTERNAL.findall(x)) for x in body)
        named = sum(len(NAMED.findall(x)) for x in body)
        print(f"  {label:26}{len(body):>7}{internal:>10}{named:>8}"
              f"{'yes' if done else 'no':>12}")


if __name__ == "__main__":
    main()
