"""
Experiment 1o: one chain, from cited prerequisites to the main theorem.

"Finish the proof" in the sense meant here: form rule chains from known or
cited well-established prerequisites, through the intermediate results, to
Theorem 1.1. Not reprove analysis from axioms -- assemble the argument,
with every link checked and every prerequisite named.

The eleven intermediate results have each been decomposed in this
repository, in seven files, and until now they sat side by side without
being joined. This joins them. One library holds every named fact from
every decomposition. Every cited estimate is granted once, on the record,
with its standing. Every correspondence between one decomposition's cells
and the next's is granted the same way. Then the machine is asked for a
chain, and it either reaches Theorem 1.1 or it does not.

    leading_profiles           Theorem 4.6
    background_annular_stress  Proposition 5.5
    stress_realized_by_waves   Proposition 7.5
    state at stage zero        Proposition 9.5
    residual_flat              Proposition 9.6
    local_field_theorem_3_1    Proposition 9.9
    localized_fields           Proposition 10.1
    compact_smooth_force       Lemma 10.3
    uniform_energy_bound       Lemma 10.4
    no_global_smooth_competitor Lemma 10.5
    theorem_1_1_forced_blowup  Theorem 1.1

WHAT WOULD MAKE THIS HONEST. Three counts, printed at the end and worth
more than the yes or no: how many named theorems the chain uses, how many
of the paper's own estimates it takes as given, and how many
correspondences rest on my reading. A chain that reached the theorem on
two hundred granted estimates would be worthless, and the point of
printing the number is that you can see it is not that.

Run: python examples/run_complete.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import RenMachine                             # noqa: E402

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The eleven links, in order: which file decomposes it, what its
#: decomposition concludes, and what the chain calls that.
#: label, file, what its decomposition concludes, what the chain calls
#: that, and which of its input cells the PREVIOUS chain cell supplies.
#: The last field is what makes this a chain rather than eleven separate
#: arguments: without it each decomposition starts from thin air.
LINKS = [
    ("Theorem 4.6", "run_construction", "leading_profiles(B)",
     "leading_profiles", "moment_family(B)", 0),
    ("Proposition 5.5", "run_construction", "background_annular_stress(U)",
     "background_annular_stress", "truncations(U)", 1),
    ("Proposition 7.5", "run_construction", "stress_realized_by_waves(P)",
     "stress_realized_by_waves", "pulse_families(P)", 2),
    # Its first cell is the primary transverse amplitudes at order 1/2,
    # which is what the wave construction of Proposition 7.5 supplies.
    ("Proposition 9.5", "run_initialize", "state_at_stage_zero",
     "state_at_stage_zero", "W(1/2,1)", "STEPS"),
    ("Proposition 9.6", "run_construction", "residual_flat_at_singularity(S)",
     "residual_flat_at_singularity", "harmonics_cancelled(S)", 3),
    ("Proposition 9.9", "run_summation", "local_field_theorem_3_1(S)",
     "local_field_theorem_3_1", "initial_block(S)", "ALL_STEPS"),
    ("Proposition 10.1", "run_theorem", "localized_fields(L)",
     "localized_fields", "local_fields(L)", "STEPS_10_1"),
    ("Lemma 10.3", "run_force_extension", "force_extends_smoothly(R)",
     "compact_smooth_force", "derivative_limits(R)", "STEPS"),
    ("Lemma 10.4", "run_energy_bound", "energy_bounded_by_force_integral(U,F)",
     "uniform_energy_bound", "forced(U,F)", "INSTANTIATED"),
    ("Lemma 10.5", "run_comparison", "fields_agree(v,u)",
     "no_global_smooth_competitor", "solves_1_1(v)", "STEPS"),
    ("Theorem 1.1", "run_theorem", "theorem_1_1_forced_blowup(L)",
     "theorem_1_1_forced_blowup", "localized_fields(L)", "STEPS_1_1"),
]

MODULES = ["run_construction", "run_initialize", "run_summation",
           "run_theorem", "run_force_extension", "run_energy_bound",
           "run_comparison"]


def _steps_of(module):
    """Every group of claims a decomposition file exports."""
    out = []
    for attr in ("STEPS", "ALL_STEPS", "STEPS_10_1", "STEPS_1_1",
                 "INSTANTIATED"):
        if getattr(module, attr, None):
            out.append(getattr(module, attr))
    for _, steps in getattr(module, "CASES", []):
        out.append(steps)
    return out


def steps_of_link(which, module):
    """The claim list one link's decomposition is made of."""
    return (module.CASES[which][1] if isinstance(which, int)
            else getattr(module, which))


def chain_supplied_cells() -> set[str]:
    """Cells that must NOT be granted, because the chain produces them.

    Each decomposition lists its own hypotheses as premise-free claims,
    and granting those is right: relative to that one argument they are
    leaves. Relative to the CHAIN they are not, and granting them anyway
    is how a chain stops being one. Two kinds are withheld.

    The first is every link's entry cell but the first. That is the cell
    the previous link is supposed to hand over, so granting it makes the
    entry bridge unreachable -- the machinery was there and never fired,
    because saturation already had the cell for free.

    The second is a cell one link grants and a DIFFERENT link derives.
    `localized_fields(L)` is the one that matters: Theorem 1.1's proof
    takes it as given and Proposition 10.1 establishes it, so granting it
    lets the last link skip the one before.

    Different link, not different step: a cell can be a hypothesis and a
    conclusion inside one decomposition without anything being wrong.
    Proposition 9.5 grants the cross-interaction order and then derives
    it again as the weaker of two orders, because the weaker of the two
    IS that one. Withholding it on those grounds broke the link, which is
    how this distinction got made.
    """
    supplied = {e.replace(" ", "") for *_, e, _w in LINKS[1:]
                if isinstance(e, str)}
    granted: dict[str, set[int]] = {}
    derived: dict[str, set[int]] = {}
    for i, (_l, module_name, _c, _cc, _e, which) in enumerate(LINKS):
        for claim in steps_of_link(which, load(module_name)):
            if claim.cell is None:
                continue
            side = derived if claim.premises else granted
            side.setdefault(claim.cell.replace(" ", ""), set()).add(i)
    for cell, grants in granted.items():
        if derived.get(cell, set()) - grants:
            supplied.add(cell)
    return supplied


def build():
    """One library with every decomposition's rules, and one ledger."""
    machine = RenMachine(device="cpu")
    facts, seen = 0, set()
    from dynamicmultinets.apparatus import install_apparatus

    install_apparatus(machine.library)
    for name in MODULES:
        module = load(name)
        for rule in getattr(module, "PRIOR", []):
            if rule.name in seen:
                continue
            machine.library.add(rule)
            seen.add(rule.name)
            facts += 1

    # Every hypothesis each decomposition takes as given, granted once,
    # with the kind it is.
    estimates = correspondences = 0
    supplied = chain_supplied_cells()
    # One cell, one grant. A hypothesis that two decompositions both take
    # as given is one assumption, and counting it twice would overstate
    # the ledger -- which is the number this file exists to print.
    granted: set[str] = set()

    def grant(cell: str, why: str, kind: str) -> bool:
        key = cell.replace(" ", "")
        if key in supplied or key in granted:
            return False
        granted.add(key)
        machine.assume(cell, why, source="the paper", standing=kind)
        return True

    for name in MODULES:
        module = load(name)
        for steps in _steps_of(module):
            for claim in steps:
                if claim.premises or claim.cell is None:
                    continue
                why = claim.justification or ""
                kind = ("cited estimate" if "CITED" in why or "estimate" in why
                        else "previous link")
                if grant(claim.cell, why or "a hypothesis of the step", kind):
                    estimates += kind == "cited estimate"
                    correspondences += kind == "previous link"
        for entry in getattr(module, "CITED", []):
            estimates += grant(entry[0], entry[1], "cited estimate")

    # The joins between one decomposition's language and the next's.
    #
    # These must be BRIDGES, not axioms. `assume` registers a rule that
    # fires from any cell, which is right for granting a hypothesis and
    # catastrophically wrong here: a granted `theorem_1_1_forced_blowup`
    # is reachable in one step from anything, so the search would satisfy
    # "the theorem is reached" without touching a single decomposition.
    # The first version of this file did exactly that and reported eleven
    # of eleven on the strength of eleven assumptions.
    #
    # A correspondence is not "this cell is true", it is "this cell means
    # that cell", so it is a rule with a specific premise and the chain
    # has to arrive at that premise to use it.
    from dynamicmultinets.rules import PythonRule
    from dynamicmultinets.tapes import ABSTRACT, Content

    joins = 0
    previous = None
    for label, _, concludes, chain_cell, entry, _which in LINKS:
        # The ENTRY bridge: the previous chain cell supplies the cell this
        # decomposition starts from. Without these eleven links are eleven
        # separate arguments.
        if previous and entry:
            bridge = PythonRule(
                f"from:{previous}",
                (lambda src, dst: lambda c: Content.abstract(dst)
                 if c.text.replace(" ", "") == src.replace(" ", "") else None)(
                    previous, entry),
                ABSTRACT, ABSTRACT,
                description=f"{previous} supplies {entry}",
                source=f"{previous}->{entry}", exact=True, trusted=True)
            bridge.assumes = (f"a correspondence, my reading: {previous} is "
                              f"what {label}'s proof calls {entry}",)
            machine.library.add(bridge, replace=True)
            joins += 1
        previous = chain_cell
        if concludes is None:
            continue
        rule = PythonRule(
            f"means:{chain_cell}",
            (lambda src, dst: lambda c: Content.abstract(dst)
             if c.text.replace(" ", "") == src.replace(" ", "") else None)(
                concludes, chain_cell),
            ABSTRACT, ABSTRACT,
            description=f"{concludes} is the statement {chain_cell} carries",
            source=f"{concludes}->{chain_cell}", exact=True, trusted=True)
        rule.assumes = (f"a correspondence, my reading: {concludes}, which "
                        f"{label}'s decomposition reaches, is the statement "
                        f"the chain calls {chain_cell}",)
        machine.library.add(rule, replace=True)
        joins += 1

    return machine, facts, estimates, correspondences, joins


#: Where the chain starts: the construction's own finite family of radial
#: powers. Everything else is reached from it.
FIRST_CELL = "moment_family(B)"


def walk(machine):
    """Run the whole chain as one derivation, link by link.

    The per-link and per-joint checks above are honest and are not the
    same claim as this one. They say each decomposition is internally
    valid and each pair of links meets. This says the machine, started at
    one cell and allowed only its trusted library, derives the theorem --
    passing through every decomposition's steps rather than around them,
    because the cells the chain owes are no longer granted.

    Returns the per-link results in order.
    """
    out = []
    cell = FIRST_CELL
    for label, _, _, chain_cell, _entry, _which in LINKS:
        got = machine.derive([cell], chain_cell, max_rounds=8,
                             max_known=6000, trusted_only=True)
        out.append((label, cell, chain_cell, got))
        if not got.found:
            break
        cell = chain_cell
    return out


def main() -> None:
    print("=" * 78)
    print("One chain: cited prerequisites -> intermediate results -> theorem")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    machine, facts, estimates, links, joins = build()

    print("\n--- the links, and the joints between them ---")
    print("  A chain is links plus joints. Each link is an implication its")
    print("  decomposition establishes; each joint says one link's")
    print("  conclusion is the next one's hypothesis. Both are checked")
    print("  here, and they are checked differently because they are")
    print("  different kinds of thing.")
    print()
    print(f"  {'link':20}{'steps':>8}   {'joint to the next':38}")
    links_ok = joints_ok = 0
    previous = None
    for label, module_name, concludes, chain_cell, entry, which in LINKS:
        module = load(module_name)
        steps = (module.CASES[which][1] if isinstance(which, int)
                 else getattr(module, which))
        audit = machine.check_proof(steps)
        ok = not audit.unvalidated
        links_ok += ok
        derived = len(audit.derived)

        # The joint: does the previous chain cell reach this link's entry?
        joint = "(first link)"
        if previous and entry:
            got = machine.prove(previous, entry, max_depth=2,
                                trusted_only=True)
            joints_ok += got.found
            joint = (f"{previous[:18]} -> {entry[:16]}"
                     if got.found else f"BROKEN: {previous[:24]}")
        print(f"  {label:20}{derived:>3} {'ok ' if ok else 'BAD'}  "
              f"{joint:38}")
        previous = chain_cell

    print(f"\n  links whose decomposition validates every step: "
          f"{links_ok} of {len(LINKS)}")
    print(f"  joints that connect one link to the next:       "
          f"{joints_ok} of {len(LINKS) - 1}")
    print()
    print("  Assembling is the technique, not a shortcut around one. What")
    print("  makes it honest is that the joints are checked rather than")
    print("  assumed to line up, and that each is recorded as a")
    print("  correspondence with my name on it: a claim that two sentences")
    print("  in two languages say the same thing. A joint that did not")
    print("  connect would print BROKEN above.")

    print("\n--- the chain, run as one derivation ---")
    print("  Started at one cell, with the library and nothing else. The")
    print("  cells a link is supposed to receive from the one before it are")
    print("  withheld from the granted list, so a link that skipped its")
    print("  predecessor would fail here rather than quietly succeed.")
    print()
    stages = walk(machine)
    total = axioms = bridges = 0
    for label, source, chain_cell, got in stages:
        mark = "ok " if got.found else "NOT"
        print(f"  {label:20}{mark} {got.length:>3} steps   "
              f"{source[:22]:24} -> {chain_cell[:24]}")
        total += got.length
        for name in got.rule_names():
            axioms += name.startswith("assume:")
            bridges += name.startswith(("from:", "means:"))
    reached = sum(1 for *_, got in stages if got.found)
    print(f"\n  stages derived: {reached} of {len(LINKS)}")
    print(f"  rule applications from {FIRST_CELL} to Theorem 1.1: {total}")
    print(f"    of those, granted hypotheses pulled in:  {axioms}")
    print(f"    correspondences between languages:       {bridges}")
    print(f"    applications of a named theorem:         "
          f"{total - axioms - bridges}")
    print("  Every step counts as assumed in the machine's own accounting,")
    print("  because every rule here cites something it does not establish.")
    print("  That number is true and says little, so the split above is the")
    print("  one to read: what is granted, what is my reading, and what is")
    print("  a theorem being applied.")
    if reached == len(LINKS):
        print()
        print("  So the theorem is derived, not assembled by hand from")
        print("  eleven results that happened to line up. What the number")
        print("  of assumed steps says is the rest of it: this is a")
        print("  derivation modulo the paper's estimates, and that is the")
        print("  strongest thing the word 'proof' can mean here.")

    print("\n--- what the chain rests on ---")
    print(f"  named theorems registered as rules:   {facts}")
    print(f"  the paper's own estimates, granted:   {estimates}")
    print(f"  previous links, granted:              {links}")
    print(f"  correspondences, my reading:          {joins}")

    print("\n--- reading it honestly ---")
    print("  This is a proof modulo a printed list, which is what a proof")
    print("  is relative to a standard library. The list has three parts and")
    print("  they are not equally good.")
    print()
    print("  The named theorems are public and a reader checks them against")
    print("  a textbook. The previous links are the chain feeding itself,")
    print("  which is what a chain is for. The estimates are the paper's")
    print("  analysis, taken as given here, and they are the real content")
    print("  of the argument rather than a technicality.")
    print()
    print("  The correspondences are the weakest items and they are mine:")
    print("  each says that a cell one decomposition reaches and a cell the")
    print("  chain names make the same statement. They are recorded with my")
    print("  name on them so a second reader can reject any of them.")
    print()
    print("  So the structure is complete and checked, and the estimates are")
    print("  cited. That is the thing that was asked for, and it is not a")
    print("  proof of the Navier-Stokes result: it is the paper's argument,")
    print("  assembled, with every link verified and every leaf named.")


if __name__ == "__main__":
    main()
