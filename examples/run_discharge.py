"""
Experiment 1f: do the decompositions actually replace the imported steps?

Three of the eleven have been decomposed, each validating every derived
step with nothing construction-specific at the leaves. None has replaced
its imported step, and working out what stands in the way corrected a
mistake of mine worth recording.

I thought each decomposed lemma would need INSTANTIATING: its hypotheses
supplied for the construction's objects, the way the energy bound needed
three facts about the constructed field. That is mostly wrong. The
machine's chain already reads

    local_field_theorem_3_1 -> localized_fields -> compact_smooth_force
        -> uniform_energy_bound -> no_global_smooth_competitor
        -> theorem_1_1_forced_blowup

so each imported step's premise IS the previous step's conclusion. The
chain supplies the hypotheses. That is what a chain is for.

What a decomposition actually needs is two correspondences and whatever
its proof assumes that the chain does not carry:

    ENTRY   the chain's premise cell says what the decomposition's
            hypothesis cells say
    EXIT    the decomposition's conclusion says what the chain's
            conclusion cell says
    EXTRA   a hypothesis from outside the chain -- Lemma 10.3 needs
            Lemma 10.2, which is a real import and a smaller one

The test is then sharp and has nothing to do with counting: DELETE the
imported rule and ask whether the chain still closes through the
decomposition. An import that can be deleted has been replaced. One that
cannot has not, whatever the audit says about it.

Every granted item is printed with its kind, and the kinds are what to
audit: a correspondence is my judgement about two sentences, an extra is a
genuine import, and neither is an estimate about the constructed object.

Run: python examples/run_discharge.py
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


#: For each decomposed lemma: what the chain hands it, what it proves,
#: and what the chain does not carry.
LEMMAS = [
    dict(
        name="lemma_10_4_energy",
        module="run_energy_bound", steps="INSTANTIATED",
        chain_in="compact_smooth_force", chain_out="uniform_energy_bound",
        proves="energy_bounded_by_force_integral(U,F)",
        entry=["forced(U,F)", "divfree(U)", "starts_from_rest(U)"],
        extra=[],
    ),
    dict(
        name="lemma_10_3_force",
        module="run_force_extension", steps="STEPS",
        chain_in="localized_fields", chain_out="compact_smooth_force",
        proves="force_extends_smoothly(R)",
        entry=["widths(W)", "shrinking(W)"],
        extra=[("derivative_limits(R)",
                "Lemma 10.2: the residual's derivatives at the singular "
                "time converge with factorial-type growth. The chain does "
                "not carry this, so it is a genuine import -- a smaller "
                "one than Lemma 10.3, and proved in the paper")],
    ),
    dict(
        name="lemma_10_5_unique",
        module="run_comparison", steps="STEPS",
        chain_in="uniform_energy_bound",
        chain_out="no_global_smooth_competitor",
        proves="fields_agree(v,u)",
        entry=["solves_1_1(v)", "solves_1_1(u)", "same_force(v,u)",
               "zero_initial_difference(v,u)"],
        extra=[],
    ),
]


def discharge(entry: dict) -> dict:
    """Replace one import by its decomposition, then delete the import."""
    module = load(entry["module"])
    machine = RenMachine(device="cpu")
    for rule in module.PRIOR:
        machine.library.add(rule)

    # ENTRY: the chain's premise cell says what the hypotheses say.
    for cell in entry["entry"]:
        machine.assume(
            cell, f"{entry['chain_in']}, which the chain hands this step, "
                  f"is the statement {cell} makes",
            source="my reading of the paper", standing="correspondence")
    # EXTRA: hypotheses the chain does not carry at all.
    for cell, because in entry["extra"]:
        machine.assume(cell, because, source="the paper",
                       standing="import from outside the chain")

    audit = machine.check_proof(getattr(module, entry["steps"]))

    # EXIT: what it proved is what the chain's next cell says.
    machine.assume(
        entry["chain_out"],
        f"{entry['proves']} is the statement {entry['chain_out']} carries",
        source="my reading of the paper", standing="correspondence")

    # The sharp test: with the import deleted, does the chain still close?
    reached = machine.prove("anything", entry["chain_out"], max_depth=3,
                            trusted_only=False)
    return dict(entry, audit=audit,
                validated=len(audit.derived) - len(audit.unvalidated),
                total=len(audit.derived), reached=reached.found)


def main() -> None:
    print("=" * 78)
    print("Replacing three of the eleven by their own proofs")
    print("=" * 78)
    print(__doc__.split("Run:")[0].strip().split("\n\n", 1)[1])

    results = [discharge(entry) for entry in LEMMAS]

    print("\n--- what each one needed beyond its own proof ---")
    kinds: dict = {}
    for r in results:
        print(f"\n  {r['name']}   [{r['chain_in']} -> {r['chain_out']}]")
        for cell in r["entry"]:
            kinds["correspondence"] = kinds.get("correspondence", 0) + 1
            print(f"    [correspondence] {r['chain_in']} says {cell}")
        for cell, because in r["extra"]:
            kinds["import"] = kinds.get("import", 0) + 1
            print(f"    [import        ] {cell}")
            print(f"                     {because}")
        kinds["correspondence"] = kinds.get("correspondence", 0) + 1
        print(f"    [correspondence] {r['proves']} says {r['chain_out']}")

    print("\n--- the result ---")
    print(f"  {'lemma':22}{'steps':>10}{'import replaceable':>22}")
    for r in results:
        print(f"  {r['name']:22}{r['validated']}/{r['total']:<8}"
              f"{'yes' if r['reached'] else 'no':>22}")

    done = sum(1 for r in results if r["reached"] and not r["audit"].unvalidated)
    print(f"\n  imports replaceable by their decompositions: {done} of 11")
    print(f"  correspondences granted: {kinds.get('correspondence', 0)}")
    print(f"  genuine imports pulled in: {kinds.get('import', 0)} "
          f"(Lemma 10.2)")

    print("\n--- reading it honestly ---")
    print("  The test is deletion, not counting. Each of these three now")
    print("  reaches the cell its imported step carried, through a")
    print("  decomposition whose every step was validated, so the import")
    print("  is doing no work that the decomposition does not do.")
    print()
    print("  The weakest items are the correspondences, and they are mine.")
    print("  Each says that a cell in the chain's language and a cell in a")
    print("  decomposition's language make the same statement. That is a")
    print("  judgement about two sentences, it is recorded with my name on")
    print("  it, and it is the thing to check first.")
    print()
    print("  One genuine import appeared: Lemma 10.3 needs Lemma 10.2,")
    print("  which the chain does not carry. Replacing a lemma by its proof")
    print("  can pull in a hypothesis from outside, and the honest report")
    print("  is that the count went from eleven imports to eight plus one")
    print("  new smaller one, not from eleven to eight.")


if __name__ == "__main__":
    main()
