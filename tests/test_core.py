"""
Tests for the parts that must not quietly break.

Split in two: everything above `torch` runs with numpy alone (the tapes, the
renderer, the symbolic rules, proof search, the halting calibration, the
conciseness accounting), and the learned-rule tests are skipped when torch is
absent. That split is the same one the package makes internally, so these tests
also check that importing dynamicmultinets does not drag in torch.

    pytest tests/            # or: python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinets import (ABSTRACT, SPECIFIC, Content, RenMachine,  # noqa: E402
                             RuleLibrary, ScriptedController, Task,
                             library_report, search)
from dynamicmultinets.codec import ChoiceCodec, TextSlotCodec  # noqa: E402
from dynamicmultinets.halting import (MAX_SAMPLES, calibrate,  # noqa: E402
                                     effective_lambda, sample_size)
from dynamicmultinets.palette import PALETTE, rgb_to_class_index  # noqa: E402
from dynamicmultinets.prior import eval_int_expression  # noqa: E402
from dynamicmultinets.render import render_text, split_views  # noqa: E402


# ---------------------------------------------------------------------------
# Tapes and domains
# ---------------------------------------------------------------------------
def test_tapes_reject_cross_domain_writes():
    m = RenMachine(with_prior=False)
    with pytest.raises(ValueError):
        m.abstract.write(Content.specific_text("12*30"))


def test_specific_cell_has_no_readable_text_without_a_rule():
    """The architectural claim: you cannot read the specific tape by asking."""
    cell = Content.specific_text("12*30")
    assert cell.image is not None
    with pytest.raises(ValueError):
        Content.abstract("x").require_image()


# ---------------------------------------------------------------------------
# Renderer / palette contract
# ---------------------------------------------------------------------------
def test_render_is_exact_palette_colours():
    """Every rendered pixel must be a palette anchor, or quantization is lossy
    and a tape cell stops being reproducible."""
    img = render_text("12*30+4")
    idx = rgb_to_class_index(img)
    assert np.array_equal(PALETTE[idx], img)


def test_two_views_are_different_readings():
    a, b = split_views(render_text("10*30+2*30"))
    assert a.shape == b.shape
    assert not np.array_equal(a, b)      # fa - fb would be meaningless otherwise


def test_render_is_deterministic():
    assert np.array_equal(render_text("7*8"), render_text("7*8"))


# ---------------------------------------------------------------------------
# Prior rules
# ---------------------------------------------------------------------------
def test_arithmetic_is_restricted():
    assert eval_int_expression("10*30+2*30") == 360
    for bad in ("__import__('os').system('ls')", "2**99", "1/0", "open('x')"):
        with pytest.raises(Exception):
            eval_int_expression(bad)


def test_known_chain_computes_a_product():
    m = RenMachine()
    m.write(ABSTRACT, "12*30")
    m.apply_rule("decimal_split")
    m.apply_rule("distribute_symbolic")
    assert m.apply_rule("eval_arith").text == "360"


def test_transcribe_declines_an_observed_cell():
    """The baseline caption-copier must refuse a cell the machine did not draw,
    or every perception task would be trivially solvable."""
    m = RenMachine()
    seen = Content.specific_text("47*83")
    seen.meta["observed"] = True
    assert m.library.get("transcribe_unsafe").apply(seen) is None


def test_a_decomposition_can_be_carried_down_to_the_times_table():
    """Splitting only the left factor stops one level above the 9x9 table:
    `46*19 -> 40*19+6*19` and then neither part has two non-zero places on the
    left. The mirror rules finish it, which is what makes an arbitrarily large
    product reachable from a small table."""
    m = RenMachine()
    from dynamicmultinets.propose import gather_analogy

    tree = ["46*19 => 40*19+6*19",          # split left
            "40*19 => 40*10+40*9",          # then right
            "6*19 => 6*10+6*9",             # and right again
            "6*9 => 54", "40*10 => 400"]    # bottoming out in arithmetic
    a = gather_analogy(m, unsolved=tree)
    assert not a.unsolved, [c.text for c in a.unsolved]
    right = next(c for c in a.solved if c.text.startswith("40*19"))
    assert right.derivation == ["decimal_split_right", "distribute_symbolic_right"]


def test_the_mirror_rules_decline_what_has_nothing_to_split():
    """A single-digit factor has one place value, so there is no sum to make
    and the rule must decline rather than emit a degenerate rewrite."""
    m = RenMachine()
    right = m.library.get("decimal_split_right")
    left = m.library.get("decimal_split")

    assert right.apply(Content.abstract("40*9")) is None    # one digit on the right
    assert right.apply(Content.abstract("40*10")) is None   # 10 is a single place
    assert right.apply(Content.abstract("40*19")).text == "40*(10+9)"

    assert left.apply(Content.abstract("6*19")) is None     # one digit on the left
    assert left.apply(Content.abstract("46*19")).text == "(40+6)*19"


def test_the_right_split_oracle_checks_each_instance_numerically():
    """Like its left-hand twin, it does not assert the identity -- it evaluates
    the rewrite and refuses to emit one that does not hold."""
    from dynamicmultinets.dataset import Example
    from dynamicmultinets.oracles import ORACLES

    fn = ORACLES["distributive_rewrite_right"].fn
    assert fn(Example(inp=Content.specific_text("40*19"))).text == "40*10+40*9"
    assert fn(Example(inp=Content.specific_text("2*15"))).text == "2*10+2*5"
    assert fn(Example(inp=Content.specific_text("6*9"))) is None      # nothing to split
    for text in ("40*19", "6*19", "12*15", "47*83"):
        out = fn(Example(inp=Content.specific_text(text)))
        if out is not None:
            assert eval_int_expression(out.text) == eval_int_expression(text)


def test_a_declined_rule_says_which_of_the_two_causes_it_was():
    """A refusal has two causes needing opposite fixes -- wrong tape, or right
    tape and wrong shape -- and they read identically from outside. The message
    has to distinguish them, and name what would work instead."""
    m = RenMachine()
    m.write(ABSTRACT, "96*83")

    with pytest.raises(ValueError) as form:
        m.apply_rule("distribute_symbolic")          # wants (a+b)*c
    said = str(form.value)
    assert "The domain is right" in said and "FORM is not" in said
    assert "(a+b)*c" in said
    assert "decimal_split" in said                   # the step it is missing

    with pytest.raises(ValueError) as dom:
        m.apply_rule("transcribe_unsafe", domain=ABSTRACT)
    assert "reads specific cells" in str(dom.value)


def test_a_check_that_examined_nothing_says_which_link_stopped():
    """'NO CHECKS RAN' looks like a failure of the rule and is almost always a
    mismatch several steps earlier, so the link that declined is measured."""
    from dynamicmultinets.verify import verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 20, seed=3, name="d",
                    a_digits=2, b_digits=2, domain=ABSTRACT)
    m.label_data("d", "distributive_rewrite")
    rep = verify_against_rules(m.library, "decimal_split",
                               ["distribute_symbolic"], m.datasets["d"])
    assert rep.n_checked == 0
    said = rep.summary()
    assert "the reference stops at 'distribute_symbolic'" in said
    assert "20/20" in said and "(a+b)*c" in said


def test_a_wrong_oracle_is_reported_as_a_wrong_oracle():
    """When nothing is labelled, the oracle does not fit the generator, and
    saying so beats blaming the rule."""
    from dynamicmultinets.verify import verify_rule

    m = RenMachine()
    m.generate_data("mul_pairs", 12, seed=1, name="d",
                    a_digits=2, b_digits=2, domain=ABSTRACT)
    rep = verify_rule(m.library, "decimal_split", m.datasets["d"],
                      oracle_name="alternate_angle_facts")
    assert rep.n_checked == 0
    assert "wrong oracle" in rep.summary()


def test_an_unsupportable_eps_names_the_one_that_would_work():
    """A refusal here is never 'impossible', only 'not at this eps', and the
    boundary is computable -- so it is computed. A caller with a step budget
    retries once instead of bisecting."""
    from dynamicmultinets.halting import halting_budget_for_library, tightest_eps

    lengths = [2, 1, 2, 1]
    with pytest.raises(ValueError) as err:
        halting_budget_for_library(lengths, 0.97, eps=0.1, delta=0.05)
    suggested = tightest_eps(len(lengths), 0.05)
    assert f"retry with eps={suggested}" in str(err.value)

    # The suggestion works, and is the tightest that does.
    cal = halting_budget_for_library(lengths, 0.97, eps=suggested, delta=0.05)
    assert cal.threshold >= 1
    with pytest.raises(ValueError):
        halting_budget_for_library(lengths, 0.97, eps=round(suggested - 0.01, 2),
                                   delta=0.05)


def test_tightest_eps_gives_up_when_nothing_would_work():
    """One sample supports no eps below 1.0, and saying so beats suggesting a
    number that would fail too."""
    from dynamicmultinets.halting import tightest_eps

    assert tightest_eps(0, 0.05) is None
    assert tightest_eps(1, 0.05) is None          # lambda >= 1 with one sample
    assert tightest_eps(500, 0.05) < 0.1          # a real sample buys a tight eps


# ---------------------------------------------------------------------------
# Deciding what to form
# ---------------------------------------------------------------------------
def test_a_case_is_measured_solved_not_taken_on_the_caller_s_word():
    """A caller who mislabels its own examples would be asking for a hypothesis
    about a distinction that is not there, so every case is put to proof search
    and moved to the list its result says it belongs in."""
    from dynamicmultinets.propose import gather_analogy

    m = RenMachine()
    # Handed in as UNSOLVED, but the machine already has the identity.
    a = gather_analogy(m, unsolved=["12*30 => 10*30+2*30"])
    assert [c.text for c in a.solved] == ["12*30 => 10*30+2*30"]
    assert not a.unsolved
    assert a.solved[0].derivation == ["decimal_split", "distribute_symbolic"]


def test_the_same_case_is_unsolved_once_it_is_a_drawing():
    """Posing the case on the specific tape is a different question: nothing
    can read the screen yet, and `observed` stops the caption being copied."""
    from dynamicmultinets.propose import gather_analogy

    m = RenMachine()
    a = gather_analogy(m, unsolved=["12*30 => 10*30+2*30"],
                       domain=SPECIFIC, observed=True)
    assert [c.text for c in a.unsolved] == ["12*30 => 10*30+2*30"]
    assert a.unsolved[0].image is not None


def test_a_case_reaches_the_controller_as_its_layout():
    """What goes up is what the cell DRAWS -- how many boxes, joined by what,
    with which factors stacked -- in readable text. Both sides of the case,
    because the regrouping is in what it must become, not in where it starts."""
    from dynamicmultinets.propose import gather_analogy

    m = RenMachine()
    a = gather_analogy(m, unsolved=["12*30 => 10*30+2*30"],
                       domain=SPECIFIC, observed=True)
    view = a.unsolved[0].view_text
    assert "1 box" in view                      # 12*30 is one box
    assert "2 boxes, joined by '+'" in view     # the target is two
    assert "must become" in view
    # Glyphs come back as the characters they were drawn from, not as marks.
    assert "12" in view and "*30" in view


def test_layout_text_shows_the_regrouping_as_a_change_in_boxes():
    """The distributive analogy is a layout fact: one box becomes two, split at
    the place-value boundary. That is what a proposer is meant to notice."""
    from dynamicmultinets.render import layout_text

    assert "1 box" in layout_text("12*30")
    two = layout_text("10*30+2*30")
    assert "2 boxes, joined by '+'" in two
    assert "box 1: 10" in two and "box 2: 2" in two
    # A product is stacked, which is how the specific domain draws it.
    assert "*30" in two
    # Relations split too, so a rewrite reads as boxed sides.
    assert "'='" in layout_text("A1+A3+A2=180")


def test_split_case_reads_a_target():
    from dynamicmultinets.propose import split_case

    assert split_case("12*30 => 10*30+2*30") == ("12*30", "10*30+2*30")
    assert split_case("12*30 -> 360") == ("12*30", "360")
    assert split_case("12*30") == ("12*30", "")


def test_a_proposal_must_name_things_that_exist():
    """The controller is a language model, so it selects from the registries
    rather than inventing. A hallucinated generator is rejected here, not
    discovered halfway through training."""
    from dynamicmultinets.propose import RuleProposal, validate

    m = RenMachine()
    ok = RuleProposal(name="d", domain_in=SPECIFIC, domain_out=SPECIFIC,
                      generator="mul_pairs", oracle="distributive_rewrite",
                      num_slots=16)
    assert validate(ok, m.library) == []

    bad = RuleProposal(name="d", domain_in=SPECIFIC, domain_out=SPECIFIC,
                       generator="invent_something", oracle="wishful", num_slots=16)
    problems = validate(bad, m.library)
    assert any("no generator" in p for p in problems)
    assert any("no oracle" in p for p in problems)

    clash = RuleProposal(name="render", domain_in=SPECIFIC, domain_out=SPECIFIC,
                         generator="mul_pairs", oracle="distributive_rewrite",
                         num_slots=16)
    assert any("already exists" in p for p in validate(clash, m.library))

    stray = RuleProposal(name="d", domain_in=SPECIFIC, domain_out=SPECIFIC,
                         generator="mul_pairs", oracle="distributive_rewrite",
                         num_slots=16, unknown={"nonsense": 1})
    assert any("takes no parameter" in p for p in validate(stray, m.library))

    # Values, not just names: these pass a spell-check and fail on execution.
    crashes = RuleProposal(name="d", domain_in=SPECIFIC, domain_out=SPECIFIC,
                           generator="mul_pairs", oracle="distributive_rewrite",
                           num_slots=16,
                           unknown={"a_digits": 2, "b_digits": 2,
                                    "tail_digits": 0, "domain": "specific"})
    assert any("fails" in p for p in validate(crashes, m.library))

    wrong_tape = RuleProposal(name="d", domain_in=SPECIFIC, domain_out=SPECIFIC,
                              generator="mul_pairs", oracle="distributive_rewrite",
                              num_slots=16,
                              unknown={"a_digits": 2, "domain": "abstract"})
    assert any("writes abstract" in p for p in validate(wrong_tape, m.library))


def test_a_shared_pattern_states_a_transfer_not_just_a_family():
    """The claim is 'what holds there also holds here', so a proposal names the
    family where the property is established AND the one it is claimed for."""
    from dynamicmultinets.propose import RuleProposal, validate

    m = RenMachine()
    p = RuleProposal(name="distributive_on_three_digits",
                     domain_in=SPECIFIC, domain_out=SPECIFIC,
                     generator="mul_pairs", oracle="distributive_rewrite",
                     known={"a_digits": 2, "b_digits": 2, "domain": "specific"},
                     unknown={"a_digits": 3, "b_digits": 2, "domain": "specific"},
                     condition="place-value splitting does not depend on width",
                     num_slots=20)
    assert validate(p, m.library) == []
    claim = p.claim()
    assert "established on" in claim and "also holds on" in claim
    assert "provided place-value" in claim
    # It trains on the family being CLAIMED, not the one already established.
    assert p.as_data_args()["params"]["a_digits"] == 3
    assert "applies to" in p.check(m)

    same = RuleProposal(name="d", domain_in=SPECIFIC, domain_out=SPECIFIC,
                        generator="mul_pairs", oracle="distributive_rewrite",
                        known={"a_digits": 2}, unknown={"a_digits": 2},
                        num_slots=16)
    assert any("no transfer" in x for x in validate(same, m.library))


def test_a_property_can_be_carried_to_another_form_of_itself():
    """The sharpest transfer is not a wider family but the same property
    mirrored: the distributive law established as a split of the left factor,
    claimed as a split of the right one, over the very same numbers."""
    from dynamicmultinets.propose import RuleProposal, validate

    m = RenMachine()
    fam = {"a_digits": 2, "b_digits": 2, "domain": SPECIFIC}
    p = RuleProposal(name="distributive_on_the_right",
                     domain_in=SPECIFIC, domain_out=SPECIFIC,
                     generator="mul_pairs",
                     known_oracle="distributive_rewrite",
                     oracle="distributive_rewrite_right",
                     known=fam, unknown=fam, num_slots=16,
                     condition="place value does not prefer a side")
    assert validate(p, m.library) == []      # identical families are fine here
    assert p.transfers_form()
    claim = p.claim()
    assert "established as 'distributive_rewrite'" in claim
    assert "also holds as 'distributive_rewrite_right'" in claim
    # Both halves are measured: the base as well as the extension.
    checked = p.check(m)
    assert "unknown family" in checked and "known family" in checked


def test_a_transfer_from_nowhere_is_caught():
    """'Established' is half the claim. A property that never held on the known
    family has nothing to carry across, and saying so beats training a net."""
    from dynamicmultinets.propose import RuleProposal, validate

    m = RenMachine()
    p = RuleProposal(name="from_nothing", domain_in=SPECIFIC, domain_out=SPECIFIC,
                     generator="mul_pairs",
                     known_oracle="distributive_rewrite",
                     oracle="distributive_rewrite_right",
                     # one digit on the left: the left-hand split never applies
                     known={"a_digits": 1, "b_digits": 2, "domain": SPECIFIC},
                     unknown={"a_digits": 2, "b_digits": 2, "domain": SPECIFIC},
                     num_slots=16)
    assert validate(p, m.library) == []
    assert "no base" in p.check(m)

    bogus = RuleProposal(name="b", domain_in=SPECIFIC, domain_out=SPECIFIC,
                         generator="mul_pairs", oracle="distributive_rewrite",
                         known_oracle="not_a_real_oracle",
                         known={"a_digits": 2}, unknown={"a_digits": 3},
                         num_slots=16)
    assert any("to be established in" in x for x in validate(bogus, m.library))


def test_an_interconversion_is_settled_by_searching_for_the_chain():
    """'The unproven case maps to a solved one.' The test is finding the chain,
    which is what proof search does -- so `check` runs it."""
    from dynamicmultinets.propose import Interconversion, validate

    m = RenMachine()
    p = Interconversion(name="transport", source="12*30", target="10*30+2*30",
                        rationale="the same product, regrouped")
    assert validate(p, m.library) == []
    assert "maps -> the established" in p.claim()
    found = p.check(m)
    assert "FOUND" in found and "decimal_split" in found

    nowhere = Interconversion(name="t2", source="12*30", target="not_reachable")
    assert "no chain" in nowhere.check(m, max_depth=3)

    degenerate = Interconversion(name="t3", source="12*30", target="12*30")
    assert any("same statement" in x for x in validate(degenerate, m.library))

    # The stated domain is honoured: the same claim about DRAWINGS is a
    # different search, and nothing can read the screen yet.
    drawn = Interconversion(name="t4", source="12*30", target="10*30+2*30",
                            domain=SPECIFIC)
    assert "no chain" in drawn.check(m, max_depth=6)

    # A suggested route is a hint, so wrong names cost the hint, not the claim.
    hinted = Interconversion(name="t5", source="12*30", target="10*30+2*30",
                             via=["decimal_split", "next_construction_step"])
    assert validate(hinted, m.library) == []
    assert hinted.via == ["decimal_split"]          # the oracle name is gone
    assert "dropped from the suggested route" in hinted.rationale


def test_bad_json_from_the_model_costs_a_proposal_not_a_crash():
    from dynamicmultinets.propose import parse_proposals

    m = RenMachine()
    assert parse_proposals("I think we should try harder", m.library) == []
    assert parse_proposals("[{not json", m.library) == []

    good = """Sure, here you go:
    [{"name": "distributive_learned", "domain_in": "specific",
      "domain_out": "specific", "generator": "mul_pairs",
      "oracle": "distributive_rewrite", "num_slots": 16,
      "kind": "shared_pattern", "rationale": "split at the place-value mark"},
     {"name": "nope", "domain_in": "specific", "domain_out": "specific",
      "generator": "does_not_exist", "oracle": "distributive_rewrite",
      "num_slots": 16}]"""
    got = parse_proposals(good, m.library)
    assert [p.name for p in got] == ["distributive_learned"]
    assert got[0].as_declare_args()["num_slots"] == 16


def test_proposing_changes_nothing_in_the_library():
    """A proposal is a question. Only declare_rule and verify_rule may answer
    it, so nothing here may quietly install a rule."""
    m = RenMachine()
    before = sorted(m.library.rules)
    _, proposals = m.propose_rules(unsolved=["12*30 => 10*30+2*30"],
                                   domain=SPECIFIC, observed=True)
    assert sorted(m.library.rules) == before
    assert proposals, "the drawn case is unsolved, so something should be proposed"
    assert "distributive_rewrite" in {p.oracle for p in proposals}
    assert all(not p.name in m.library for p in proposals)


# ---------------------------------------------------------------------------
# What agreement with a reference is worth
# ---------------------------------------------------------------------------
def test_primitives_sees_through_a_composite():
    """A chain hides its members, so comparing top-level names would call a
    reference independent of a rule it actually runs."""
    from dynamicmultinets.rules import CompositeRule
    from dynamicmultinets.verify import primitives

    m = RenMachine()
    chain = CompositeRule("ref", [m.library.get("decimal_split"),
                                  m.library.get("distribute_symbolic")])
    assert primitives(chain) == {"ref", "decimal_split", "distribute_symbolic"}


def test_verification_against_a_reference_that_runs_the_rule_is_refused():
    """The one failure mode that produces a perfect score: a reference which
    invokes the rule under test agrees with it for free."""
    from dynamicmultinets.verify import verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 20, seed=3, name="d", a_digits=2, b_digits=2,
                    domain=ABSTRACT)
    m.label_data("d", "distributive_rewrite")
    with pytest.raises(ValueError, match="cannot be verified against a reference"):
        verify_against_rules(m.library, "decimal_split",
                             ["decimal_split", "distribute_symbolic"],
                             m.datasets["d"])


def test_sharing_an_exact_rule_is_not_circularity():
    """What makes a shared component fatal is that its ERRORS are shared. An
    exact rule has none, so two routes that both finish by computing still
    disagree wherever their perception differs -- refusing that check would
    leave any rule that ends in arithmetic unverifiable by any route."""
    from dynamicmultinets.compose import compose
    from dynamicmultinets.verify import fallible, verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 20, seed=5, name="d", a_digits=2, b_digits=2,
                    domain=ABSTRACT)
    m.label_data("d", "distributive_rewrite")
    compose(m.library, ["decimal_split", "distribute_symbolic", "eval_arith"],
            "value_route")

    assert not fallible(m.library.get("eval_arith"))
    assert not fallible(m.library.get("times_table_9"))

    rep = verify_against_rules(m.library, "value_route",
                               ["mul_by_definition", "eval_arith"],
                               m.datasets["d"], threshold=0.95)
    assert rep.n_checked == 20                      # it ran, rather than refusing
    assert rep.independence.shared_exact == ["eval_arith"]
    assert rep.independence.independent             # still counts as independent
    assert "contributes no error" in rep.summary()


def test_sharing_a_fallible_rule_is_still_refused():
    """A component that can be wrong puts its mistakes on both sides."""
    from dynamicmultinets.compose import compose
    from dynamicmultinets.rules import PythonRule
    from dynamicmultinets.verify import fallible, verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 20, seed=5, name="d", a_digits=2, b_digits=2,
                    domain=ABSTRACT)
    m.label_data("d", "distributive_rewrite")

    shaky = PythonRule("shaky", lambda c: c, ABSTRACT, ABSTRACT,
                       source="x->x", exact=False)
    shaky.trusted = True            # trusted, but not exact: it can be wrong
    m.library.add(shaky)
    assert fallible(shaky)
    compose(m.library, ["shaky", "eval_arith"], "under_test")

    with pytest.raises(ValueError, match="which can be wrong"):
        verify_against_rules(m.library, "under_test",
                             ["shaky", "mul_by_definition"], m.datasets["d"])


def test_collision_probability_separates_wide_from_narrow_answers():
    """Agreement is only worth something when the routes could have differed.
    Two unrelated rules choosing one of two actions agree half the time."""
    from dynamicmultinets.verify import collision_probability

    assert collision_probability([f"{i}0*55+{i}*55" for i in range(1, 90)]) < 0.01
    assert collision_probability(["a", "b"] * 45) > 0.4
    assert collision_probability(["only"]) == 1.0          # nothing to compare


def test_a_shared_training_oracle_defeats_independence():
    """Different weights are not independence. Two nets taught by the same
    oracle inherit its mistakes and can agree while both are wrong."""
    from dynamicmultinets.rules import PythonRule, Recipe
    from dynamicmultinets.verify import independence

    def make(name, oracle):
        r = PythonRule(name, lambda c: c, SPECIFIC, ABSTRACT, source=name)
        r.recipe = Recipe(oracle=oracle)
        return r

    spread = [f"{i}*7" for i in range(80)]
    same = independence(make("a", "distributive_rewrite"),
                        make("b", "distributive_rewrite"), spread)
    assert not same.independent
    assert same.shared_oracles == ["distributive_rewrite"]
    assert "taught by" in same.why_not()

    other = independence(make("a", "distributive_rewrite"),
                         make("b", "read_back"), spread)
    assert other.independent
    assert other.why_not() == ""


def test_the_construction_policy_agrees_with_the_moves_it_can_make():
    """`next_construction_step` decides when to stop by comparing the error
    against the size of a rotation, so the two numbers must be the same ones
    `SceneActionCodec` actually applies. If they drift, the loop stops where
    nothing was trained -- which is what left `triangle_180` unproved."""
    from dynamicmultinets.codec import SceneActionCodec
    from dynamicmultinets.oracles import _ANGLE_STEP, _ANGLE_TOL, _STEP

    codec = SceneActionCodec()
    assert (_STEP, _ANGLE_STEP) == (codec.step, codec.angle_step)
    # The construction must halt strictly inside what the reader accepts,
    # leaving a margin rather than finishing on the boundary.
    assert _ANGLE_STEP / 2.0 < _ANGLE_TOL


def test_a_finished_construction_is_one_the_reader_is_trained_on():
    """The scenes the generator calls solved must lie where the loop stops.
    They were exactly zero while the loop stopped anywhere within tolerance,
    so every finished construction was off-distribution."""
    from dynamicmultinets.generators import generate
    from dynamicmultinets.oracles import _ANGLE_STEP, _OFFSET_TOL, _geometry_state

    es = generate("triangle_scenes", 400, seed=17, solved_fraction=1.0)
    for ex in es.examples:
        sc = ex.inp.meta["scene"]
        through, parallel, err = _geometry_state(sc)
        assert through and parallel                      # genuinely solved
        assert abs(sc["line_offset"]) <= _OFFSET_TOL
        assert abs(err) <= _ANGLE_STEP / 2.0             # where the loop stops
    offsets = {round(ex.inp.meta["scene"]["line_offset"], 3) for ex in es.examples}
    assert len(offsets) > 50, "solved scenes must vary, not sit on one value"


def test_an_iterated_rule_keeps_the_best_cell_not_the_last():
    """The point of iterating inside a rule: a wrong step becomes one more
    candidate instead of the end of the proof, so the loop may run past the
    good cell as long as the judge picks it back out."""
    from dynamicmultinets.rules import IteratedRule, PythonRule

    step = PythonRule("grow", lambda c: Content.specific_text(c.text + "x"),
                      SPECIFIC, SPECIFIC, source="grow")
    judge = PythonRule("is_three",
                       lambda c: Content.abstract("yes" if len(c.text) == 3 else "no"),
                       SPECIFIC, ABSTRACT, source="is_three")
    loop = IteratedRule("grow_loop", step, judge, "yes", max_iters=5)

    out = loop.apply(Content.specific_text("a"))
    assert out.text == "axx"                  # the third cell, not the sixth
    assert out.meta["iterations"] == 5        # it kept going and came back


def test_an_iterated_rule_refuses_a_step_that_leaves_its_domain():
    """Only a rule that stays put can be run to a fixed point, and a judge has
    to read what the step writes."""
    from dynamicmultinets.rules import IteratedRule, PythonRule

    crossing = PythonRule("read", lambda c: Content.abstract(c.text),
                          SPECIFIC, ABSTRACT, source="read")
    same = PythonRule("edit", lambda c: c, SPECIFIC, SPECIFIC, source="edit")
    judge = PythonRule("j", lambda c: Content.abstract("yes"),
                       SPECIFIC, ABSTRACT, source="j")

    with pytest.raises(ValueError, match="stays in one domain"):
        IteratedRule("bad", crossing, judge, "yes")
    with pytest.raises(ValueError, match="reads"):
        IteratedRule("bad", same, crossing_judge := PythonRule(
            "k", lambda c: Content.abstract("yes"), ABSTRACT, ABSTRACT, source="k"),
            "yes")
    assert crossing_judge.domain_in == ABSTRACT


def test_an_exact_rule_offers_one_successor_whatever_the_beam():
    """The beam is for rules that choose. A rule that computes has one answer,
    and a wider search must not invent alternatives for it."""
    m = RenMachine()
    cell = Content.abstract("12*30")
    rule = m.library.get("decimal_split")
    assert len(rule.successors(cell, 1)) == 1
    assert len(rule.successors(cell, 5)) == 1
    assert rule.successors(cell, 5)[0][1] == 1.0


def test_a_rule_that_writes_a_conclusion_never_offers_its_runner_up():
    """A specific->specific rule proposes a drawing that perception must still
    read. A specific->abstract rule asserts, and nothing re-examines it -- so
    expanding its second choice would let a proof assert the very fact the
    rule's own perception rejected."""
    pytest.importorskip("torch")
    from dynamicmultinets.codec import ChoiceCodec, SceneActionCodec
    from dynamicmultinets.rules import NeuralRule

    proposes = NeuralRule("construct", SceneActionCodec(), SPECIFIC)
    asserts = NeuralRule("read_facts", ChoiceCodec(["a=b", "no_facts"]), SPECIFIC)

    assert proposes.domain_out == SPECIFIC and proposes.offers_alternatives()
    assert asserts.domain_out == ABSTRACT and not asserts.offers_alternatives()


# ---------------------------------------------------------------------------
# Proof search
# ---------------------------------------------------------------------------
def test_proof_requires_the_target_domain():
    """A picture of '47*83' is not a proof of the symbols '47*83'."""
    m = RenMachine()
    p = search(m.library, Content.specific_text("47*83"), "47*83",
               target_domain=ABSTRACT)
    assert not p.found


def test_proof_only_uses_trusted_rules():
    m = RenMachine()
    m.library.get("decimal_split").trusted = False
    p = m.prove("12*30", "10*30+2*30", max_depth=4)
    assert not p.found


def test_kept_proof_becomes_one_rule():
    m = RenMachine()
    p = m.prove("12*30", "10*30+2*30", max_depth=4)
    assert p.found and p.length == 2
    rule = m.keep_proof(p, "distribute_two_digit")
    assert rule.steps() == 2 and rule.trusted
    assert rule.apply(Content.abstract("47*30")).text == "40*30+7*30"


# ---------------------------------------------------------------------------
# Conciseness accounting
# ---------------------------------------------------------------------------
def test_objective_prices_description_and_derivation():
    m = RenMachine()
    m.add_task("value", "12*34", "408")
    rep = m.report()
    assert rep.solved["value"]
    assert rep.objective == pytest.approx(rep.total_bits + 1000 * rep.total_steps)


def test_unused_rules_are_reported_as_waste():
    m = RenMachine()
    m.add_task("value", "12*34", "408")
    rep = m.report()
    assert "times_table_9" in rep.unused          # eval_arith is cheaper
    actions, before, after = m.simplify(apply_changes=False)
    assert after is None                          # dry run must not mutate
    assert any(a.rule == "times_table_9" for a in actions)
    assert len(m.library) == len(before.per_rule_bits)


# ---------------------------------------------------------------------------
# Halting (section 5)
# ---------------------------------------------------------------------------
def test_sample_size_matches_the_paper():
    assert sample_size(0.02, 0.05) == int(np.ceil(np.log(20) / (2 * 0.02 ** 2)))


def test_threshold_grows_with_the_step_error():
    times = list(np.random.default_rng(0).integers(1, 400, size=4000))
    clean = calibrate(times, eps=0.05, lam=0.02, delta=0.05, sigma=0.0)
    noisy = calibrate(times, eps=0.05, lam=0.02, delta=0.05, sigma=1e-8)
    assert noisy.order_k >= clean.order_k        # less reliable steps => wait longer
    assert 1 <= clean.order_k <= clean.n_samples


def test_calibration_refuses_an_error_rate_it_cannot_absorb():
    times = list(range(1, 4001))
    with pytest.raises(ValueError, match="too large"):
        calibrate(times, sigma=0.5)


def test_sample_size_is_capped():
    """N grows as 1/lambda^2, so an innocuous-looking lambda asks for a sample
    no run can produce. The cap keeps the requirement reachable."""
    assert sample_size(1e-4, 0.05, None) > MAX_SAMPLES
    assert sample_size(1e-4, 0.05) == MAX_SAMPLES


def test_a_capped_calibration_reports_the_lambda_it_actually_bought():
    times = list(np.random.default_rng(0).integers(1, 400, size=MAX_SAMPLES))
    cal = calibrate(times, eps=0.05, lam=1e-4, delta=0.05)
    assert cal.lam > 1e-4                                  # not the one requested
    assert cal.lam == pytest.approx(effective_lambda(MAX_SAMPLES, 0.05))


def test_fitted_lambda_round_trips_through_sample_size():
    """Fitting lambda to a sample must not then demand a bigger sample -- the
    sqrt/square round trip is inexact and can land a hair under the true root."""
    for n in (2, 7, 100, 3745):
        for delta in (0.05, 0.5):
            lam = effective_lambda(n, delta)
            assert sample_size(lam, delta, None) <= n


def test_a_sample_too_small_for_any_guarantee_is_refused():
    """One program supports no lambda below 1 at delta=0.05, and a lambda of 1
    is a bound that says nothing -- refuse rather than return it."""
    with pytest.raises(ValueError, match="no lambda below 1"):
        effective_lambda(1, 0.05)


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------
def test_slot_codec_round_trips():
    codec = TextSlotCodec(num_slots=12)
    idx = codec.target(Content.abstract("10*30+2*30"))
    assert codec.decode(idx).text == "10*30+2*30"


def test_slot_codec_refuses_what_it_cannot_express():
    codec = TextSlotCodec(num_slots=4)
    with pytest.raises(ValueError, match="slots"):
        codec.target(Content.abstract("10*30+2*30"))
    with pytest.raises(ValueError, match="vocabulary"):
        TextSlotCodec(num_slots=12).target(Content.abstract("a&b"))


def test_choice_codec_round_trips():
    codec = ChoiceCodec(["direct", "+x", "-x"])
    assert codec.decode(codec.target(Content.abstract("+x", label="+x"))).text == "+x"


# ---------------------------------------------------------------------------
# Learned rules (torch)
# ---------------------------------------------------------------------------
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _torch_available(), reason="torch not installed")


@needs_torch
def test_rulenet_shapes_and_detournet_equivalence():
    import torch

    from dynamicmultinets.nets import RuleNet

    single = RuleNet(num_classes=19, num_slots=1)
    a = torch.randint(0, 256, (2, 3, 64, 192), dtype=torch.float32)
    assert single(a, a).shape == (2, 19)          # exactly DetourNet's head
    order, probs = single.rank(a, a)
    assert order.shape == (2, 19) and probs.shape == (2, 19)

    multi = RuleNet(num_classes=20, num_slots=16)
    assert multi(a, a).shape == (2, 16, 20)


@needs_torch
def test_quantizers_agree_between_numpy_and_torch():
    import torch

    from dynamicmultinets.nets import rgb_to_class_channels

    view = split_views(render_text("12*30+4"))[0]
    want = rgb_to_class_index(view)
    got = rgb_to_class_channels(
        torch.from_numpy(view.astype(np.float32)).permute(2, 0, 1)[None]
    ).argmax(dim=1)[0].numpy()
    assert np.array_equal(want, got)


@needs_torch
def test_a_rule_can_be_declared_trained_and_verified():
    """The whole loop on a task small enough to finish in seconds."""
    m = RenMachine()
    m.generate_data("rendered_expressions", 120, seed=0, name="d",
                    max_terms=1, digits=1)
    m.label_data("d", "read_back")
    m.declare_rule("reader", SPECIFIC, ABSTRACT, num_slots=3)
    report = m.train("reader", "d", epochs=2)
    assert report.n_train > 0 and report.dropped == []
    v = m.verify("reader", "d", "read_back", threshold=1.01)   # unreachable
    assert not v.became_trusted and v.grounding == "constructed"
    assert m.library.get("reader").stats.n_checked == len(m.datasets["d"].labeled)


@needs_torch
def test_ranking_a_rule_is_deterministic():
    """`rank_options` must run the net in eval mode. A freshly declared (or
    reloaded) rule is in TRAINING mode, where the trunk's dropout makes the
    ranked list a different list on every call -- and top-k is the number the
    planner acts on."""
    m = RenMachine(device="cpu")
    m.generate_data("robot_scenes", 6, seed=5, name="s")
    m.label_data("s", "best_escape_direction")
    rule = m.declare_rule("chooser", SPECIFIC, ABSTRACT,
                          from_oracle="best_escape_direction")
    assert rule.net.training                       # the condition being guarded
    cell = m.datasets["s"].examples[0].inp
    first = rule.rank_options(cell, 3)
    assert all(rule.rank_options(cell, 3) == first for _ in range(4))


@needs_torch
def test_scene_action_rule_declares_without_an_oracle():
    """`kind='scene_action'` documents a default action set; asking for it with
    no oracle and no classes must use that, not raise a bare KeyError."""
    m = RenMachine(device="cpu")
    rule = m.declare_rule("edit", SPECIFIC, SPECIFIC, kind="scene_action")
    assert rule.codec.actions == ["move_up", "rotate_cw", "rotate_ccw", "done"]


@needs_torch
def test_saved_library_reloads_with_recipes():
    import tempfile

    m = RenMachine()
    m.generate_data("rendered_expressions", 40, seed=0, name="d", max_terms=1, digits=1)
    m.label_data("d", "read_back")
    m.declare_rule("reader", SPECIFIC, ABSTRACT, num_slots=3)
    m.train("reader", "d", epochs=1)
    with tempfile.TemporaryDirectory() as tmp:
        m.save(tmp)
        lib = RuleLibrary.load(tmp)
    assert "reader" in lib
    assert lib.get("reader").recipe.oracle == "read_back"
    assert lib.get("reader").apply(Content.specific_text("3*4")) is not None


@needs_torch
def test_distilling_a_chain_collapses_its_steps():
    """A two-step derivation becomes one rule application. The distilled rule
    is a new empirical claim, so it must NOT inherit the chain's trust."""
    m = RenMachine()
    m.compose(["decimal_split", "distribute_symbolic"], "distribute_chain")
    assert m.library.get("distribute_chain").steps() == 2

    m.generate_data("mul_pairs", 40, seed=2, name="d", a_digits=2, b_digits=2,
                    domain=ABSTRACT)
    rule, report, agreement = m.distill("distribute_chain", "distribute_net", "d",
                                        epochs=1)
    assert rule.steps() == 1
    assert not rule.trusted
    assert 0.0 <= agreement <= 1.0
    assert rule.recipe.oracle == "rule:distribute_chain"


def test_simplify_never_trades_a_trusted_rule_for_an_untrusted_one():
    """`transcribe_unsafe` copies the caption, so on cells the machine drew it
    agrees with a real reader everywhere and costs less. Dropping the reader in
    its favour would delete the only rule that can read an OBSERVED cell."""
    from dynamicmultinets.rules import PythonRule

    m = RenMachine()
    reader = PythonRule("reader", lambda c: Content.abstract(c.text),
                        SPECIFIC, ABSTRACT, source="x" * 200)   # trusted, expensive
    m.library.add(reader)
    m.add_task("read_screen", "47*83", "47*83", domain=SPECIFIC, observed=True)
    m.generate_data("mul_pairs", 20, seed=99, name="probe", domain=SPECIFIC)

    actions, _, _ = m.simplify(probe="probe", apply_changes=False)
    assert not any(a.rule == "reader" for a in actions)


def test_simplify_leaves_a_working_library_when_a_drop_would_break_a_task():
    """One bad drop must not cascade: with the dropped rule gone every other
    rule stops being used, and a single pass would empty the library."""
    from dynamicmultinets.rules import PythonRule

    m = RenMachine()
    m.library.add(PythonRule("reader", lambda c: Content.abstract(c.text),
                             SPECIFIC, ABSTRACT, source="x" * 200))
    m.add_task("read_screen", "47*83", "47*83", domain=SPECIFIC, observed=True)
    m.generate_data("mul_pairs", 20, seed=99, name="probe", domain=SPECIFIC)

    _, before, after = m.simplify(probe="probe", apply_changes=True)
    assert after is not None
    assert all(after.solved.values())              # the task still passes
    assert "reader" in m.library                   # the rule solving it survived


def test_simplify_can_apply_and_refuses_to_lose_ground():
    m = RenMachine()
    m.add_task("value", "12*34", "408")
    before_n = len(m.library)
    actions, before, after = m.simplify(apply_changes=True)
    assert after is not None
    assert len(m.library) < before_n              # dead weight actually removed
    assert after.objective < before.objective     # and the objective improved
    assert all(after.solved.values())             # without losing a task
    assert "eval_arith" in m.library              # the rule doing the work stays


def test_composite_does_not_launder_trust():
    m = RenMachine()
    m.library.get("decimal_split").trusted = False
    chain = m.compose(["decimal_split", "distribute_symbolic"], "chain")
    assert not chain.trusted


# ---------------------------------------------------------------------------
# Controller plumbing
# ---------------------------------------------------------------------------
def test_scripted_controller_drives_the_tool_table():
    m = RenMachine()
    run = ScriptedController(m, log=lambda _s: None).run([
        ("inspect_machine", {}),
        ("write_tape", {"domain": ABSTRACT, "text": "12*30"}),
        ("apply_rule", {"rule": "decimal_split"}),
        ("finish", {"summary": "done"}),
    ])
    assert run.finished and len(run.tool_calls) == 4
    assert m.abstract.read().text == "(10+2)*30"


def test_tool_errors_come_back_as_text_not_exceptions():
    m = RenMachine()
    run = ScriptedController(m, log=lambda _s: None).run([
        ("apply_rule", {"rule": "no_such_rule"}),
        ("write_tape", {"domain": ABSTRACT, "text": "12*30"}),
        ("apply_rule", {"rule": "substitute_equalities"}),   # does not match
    ])
    assert run.transcript[0].startswith("ERROR")
    assert run.transcript[2].startswith("ERROR")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
