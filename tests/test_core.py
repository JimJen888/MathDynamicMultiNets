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


def test_the_caption_copier_is_an_ordinary_trusted_rule():
    """Copying a caption is lossless, so the baseline reader is exact, its
    confidence is 1.0, and it is trusted on the same terms as every other
    prior rule. It reads a cell the machine drew, in a proof, like anything
    else."""
    m = RenMachine()
    r = m.library.get("transcribe_unsafe")
    assert r.confidence() == 1.0 and r.trusted

    drawn = m.prove("47*83", "47*83", domain=SPECIFIC)
    assert drawn.found and drawn.rule_names() == ["transcribe_unsafe"]
    assert drawn.confidence == 1.0


def test_what_stops_the_copier_standing_in_for_perception_is_the_observed_flag():
    """The whole architectural claim now rests here rather than on a withheld
    trust flag, so it is measured rather than asserted: on a cell nothing wrote
    a caption for, a machine holding every prior rule and no reader can reach
    nothing at all."""
    m = RenMachine()
    assert not m.prove("47*83", "47*83", domain=SPECIFIC, observed=True).found
    assert not m.prove("12*30", "360", domain=SPECIFIC, observed=True).found

    # And it is the rule declining, not the search running out of room.
    seen = Content.specific_text("47*83")
    seen.meta["observed"] = True
    assert m.library.get("transcribe_unsafe").apply(seen) is None

    # Which is what keeps the benchmark's perception tasks unsolved: every one
    # of them starts from an observed cell.
    m.add_task("read_screen", "47*83", "47*83", domain=SPECIFIC, observed=True)
    m.add_task("screen_to_value", "12*30", "360", domain=SPECIFIC, observed=True)
    report = m.report()
    assert not any(report.solved.values())


def test_a_reader_checked_against_the_copier_is_graded_as_handwriting():
    """Verifying a reader against `transcribe_unsafe` IS the read_back check.
    It is worth having and it is the weakest grounding on the scale, so it must
    not come back dressed as an independent second route."""
    from dynamicmultinets.rules import PythonRule
    from dynamicmultinets.verify import GROUNDING_STRENGTH, verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 24, seed=3, name="probe", domain=SPECIFIC)
    reader = PythonRule("reader", lambda c: Content.abstract(c.text),
                        SPECIFIC, ABSTRACT, source="pixels->text", exact=False)
    m.library.add(reader)

    rep = verify_against_rules(m.library, "reader", ["transcribe_unsafe"],
                               m.datasets["probe"])
    assert rep.accuracy == 1.0                      # it agrees perfectly, of course
    assert rep.grounding == "constructed"
    assert not rep.independence.independent
    assert "own handwriting" in rep.independence.why_not()
    assert (GROUNDING_STRENGTH["constructed"]
            < GROUNDING_STRENGTH["independent_chain"])


def test_two_chains_sharing_only_the_copier_are_independent():
    """Sharing the caption copier is not sharing an answer. It is exact, so it
    hands both routes the same correct text -- which is the same input they
    were both given anyway -- and what is compared is what they do next. The
    argument is the one that already lets two routes both finish with
    `eval_arith`."""
    from dynamicmultinets.compose import compose
    from dynamicmultinets.verify import fallible, verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 24, seed=4, name="probe", domain=SPECIFIC,
                    a_digits=2, b_digits=2)
    copier = m.library.get("transcribe_unsafe")
    assert not fallible(copier)                 # exact, trusted, no errors

    # Read the drawing then evaluate the expression, against read the drawing
    # then multiply by the definition. The copier is the only thing in common;
    # the tails are an evaluator and repeated addition.
    compose(m.library, ["transcribe_unsafe", "eval_arith"], "drawn_value")
    rep = verify_against_rules(m.library, "drawn_value",
                               ["transcribe_unsafe", "mul_by_definition"],
                               m.datasets["probe"])
    assert rep.n_checked and rep.accuracy == 1.0
    assert rep.independence.shared_exact == ["transcribe_unsafe"]
    assert not rep.independence.shared_rules
    assert rep.independence.independent, rep.independence.why_not()
    assert rep.grounding == "independent_chain"


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
    # different search. On a cell the machine drew, the caption copier reads it
    # and the chain goes through; posed about an OBSERVED cell, which is the
    # version that needs perception, nothing can start it.
    drawn = Interconversion(name="t4", source="12*30", target="10*30+2*30",
                            domain=SPECIFIC)
    assert "transcribe_unsafe" in drawn.check(m, max_depth=6)
    seen = Interconversion(name="t5", source="12*30", target="10*30+2*30",
                           domain=SPECIFIC, observed=True)
    assert "no chain" in seen.check(m, max_depth=6)

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


def test_sharing_the_machine_s_basic_rules_never_defeats_independence():
    """The general form of the rule above: reading, writing, transcribing and
    the rest of the prior library are exact, so two routes may share any
    number of them and still be independent. What is compared is what they do
    that is NOT shared, and only a shared component that can be wrong puts the
    same mistake on both sides."""
    from dynamicmultinets.compose import compose
    from dynamicmultinets.verify import fallible, verify_against_rules

    m = RenMachine()
    assert not any(fallible(r) for r in m.library), (
        "a prior rule that can be wrong would change this policy: "
        + ", ".join(r.name for r in m.library if fallible(r)))

    m.generate_data("mul_pairs", 20, seed=7, name="d", a_digits=2, b_digits=2,
                    domain=ABSTRACT)

    # Draw it, read it back, then evaluate -- against draw it, read it back,
    # decompose, and evaluate. Three basic rules in common, including the write
    # rule and the read rule, and one real difference.
    compose(m.library, ["render", "transcribe_unsafe", "eval_arith"],
            "draw_and_value")
    rep = verify_against_rules(m.library, "draw_and_value",
                               ["render", "transcribe_unsafe", "decimal_split",
                                "distribute_symbolic", "eval_arith"],
                               m.datasets["d"])
    assert rep.n_checked and rep.accuracy == 1.0
    assert rep.independence.shared_exact == ["eval_arith", "render",
                                             "transcribe_unsafe"]
    assert not rep.independence.shared_rules
    assert rep.independence.independent, rep.independence.why_not()


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
# The Navier-Stokes construction, rebuilt as rules
# ---------------------------------------------------------------------------
def test_the_construction_rules_stay_out_of_every_other_machine():
    """A machine built for multiplication must not acquire the admissible
    stress cone because something else imported it earlier in the process."""
    from dynamicmultinets.navierstokes import install_navier_stokes_rules

    ns = RenMachine()
    install_navier_stokes_rules(ns.library)
    assert "lemma_4_5_cone" in ns.library
    assert "lemma_4_5_cone" not in RenMachine().library


def test_every_checked_result_starts_unverified_and_earns_its_trust():
    """The six results that reduce to a computation are declared as claims and
    are worth nothing until an oracle that answers the same question another
    way has agreed with them. Before that, no proof may use one."""
    from dynamicmultinets.navierstokes import (CLAIM_DATA, CLAIM_RULES,
                                               install_navier_stokes_rules)

    m = RenMachine()
    install_navier_stokes_rules(m.library)
    assert all(not m.library.get(r).trusted for r in CLAIM_RULES)
    assert not m.prove("h=1/200", "bounded_energy_unbounded_velocity,h=1/200").found

    for i, (rule, oracle) in enumerate(CLAIM_RULES.items()):
        generator, params = CLAIM_DATA[rule]
        m.generate_data(generator, 24, seed=400 + i, name=f"c{i}", **params)
        report = m.verify(rule, f"c{i}", oracle, threshold=0.99)
        assert report.accuracy == 1.0, (rule, report.counterexamples[:2])
        assert m.library.get(rule).trusted

    proof = m.prove("h=1/200", "bounded_energy_unbounded_velocity,h=1/200")
    assert proof.found and proof.rule_names()[-1] == "energy_budget_3_5"


def test_asking_for_more_instances_asks_more_questions():
    """A generator that saturates turns extra instances into repeats, and
    repeats do not raise confidence any more -- `verify` charges per
    distinct question. The one that mattered was the increment identity,
    whose generator emitted exactly two cells however many were asked for,
    so 160 checks were two facts and a Laplace value of 0.9938.
    """
    from dynamicmultinets.navierstokes import CLAIM_DATA, install_navier_stokes_rules
    from dynamicmultinets.nsderivation import (DERIVATION_CHECKS,
                                               install_derivation_rules)
    from dynamicmultinets.nsmechanisms import (MECHANISM_CHECKS,
                                               install_mechanism_rules)

    m = RenMachine()
    install_navier_stokes_rules(m.library)
    install_mechanism_rules(m.library)
    install_derivation_rules(m.library)

    sources = {r: (g, p) for r, (g, p) in CLAIM_DATA.items()}
    for rule, (_, gen) in {**MECHANISM_CHECKS, **DERIVATION_CHECKS}.items():
        sources.setdefault(rule, (gen, {}))

    for rule, (gen, params) in sources.items():
        data = m.generate_data(gen, 400, seed=5, name="q", **(params or {}))
        distinct = len({e.inp.text for e in data})
        # 400 is the smallest count any check in the run uses, and no
        # generator may be answering fewer than four fifths of them freshly.
        assert distinct >= 320, (rule, gen, distinct)


# ---------------------------------------------------------------------------
# Deciding a claim instead of sampling it
# ---------------------------------------------------------------------------
def test_linear_arithmetic_decides_and_does_not_sample():
    """`least` and `short` are `min` and `<` on numbers, so rule code that
    calls them reads and behaves the same. On linear forms they defer, and
    the whole conjunction is then decided by reading coefficients."""
    from fractions import Fraction

    from dynamicmultinets.linarith import Lin, least, obligations, short, solve

    assert least(3, 1, 2) == 1
    assert short(1, 2) is True and short(2, 1) is False

    n = Lin.var("n")
    # A form that decreases in an unbounded variable cannot hold throughout.
    with obligations() as obs:
        short(Lin(Fraction(5)) - n, Lin(Fraction(0)), "decreasing")
    assert not solve(obs).feasible

    # One that does not decrease holds, and a parameter is solved for
    # rather than quantified: 2/5 - 4k >= 0 is k <= 1/10.
    k = Lin.var("k")
    with obligations() as obs:
        short(n + Fraction(2, 5) - 4 * k, Lin(Fraction(0)), "budget")
    region = solve(obs, parameter="k")
    assert region.feasible and region.upper == Fraction(1, 10)
    assert region.lower is None
    assert region.contains(Fraction(1, 100)) and not region.contains(Fraction(1, 5))


def test_symbolic_values_cannot_be_compared_outside_a_proof():
    """`short` answering False is only sound inside a scope that records
    what it assumed. Reached anywhere else it would be a silent lie."""
    from dynamicmultinets.linarith import Lin, short

    with pytest.raises(RuntimeError):
        short(Lin.var("n"), 0)


def test_polynomial_algebra_settles_a_pointwise_identity():
    """The identity `div(f tensor f) = f.grad f + f div f` is algebra among
    the field and its first derivatives at a point, so expanding it in
    independent symbols settles every smooth field at once."""
    from dynamicmultinets.symalg import (Poly, advect, divergence, jacobian,
                                         tensor_divergence, vector)

    f, jf = vector("f"), jacobian("f")
    left = tensor_divergence(f, jf)
    right = [advect(f, jf)[i] + f[i] * divergence(jf) for i in range(3)]
    assert all((left[i] - right[i]).is_zero for i in range(3))

    # And it is not vacuous: drop the second term and it stops holding.
    assert not (left[0] - advect(f, jf)[0]).is_zero
    assert Poly.sym("a") * Poly.sym("b") == Poly.sym("b") * Poly.sym("a")
    assert (Poly.constant(2) * Poly.sym("a") - Poly.sym("a")
            - Poly.sym("a")).is_zero


def test_the_increment_identity_is_proved_not_measured():
    """Section 3.3 is what Sections 7 and 9 rest on, and its derivation is
    four lines of algebra. Expanding it in the one-jet proves it for every
    point of every smooth field, and leaves the defect in closed form."""
    from dynamicmultinets.nsderivation import install_derivation_rules

    m = RenMachine()
    install_derivation_rules(m.library)
    report = m.prove_rule("increment_identity",
                          "increment_identity_by_polynomial_algebra")
    assert report.established and report.judgement.whole_domain
    assert "-w div(w)" in report.judgement.statement
    rule = m.library.get("increment_identity")
    assert rule.exact and rule.trusted and rule.proved
    assert rule.confidence() == 1.0

    # A rule that asserted the split unconditionally is refused: the
    # algebra says the defect is there, so a rule that cannot see it is
    # not the rule that was proved.
    m2 = RenMachine()
    install_derivation_rules(m2.library)
    broken = m2.library.get("increment_identity")
    broken.fn = lambda c: Content.abstract("residual_splits_exactly")
    refused = m2.prove_rule("increment_identity",
                            "increment_identity_by_polynomial_algebra")
    assert not refused.established and not broken.exact


def test_the_stage_induction_is_proved_and_its_budget_derived():
    """Proposition 9.6 quantifies over every correction stage, which no
    number of instances reaches. It also quantifies over a rational
    recursion on an integer index, which is decidable -- so this one claim
    is settled by a decision procedure, and the loss the cycle can absorb
    comes out of the same inequalities instead of being copied in."""
    from fractions import Fraction

    from dynamicmultinets.nsderivation import (install_derivation_rules,
                                               prove_cycle_closes_at_every_stage)

    m = RenMachine()
    install_derivation_rules(m.library)

    proof = prove_cycle_closes_at_every_stage(m.library)
    assert proof.base_ok and proof.landed and proof.budget.feasible
    assert proof.budget.upper == Fraction(1, 10)
    assert proof.closes_at(Fraction(1, 100_000))
    assert not proof.closes_at(Fraction(1, 5))

    report = m.prove_rule("prop_9_6_all_stages",
                          "stage_induction_by_linear_arithmetic")
    assert report.established and report.judgement.whole_domain
    rule = m.library.get("prop_9_6_all_stages")
    assert rule.exact and rule.trusted and rule.proved
    assert rule.confidence() == 1.0
    # A proved rule is not an accuracy over instances and must not read
    # like one anywhere.
    assert "PROVED" in m.library.table()


def test_a_proof_fails_when_the_rule_or_the_cycle_is_wrong():
    """The guard against the expensive mistake. A prover that certified
    whatever it was handed would be worse than no prover at all, so both
    halves are broken on purpose here: the rule's threshold, and the moves
    the threshold is supposed to describe."""
    from fractions import Fraction

    from dynamicmultinets.linarith import least
    from dynamicmultinets.nsderivation import (install_derivation_rules,
                                               prove_cycle_closes_at_every_stage)

    # 1. The rule claims a budget the cycle does not have.
    m = RenMachine()
    install_derivation_rules(m.library)
    rule = m.library.get("prop_9_6_all_stages")
    rule.fn = lambda c: Content.abstract("cycle_closes_at_every_stage")
    report = m.prove_rule("prop_9_6_all_stages",
                          "stage_induction_by_linear_arithmetic")
    assert not report.established
    assert "threshold" in report.judgement.obstruction
    assert not rule.exact and not rule.proved

    # 2. A cycle whose first move gains nothing closes at no stage at all,
    #    and the prover says so rather than deriving a budget.
    m2 = RenMachine()
    install_derivation_rules(m2.library)
    op1 = m2.library.get("cycle_op1_harmonics")

    def gains_nothing(f):
        f["B"], f["C"], f["S"] = f["B0"], f["C0"], f["C0"]
        f["step"] = Fraction(1)
        return f

    op1.update = gains_nothing
    assert not prove_cycle_closes_at_every_stage(m2.library).budget.feasible

    # 3. A cheaper stress correction is a different budget, and the derived
    #    number follows the moves rather than the docstring.
    m3 = RenMachine()
    install_derivation_rules(m3.library)
    op2 = m3.library.get("cycle_op2_stress")

    def cheaper(f):
        k, b0, c0 = f["k"], f["B0"], f["C0"]
        f["B"] = least(f["B"], b0 + Fraction(1, 2) - k, b0 + Fraction(2, 5) - k,
                       b0 + Fraction(1, 2) - 2 * k, 2 * b0 - 3 * k)
        f["C"] = c0 - 2 * k
        f["S"] = c0 - 2 * k
        f["step"] = Fraction(2)
        return f

    op2.update = cheaper
    widened = prove_cycle_closes_at_every_stage(m3.library)
    assert widened.budget.feasible and widened.budget.upper > Fraction(1, 10)


# ---------------------------------------------------------------------------
# Statements about function space norms, by derivation
# ---------------------------------------------------------------------------
def test_a_norm_estimate_chains_to_a_statement_about_the_whole_function():
    """An assumed band estimate is carried to a limit statement by rules.

    Two of the quantifiers the Navier-Stokes run calls out of reach are
    discharged on the way, and neither by instances: the dyadic sum is
    decided by the sign of its exponent, and so is the limit.
    """
    from fractions import Fraction

    from dynamicmultinets.normcalc import (RESIDUAL_START, RESIDUAL_TARGET,
                                           assumptions_behind, est,
                                           install_norm_rules)

    m = RenMachine()
    install_norm_rules(m.library)

    proof = m.prove(RESIDUAL_START, RESIDUAL_TARGET, max_depth=6)
    assert proof.found and not proof.unmeasured
    assert proof.rule_names() == ["bernstein_uniform", "sum_over_bands",
                                  "limit_at_the_singularity"]

    # The exactness is only as good as what it leaned on, and the chain
    # has to be able to say what that was.
    leaned = assumptions_behind(m.library, proof.rule_names())
    assert any("Bernstein" in a for a in leaned)
    assert any("geometric series" in a for a in leaned)

    # A divergent dyadic sum stops the search rather than passing.
    diverges = est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(1),
                   sc=Fraction(1, 5))
    assert not m.prove(diverges, RESIDUAL_TARGET, max_depth=6).found

    # So does a bound that grows at the singularity instead of vanishing.
    grows = est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-3),
                sc=Fraction(-1, 5))
    assert not m.prove(grows, RESIDUAL_TARGET, max_depth=6).found


def test_the_band_exponent_is_derived_from_scaling_not_quoted():
    """`d(1/p - 1/r)` is forced by the inequality surviving a rescaling of
    the function, and the prover rearranges that equation rather than
    trusting a table. It runs the rule's own arithmetic, so a rule with the
    wrong exponent is refused."""
    from fractions import Fraction

    from dynamicmultinets.normcalc import install_norm_rules

    m = RenMachine()
    install_norm_rules(m.library)
    for name in ("bernstein_uniform", "bernstein_to_energy",
                 "bernstein_derivative"):
        report = m.prove_rule(name, "band_exponents_by_scaling")
        assert report.established, (name, report.judgement.obstruction)
        assert m.library.get(name).proved

    # Two powers of 1/p instead of three, in the arithmetic the rule runs.
    m2 = RenMachine()
    install_norm_rules(m2.library)
    rule = m2.library.get("bernstein_uniform")

    def wrong(f):
        f["fr"] = f["fr"] + 2 * f["ip"]
        f["ip"] = Fraction(0)
        return f

    rule.shift = wrong
    refused = m2.prove_rule("bernstein_uniform", "band_exponents_by_scaling")
    assert not refused.established and not rule.proved
    assert "2ip" in refused.judgement.obstruction

    # A summation guard that lets a divergent series through is refused.
    m3 = RenMachine()
    install_norm_rules(m3.library)
    loose = m3.library.get("sum_over_bands")
    loose.fn = lambda c: Content.abstract("glob(d=3,dv=0,ip=0,sc=1/5)")
    assert not m3.prove_rule("sum_over_bands",
                             "summation_and_limit_by_sign").established


def test_holder_adds_the_indices_and_refuses_when_it_cannot():
    """The quadratic term needs a product estimate, and the integrability
    index of a product is the sum of the two. Past one it is not a
    Lebesgue index any more and the rule declines."""
    from fractions import Fraction

    from dynamicmultinets.normcalc import est, install_norm_rules, pair

    m = RenMachine()
    install_norm_rules(m.library)
    half = est(d=3, dv=0, ip=Fraction(1, 2), fr=Fraction(-2), sc=Fraction(1, 10))
    got = m.library.get("holder_product").apply(Content.abstract(pair(half, half)))
    assert got is not None
    assert got.text == est(d=3, dv=0, ip=Fraction(1), fr=Fraction(-4),
                           sc=Fraction(1, 5))

    steep = est(d=3, dv=0, ip=Fraction(3, 4), fr=Fraction(-2), sc=Fraction(1, 10))
    assert m.library.get("holder_product").apply(
        Content.abstract(pair(steep, steep))) is None


def test_the_theorem_is_reachable_only_through_the_imported_estimates():
    """Every result in the paper is a rule here, so the derivation of Theorem
    1.1 is a chain the machine can find. Nine of its steps are estimates on
    function spaces that nothing here checked, so the chain exists and is not
    a proof: trusted-only search refuses it, and allowing the imports in
    reports a confidence that is the product of their Laplace priors."""
    from dynamicmultinets.navierstokes import (IMPORTED_NAMES,
                                               install_navier_stokes_rules)
    from dynamicmultinets.nsderivation import (DERIVATION_IMPORTED,
                                               install_derivation_rules)

    m = RenMachine()
    install_navier_stokes_rules(m.library)
    install_derivation_rules(m.library)
    everything = tuple(IMPORTED_NAMES) + DERIVATION_IMPORTED
    assert all(not m.library.get(n).trusted for n in everything)

    assert not m.prove("h=1/200", "theorem_1_1_forced_blowup", max_depth=24).found
    allowed = m.prove("h=1/200", "theorem_1_1_forced_blowup", max_depth=24,
                      trusted_only=False)
    assert allowed.found
    imported = [n for n in allowed.rule_names() if n in everything]
    assert len(imported) == len(everything)
    assert allowed.confidence <= 0.5 ** len(imported)

    # The derivation is not a run of citations: it passes through the four
    # moves of a correction cycle and the wave increment, all of which the
    # machine applies itself.
    for move in ("apply_wave_increment", "cycle_op1_harmonics",
                 "cycle_op4_moments"):
        assert move in allowed.rule_names()


def test_the_scaling_exponents_are_the_ones_the_construction_needs():
    """A+D=1 is what keeps every transport product at the same power of q, and
    the energy budget closes exactly when h < 1/6. Both are exact rational
    arithmetic, so the rules decide them rather than estimating them."""
    from fractions import Fraction

    from dynamicmultinets.navierstokes import install_navier_stokes_rules

    m = RenMachine()
    install_navier_stokes_rules(m.library)

    # Ecore = 1/2-3h and Dcore = -1/2-3h fail together, both exactly at
    # h = 1/6, which is why the paper states that one inequality rather than
    # two; past it the rule reports whichever it tests first.
    for h, expected in ((Fraction(1, 200), "bounded_energy_unbounded_velocity"),
                        (Fraction(1, 5), "energy_does_not_vanish")):
        cell = Content.abstract(f"h={h}")
        for name in ("ns_exponents", "ns_core_scales", "ns_core_energy"):
            cell = m.library.get(name).apply(cell)
        assert f"Ecore={Fraction(1, 2) - 3 * h}" in cell.text
        assert f"Dcore={-Fraction(1, 2) - 3 * h}" in cell.text
        verdict = m.library.get("energy_budget_3_5").apply(cell)
        assert verdict.text.startswith(expected)

    # h = 1/5 exceeds 1/6, so the dissipation integral diverges and Theorem
    # 4.6's profiles are unavailable: the derivation stops at the arithmetic.
    assert m.library.get("thm_4_6_profiles").apply(
        Content.abstract("bounded_energy_unbounded_velocity,h=1/5")) is None


def test_each_imported_estimate_has_its_method_carried_out():
    """The nine steps the machine cannot check as stated are not left as
    citations: seven of them have a mechanism here that is built and run.
    Each mechanism is declared unverified and has to agree with an oracle
    that carries the construction out."""
    from dynamicmultinets.navierstokes import (IMPORTED_NAMES,
                                               install_navier_stokes_rules)
    from dynamicmultinets.nsderivation import (DERIVATION_IMPORTED,
                                               install_derivation_rules)
    from dynamicmultinets.nsmechanisms import (MECHANISM_CHECKS, SUPPORTS,
                                               install_mechanism_rules)

    m = RenMachine()
    install_navier_stokes_rules(m.library)
    install_mechanism_rules(m.library)
    install_derivation_rules(m.library)
    assert set(SUPPORTS) == set(IMPORTED_NAMES) | set(DERIVATION_IMPORTED)
    assert all(not m.library.get(r).trusted for r in MECHANISM_CHECKS)

    for i, (rule, (oracle, generator)) in enumerate(MECHANISM_CHECKS.items()):
        m.generate_data(generator, 10, seed=900 + i, name=f"mech{i}")
        report = m.verify(rule, f"mech{i}", oracle, threshold=0.99)
        assert report.accuracy == 1.0, (rule, report.counterexamples[:2])
        assert m.library.get(rule).trusted


def test_carrying_out_the_methods_does_not_make_the_estimates_trusted():
    """The point of the split. A mechanism decides an instance; the step it
    speaks for asserts something uniform in the scale, the band and the
    stage. Verifying every mechanism must leave every imported step
    untrusted and Theorem 1.1 out of reach, or the machine would be claiming
    a proof it does not have."""
    from dynamicmultinets.navierstokes import (IMPORTED_NAMES,
                                               install_navier_stokes_rules)
    from dynamicmultinets.nsmechanisms import (MECHANISM_CHECKS,
                                               install_mechanism_rules)

    m = RenMachine()
    install_navier_stokes_rules(m.library)
    install_mechanism_rules(m.library)
    for i, (rule, (oracle, generator)) in enumerate(MECHANISM_CHECKS.items()):
        m.generate_data(generator, 8, seed=950 + i, name=f"m{i}")
        m.verify(rule, f"m{i}", oracle, threshold=0.99)

    assert all(not m.library.get(n).trusted for n in IMPORTED_NAMES)
    assert not m.prove("h=1/200", "theorem_1_1_forced_blowup", max_depth=24).found


def test_a_chain_checked_as_a_chain_is_not_priced_by_its_parts():
    """Keeping a derivation as one rule and then checking it is the whole
    point of composing. Before the check, the chain is worth the product of
    what is known about its steps, which for unverified steps is the prior
    1/2 each. After forty agreements with an oracle outside it, the chain's
    own measurement is the better answer and the compounded prior is not."""
    from dynamicmultinets.compose import compose
    from dynamicmultinets.rules import PythonRule

    m = RenMachine()
    for name in ("shaky_a", "shaky_b"):
        r = PythonRule(name, lambda c: Content.abstract(c.text),
                       ABSTRACT, ABSTRACT, source="x->x", exact=False)
        r.trusted = False
        m.library.add(r)
    chain = compose(m.library, ["shaky_a", "shaky_b"], "chained")
    assert chain.confidence() == 0.25 and not chain.trusted

    chain.stats.merge(40, 40, "an oracle outside the chain", [])
    assert chain.confidence() > 0.9
    assert not chain.trusted          # measuring is not the same as trusting

    # A chain of exact rules keeps its 1.0 rather than being pulled down to
    # the Laplace value of however many times it happened to be checked.
    exact = compose(m.library, ["decimal_split", "distribute_symbolic"], "value")
    exact.stats.merge(4, 4, "a small probe", [])
    assert exact.confidence() == 1.0


def test_a_chain_reports_unmeasured_steps_as_a_count_not_a_probability():
    """Eleven never-checked steps compound to 0.0005, which reads like a
    verdict of near-certain failure when what it records is that nothing
    has looked. A proof separates the two: a confidence over the steps
    that have evidence, and a count of the steps that do not."""
    from dynamicmultinets.rules import PythonRule

    m = RenMachine()
    for i in range(3):
        r = PythonRule(f"leg{i}", (lambda i: lambda c: Content.abstract(
            f"s{i + 1}") if c.text == f"s{i}" else None)(i),
            ABSTRACT, ABSTRACT, source=f"s{i}->s{i + 1}", exact=False)
        r.trusted = False
        m.library.add(r)
    p = m.prove("s0", "s3", max_depth=4, trusted_only=False)
    assert p.found and p.length == 3
    assert p.confidence == pytest.approx(0.125)      # the bare product
    assert p.unmeasured == 3
    assert "nothing measured, 3 unmeasured steps" in p.evidence()

    # Check one leg and it stops being a count.
    m.library.get("leg1").stats.merge(100, 100, "an oracle", [])
    p = m.prove("s0", "s3", max_depth=4, trusted_only=False)
    assert p.unmeasured == 2
    assert p.measured_confidence == pytest.approx(101 / 102)


def test_exactness_counts_as_measured():
    """An exact rule's 1.0 is not an estimate, so a chain of them has
    nothing unmeasured about it and reports one number."""
    m = RenMachine()
    assert m.library.get("decimal_split").measured()
    assert not m.library.get("transcribe_unsafe").measured() or \
        m.library.get("transcribe_unsafe").confidence() >= 1.0


# ---------------------------------------------------------------------------
# Proof search
# ---------------------------------------------------------------------------
def test_proof_requires_the_target_domain():
    """A picture of '47*83' is not a proof of the symbols '47*83'.

    The cell's caption says '47*83' and the target is '47*83', so a goal test
    that ignored the domain would call this proved before any rule ran. It has
    to be CROSSED instead, by a rule, and on a cell nothing captioned there is
    no rule that can.
    """
    m = RenMachine()
    p = search(m.library, Content.specific_text("47*83"), "47*83",
               target_domain=ABSTRACT)
    assert p.found and p.length == 1        # crossed, not relabelled
    assert p.steps[0].domain_in == SPECIFIC and p.steps[0].domain_out == ABSTRACT

    seen = Content.specific_text("47*83")
    seen.meta["observed"] = True
    assert not search(m.library, seen, "47*83", target_domain=ABSTRACT).found


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
    labeled = len(m.datasets["d"].labeled)
    assert v.n_checked == labeled
    # 120 draws from one-digit products repeat, and the renderer is
    # deterministic, so a repeat is the same pixels and the same question.
    # The rule's confidence is charged for the questions, not the draws.
    assert 0 < v.n_distinct < labeled
    assert m.library.get("reader").stats.n_checked == v.n_distinct


def test_repeats_are_not_evidence():
    """A generator that asks one question a hundred times has asked once.

    Confidence is Laplace-smoothed over the number of checks, so counting
    repeats would let a two-case generator drive a rule to 0.995 on the
    strength of two facts. This is the guard for that.
    """
    from dynamicmultinets.dataset import Example, ExampleSet

    m = RenMachine()
    es = ExampleSet(name="repeats", examples=[
        Example(inp=Content.abstract("2+3")) for _ in range(100)])
    m.datasets.put(es)
    report = m.verify("eval_arith", "repeats", "arith_value", threshold=0.99)
    assert report.n_checked == 100
    assert report.n_distinct == 1
    assert m.library.get("eval_arith").stats.n_checked == 1
    assert "NARROW" in report.summary()


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


# ---------------------------------------------------------------------------
# The RH criteria (examples/run_riemann.py)
# ---------------------------------------------------------------------------
def test_divisor_sum_matches_values_known_independently():
    from dynamicmultinets.prior import sigma

    assert sigma(1) == 1
    assert sigma(6) == 12 and sigma(28) == 56          # perfect numbers
    assert sigma(5040) == 19344                        # 31*13*6*8


def test_robin_criterion_finds_exactly_the_known_exceptional_set():
    """Robin's inequality fails at 27 integers and nowhere else below 5040.

    n=2 is in the published list and absent here on purpose: ln ln 2 is
    negative, so the ratio's sign flips and the comparison would not mean what
    it means everywhere else. `robin_ratio` declines n<3 rather than emit it.
    """
    import math

    from dynamicmultinets.prior import EXP_GAMMA, sigma

    fails = {n for n in range(3, 20001)
             if sigma(n) / (n * math.log(math.log(n))) >= EXP_GAMMA}
    assert fails == {3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 30, 36, 48, 60,
                     72, 84, 120, 180, 240, 360, 720, 840, 2520, 5040}


def test_the_two_criteria_agree_above_5040_and_diverge_only_below():
    """Two independent routes to the same conclusion, which is the point of
    keeping both: a disagreement above 5040 would mean one of them is wrong."""
    m = RenMachine(with_prior=True)
    lib = m.library

    def verdicts(n):
        s = lib.get("divisor_sum").apply(Content.abstract(str(n)))
        robin = lib.get("robin_decide").apply(lib.get("robin_ratio").apply(s))
        return robin.text == "robin_holds", \
            lib.get("lagarias_decide").apply(s).text == "lagarias_holds"

    assert all(verdicts(n)[0] == verdicts(n)[1] for n in range(5041, 7000))
    below = [n for n in range(3, 5041) if verdicts(n)[0] != verdicts(n)[1]]
    assert len(below) == 26 and below[0] == 3 and below[-1] == 5040


def test_the_chain_from_an_integer_to_a_verdict_is_exact():
    m = RenMachine(with_prior=True)
    p = m.prove("5040", "robin_fails", max_depth=5)
    assert p.found and p.confidence == 1.0        # every link is a PythonRule
    assert m.prove("10080", "robin_holds", max_depth=5).found


def test_robin_verdict_above_5040_is_a_vacuous_verification_target():
    """The reason `robin_cases` exists. A test set of integers above 5040
    carries one label, so any constant answer scores 1.000 on it and the
    accuracy reports the base rate rather than the rule."""
    from collections import Counter

    from dynamicmultinets import generators, oracles
    from dynamicmultinets.verify import answer_text

    flat = generators.generate("rendered_integers", 80, seed=11,
                               min_digits=4, max_digits=5, above=5040)
    oracles.label(flat, "robin_verdict")
    assert len(Counter(answer_text(e.out) for e in flat.examples if e.labeled)) == 1

    balanced = generators.generate("robin_cases", 60, seed=3)
    oracles.label(balanced, "robin_verdict")
    counts = Counter(answer_text(e.out) for e in balanced.examples if e.labeled)
    assert set(counts) == {"robin_holds", "robin_fails"}
    assert max(counts.values()) / sum(counts.values()) < 0.7


def test_the_divisor_chain_cannot_tell_that_a_cell_is_not_an_integer():
    """Guards the claim run_riemann.py makes about why the universal statement
    must not be posed as a cell: the chain declines a non-numeric cell only
    because `divisor_sum` pattern-matches digits, and anything a reader turns
    into digits gets a verdict regardless of what was drawn."""
    m = RenMachine(with_prior=True)
    assert m.library.get("divisor_sum").apply(Content.abstract("zeta(s)=0")) is None
    # ...but any digits at all are accepted, with no notion of whether they
    # were a faithful reading of the drawing.
    got = m.library.get("divisor_sum").apply(Content.abstract("847213"))
    assert got is not None and got.text.startswith("sigma(847213)=")


# ---------------------------------------------------------------------------
# Zeta zeros and level spacings (examples/run_montgomery.py)
# ---------------------------------------------------------------------------
def test_riemann_siegel_finds_the_published_zeros():
    """Checked against values computed by a different method entirely.

    The tolerance is the measured accuracy of the leading-correction
    Riemann-Siegel formula at low t, where it is at its worst: 7.5e-3 absolute
    at the third zero, falling to ~1e-4 by the thousandth as the asymptotic
    formula comes into its own. What matters for this package is the error in
    MEAN-SPACING units, since that is what the histograms are built from --
    7.5e-3 at t=25 is 1.7e-3 of a mean spacing, against a histogram bin width
    of 0.125.
    """
    from dynamicmultinets.zeta import zeta_zeros

    published = np.array([14.134725142, 21.022039639, 25.010857580,
                          30.424876126, 32.935061588, 37.586178159])
    err = np.abs(zeta_zeros(6) - published)
    assert err.max() < 1e-2
    # and the error in mean-spacing units, which is the one that could matter
    density = np.log(published / (2 * np.pi)) / (2 * np.pi)
    assert float((err * density).max()) < 5e-3


def test_the_zero_scan_does_not_skip_any():
    """Zeros skipped IN PAIRS leave the sign pattern intact, so a scan that
    steps over two of them looks perfectly healthy. The Riemann-von Mangoldt
    count is the independent check that catches it."""
    from dynamicmultinets.zeta import zero_counting_function, zeta_zeros

    z = zeta_zeros(400)
    assert abs(zero_counting_function(float(z[-1])) - 400) < 1.0


def test_unfolding_removes_the_logarithmic_trend():
    from dynamicmultinets.zeta import unfolded_spacings, zeta_zeros

    s = unfolded_spacings(zeta_zeros(600))
    # Mean spacing 1 is what unfolding is FOR, and it holds asymptotically
    # rather than exactly: N(t) is the smooth part of the counting function, so
    # over a finite block the mean comes out near 1, not at it. Measured at
    # 1.0004 over 600 zeros.
    assert abs(float(s.mean()) - 1.0) < 3e-3
    assert float(s.min()) > 0.0
    # the trend really is removed: raw gaps shrink with height, unfolded do not
    raw = np.diff(zeta_zeros(600))
    assert raw[:100].mean() > 1.5 * raw[-100:].mean()
    assert 0.8 < s[:100].mean() / s[-100:].mean() < 1.25


def test_the_ensembles_are_actually_distinguishable():
    """If GUE and Poisson spacings did not separate at small s, the rule in
    run_montgomery.py would be learning noise and scoring on it."""
    from dynamicmultinets.render import spacing_scene
    from dynamicmultinets.zeta import ensemble_spacings

    rng = np.random.default_rng(0)
    gue = np.array(spacing_scene(ensemble_spacings("gue", 4000, rng))["hist"])
    poisson = np.array(spacing_scene(ensemble_spacings("poisson", 4000, rng))["hist"])
    # level repulsion: GUE almost never puts two levels on top of each other.
    assert gue[0] < 0.15 and poisson[0] > 0.5


def test_zeta_cells_carry_no_ensemble_label_and_cannot_be_verified():
    """The property the whole conjecture rests on. If `spacing_ensemble` ever
    labelled a zeta cell, a verification number could be produced for the open
    question and would look exactly like the verified ones beside it."""
    from dynamicmultinets import generators, oracles

    zs = generators.generate("zeta_spacings", 2, seed=0, n_spacings=200)
    oracles.label(zs, "spacing_ensemble")
    assert all(not ex.labeled for ex in zs.examples)

    es = generators.generate("spacing_histograms", 3, seed=0, n_spacings=200)
    oracles.label(es, "spacing_ensemble")
    assert all(ex.labeled for ex in es.examples)
    # ...and the cell's own caption must not give the answer away, because
    # `transcribe_unsafe` copies captions.
    assert not any(k in es.examples[0].inp.text for k in ("gue", "goe", "poisson"))


def test_both_kinds_of_cell_go_through_one_reduction():
    """Ensemble cells and zeta cells must be binned by identical code, or any
    difference found later could be an artefact of the reduction."""
    from dynamicmultinets import generators
    from dynamicmultinets.render import SPACING_BINS

    a = generators.generate("spacing_histograms", 1, seed=0, n_spacings=200)
    b = generators.generate("zeta_spacings", 1, seed=0, n_spacings=200)
    for es in (a, b):
        scene = es.examples[0].inp.meta["scene"]
        assert scene["kind"] == "spacing"
        assert len(scene["hist"]) == SPACING_BINS == len(scene["cdf"])
        assert es.examples[0].inp.image.shape == (64, 384, 3)


def test_the_adaptive_scan_does_not_lose_close_pairs():
    """The bug this replaced was silent AND biased.

    A fixed step of 0.05 lost ~9 zeros in 80000 by t~60000, because the mean
    gap shrinks like 1/log(t) while the step did not. Skipped zeros come in
    pairs, so Z's sign pattern stays consistent and nothing complains -- and
    what gets skipped is the CLOSEST pairs, which depletes small spacings and
    imitates level repulsion. Any measurement of a deviation from GUE would
    have inherited it.
    """
    from dynamicmultinets.zeta import scan_step, zeta_zeros, zero_counting_function

    assert scan_step(60000.0) < scan_step(100.0)          # adapts downward
    z = zeta_zeros(3000)
    assert abs(zero_counting_function(float(z[-1])) - 3000) < 2.0


def test_the_finite_height_deviation_transfers_to_unseen_heights():
    """The claim of examples/run_finite_height.py, at small scale.

    A shape fitted on low bands must predict higher bands it never saw, and it
    must do so BECAUSE of its shape: scrambling the bins across s keeps the
    magnitude and destroys the structure, and has to make the fit worse.
    """
    from dynamicmultinets.render import SPACING_BINS, SPACING_MAX
    from dynamicmultinets.zeta import (ensemble_spacings, unfolded_spacings,
                                       zeta_zeros)

    ds = SPACING_MAX / SPACING_BINS
    rng = np.random.default_rng(0)

    def dens(x):
        h, _ = np.histogram(x, bins=SPACING_BINS, range=(0.0, SPACING_MAX),
                            density=True)
        return h

    def l1(x):
        return float(np.abs(x).sum() * ds)

    p_gue = dens(np.concatenate([ensemble_spacings("gue", 120000, rng, dim=60)
                                 for _ in range(2)]))
    # 12000 zeros over 4 bands is 3000 spacings each, and that is not an
    # arbitrary size: the deviation is ~2x the per-band noise floor, so below
    # ~3000 it is simply buried. Measured gains, fitting on half and predicting
    # the rest -- 1500/band: +3%, 3000/band: +37%, 5000/band: +47%.
    z = zeta_zeros(12000)
    sp, heights = unfolded_spacings(z), z[1:]
    edges = np.linspace(0, len(sp), 5).astype(int)
    Ls, devs = [], []
    for lo, hi in zip(edges, edges[1:]):
        t = float(np.exp(np.mean(np.log(heights[lo:hi]))))
        Ls.append(np.log(t / (2 * np.pi)))
        devs.append(dens(sp[lo:hi]) - p_gue)

    g = np.mean([devs[i] * Ls[i] for i in range(2)], axis=0)
    assert abs(float(g.sum() * ds)) < 0.05        # both sides are densities

    gain = float(np.mean([1.0 - l1(devs[i] - g / Ls[i]) / l1(devs[i])
                          for i in (2, 3)]))
    shuffled = float(np.mean([1.0 - l1(devs[i] - g[rng.permutation(len(g))] / Ls[i])
                              / l1(devs[i]) for i in (2, 3)]))
    assert gain > 0.15          # the fitted shape helps on unseen bands
    assert gain > shuffled      # ...and it is the shape doing it, not the size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
