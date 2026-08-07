"""
Tests for the parts that must not quietly break.

Split in two: everything above `torch` runs with numpy alone (the tapes, the
renderer, the symbolic rules, proof search, the halting calibration, the
conciseness accounting), and the learned-rule tests are skipped when torch is
absent. That split is the same one the package makes internally, so these tests
also check that importing dynamicmultinet does not drag in torch.

    pytest tests/            # or: python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamicmultinet import (ABSTRACT, SPECIFIC, Content, RenMachine,  # noqa: E402
                             RuleLibrary, ScriptedController, Task,
                             library_report, search)
from dynamicmultinet.codec import ChoiceCodec, TextSlotCodec  # noqa: E402
from dynamicmultinet.halting import (MAX_SAMPLES, calibrate,  # noqa: E402
                                     effective_lambda, sample_size)
from dynamicmultinet.palette import PALETTE, rgb_to_class_index  # noqa: E402
from dynamicmultinet.prior import eval_int_expression  # noqa: E402
from dynamicmultinet.render import render_text, split_views  # noqa: E402


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


# ---------------------------------------------------------------------------
# What agreement with a reference is worth
# ---------------------------------------------------------------------------
def test_primitives_sees_through_a_composite():
    """A chain hides its members, so comparing top-level names would call a
    reference independent of a rule it actually runs."""
    from dynamicmultinet.rules import CompositeRule
    from dynamicmultinet.verify import primitives

    m = RenMachine()
    chain = CompositeRule("ref", [m.library.get("decimal_split"),
                                  m.library.get("distribute_symbolic")])
    assert primitives(chain) == {"ref", "decimal_split", "distribute_symbolic"}


def test_verification_against_a_reference_that_runs_the_rule_is_refused():
    """The one failure mode that produces a perfect score: a reference which
    invokes the rule under test agrees with it for free."""
    from dynamicmultinet.verify import verify_against_rules

    m = RenMachine()
    m.generate_data("mul_pairs", 20, seed=3, name="d", a_digits=2, b_digits=2,
                    domain=ABSTRACT)
    m.label_data("d", "distributive_rewrite")
    with pytest.raises(ValueError, match="cannot be verified against a reference"):
        verify_against_rules(m.library, "decimal_split",
                             ["decimal_split", "distribute_symbolic"],
                             m.datasets["d"])


def test_collision_probability_separates_wide_from_narrow_answers():
    """Agreement is only worth something when the routes could have differed.
    Two unrelated rules choosing one of two actions agree half the time."""
    from dynamicmultinet.verify import collision_probability

    assert collision_probability([f"{i}0*55+{i}*55" for i in range(1, 90)]) < 0.01
    assert collision_probability(["a", "b"] * 45) > 0.4
    assert collision_probability(["only"]) == 1.0          # nothing to compare


def test_a_shared_training_oracle_defeats_independence():
    """Different weights are not independence. Two nets taught by the same
    oracle inherit its mistakes and can agree while both are wrong."""
    from dynamicmultinet.rules import PythonRule, Recipe
    from dynamicmultinet.verify import independence

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

    from dynamicmultinet.nets import RuleNet

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

    from dynamicmultinet.nets import rgb_to_class_channels

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
    from dynamicmultinet.rules import PythonRule

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
    from dynamicmultinet.rules import PythonRule

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
