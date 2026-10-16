"""
Phase 6 tests — Support/Confidence/Lift computation, hold-out re-validation,
and the coach-validation workflow.

These correspond directly to the execution plan's phase 6 acceptance tests
(tasks 6.0, 6.1, 6.2, 6.3), rewritten against this project's real data
instead of the plan's illustrative fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tactical_ilp.evaluation.coach_validation import (
    validate_coach_file,
    write_coach_validation_file,
)
from tactical_ilp.evaluation.rule_metrics import (
    BK_FILE,
    HOLDOUT_FILE,
    KNOWN_HEALTHY_PATTERN_MAX_LIFT,
    MIN_SUPPORT,
    TRAIN_FILE,
    compute_lift,
    known_healthy_pattern_metrics,
    score_rule,
)
from tactical_ilp.evaluation.rule_registry import ALL_RULES, DISCOVERED_1


def _phase6_inputs_available() -> bool:
    return BK_FILE.exists() and TRAIN_FILE.exists() and HOLDOUT_FILE.exists()


pytestmark = pytest.mark.skipif(
    not _phase6_inputs_available(),
    reason="data/prolog/popper/{bk.pl,exs.pl,exs_holdout.pl} missing — run scripts/run_phase5.py first",
)


# --------------------------------------------------------------------------
# task 6.2a — synthetic lift check (execution plan acceptance criterion a)
# --------------------------------------------------------------------------
def test_lift_calculation_correct_on_synthetic_case():
    """A pattern whose confidence equals the base rate carries no information
    -- lift must be ~1.0, independent of any real data."""
    lift = compute_lift(confidence=0.25, base_rate=0.25)
    assert abs(lift - 1.0) < 0.1


def test_lift_above_one_means_riskier_than_average():
    assert compute_lift(confidence=0.5, base_rate=0.25) == pytest.approx(2.0)


def test_lift_below_one_means_safer_than_average():
    assert compute_lift(confidence=0.1, base_rate=0.25) == pytest.approx(0.4)


def test_lift_handles_zero_base_rate_without_crashing():
    assert compute_lift(confidence=0.5, base_rate=0.0) == 0.0


# --------------------------------------------------------------------------
# task 6.2b — known-healthy-pattern check (execution plan acceptance criterion b)
# --------------------------------------------------------------------------
def test_healthy_pattern_has_low_lift_if_present():
    """technical document section 2.7's own example: a frequent, healthy
    pattern (a teammate offering an open passing lane) must not read as
    dangerous just because it appears often."""
    metrics = known_healthy_pattern_metrics()
    if metrics is None:
        pytest.skip("bk.pl / exs.pl not available")
    assert metrics.lift < KNOWN_HEALTHY_PATTERN_MAX_LIFT


def test_healthy_pattern_is_actually_frequent():
    """The check above is only meaningful if the pattern is common enough
    that a high lift would have been noticeable."""
    metrics = known_healthy_pattern_metrics()
    if metrics is None:
        pytest.skip("bk.pl / exs.pl not available")
    assert metrics.support > 100


# --------------------------------------------------------------------------
# task 6.1 — Support / Confidence computed for every registered rule
# --------------------------------------------------------------------------
def test_every_registered_rule_has_support_and_confidence():
    for rule in ALL_RULES:
        r = score_rule(rule.name, rule.body, source=rule.source)
        assert r.train.support is not None
        assert 0.0 <= r.train.confidence <= 1.0
        assert 0.0 <= r.holdout.confidence <= 1.0


def test_low_support_rules_are_flagged_not_silently_kept():
    """A rule with support below MIN_SUPPORT must be flagged, per technical
    document section 6 ('3 to 5 occurrences minimum')."""
    tiny = score_rule(
        "synthetic_tiny_support",
        ["seq_keeper_build_up_start", "seq_triangle_trap", "seq_pressured_cage",
         "seq_persistent_half_space_occupier"],
    )
    if tiny.train.support < MIN_SUPPORT:
        assert tiny.flagged_low_support is True


# --------------------------------------------------------------------------
# task 6.0 — hold-out re-validation
# --------------------------------------------------------------------------
def test_discovered_rule_confidence_reported_on_both_splits():
    r = score_rule(DISCOVERED_1.name, DISCOVERED_1.body, source=DISCOVERED_1.source)
    assert r.train is not None and r.holdout is not None
    assert r.train.split == "exs"
    assert r.holdout.split == "exs_holdout"


def test_discovered_rule_does_not_collapse_on_holdout():
    """This is the rule actually shipped to the coach in phase 5 -- it should
    survive hold-out re-validation, not just look good on training data."""
    r = score_rule(DISCOVERED_1.name, DISCOVERED_1.body, source=DISCOVERED_1.source)
    assert r.flagged_holdout_drop is False
    assert r.holdout.lift > 1.2, "the shipped rule should still show real lift on unseen matches"


def test_overfit_candidate_is_correctly_flagged():
    """Regression check for the exact failure mode task 6.0 exists to catch:
    candidate_2a looks perfect on training (support=5, confidence=1.00) and
    completely collapses on holdout (confidence=0.00). If this predicate set
    ever stops collapsing, the flag logic itself needs to be re-checked, not
    just this test updated blindly."""
    from tactical_ilp.evaluation.rule_registry import CANDIDATE_2A

    r = score_rule(CANDIDATE_2A.name, CANDIDATE_2A.body, source=CANDIDATE_2A.source)
    assert r.train.confidence > r.holdout.confidence
    assert r.flagged_holdout_drop is True


def test_holdout_drop_ratio_is_between_zero_and_one_when_defined():
    for rule in ALL_RULES:
        r = score_rule(rule.name, rule.body, source=rule.source)
        if r.holdout_drop_ratio is not None:
            assert -0.01 <= r.holdout_drop_ratio <= 1.01  # small tolerance for confidence increasing


def test_test_holdout_confidence_reported_matches_execution_plan_shape(tmp_path):
    """Execution plan's own acceptance test, adapted: every rule carries both
    confidence_train and confidence_holdout, and rules with a large drop are
    flagged rather than silently kept."""
    rules_with_metrics = []
    for rule in ALL_RULES:
        r = score_rule(rule.name, rule.body, source=rule.source)
        rules_with_metrics.append(
            {
                "name": r.name,
                "confidence_train": r.train.confidence,
                "confidence_holdout": r.holdout.confidence,
                "flagged_for_review": r.flagged_holdout_drop,
            }
        )
    for r in rules_with_metrics:
        assert "confidence_train" in r and "confidence_holdout" in r
        if r["confidence_train"] > 0:
            drop_ratio = (r["confidence_train"] - r["confidence_holdout"]) / r["confidence_train"]
            assert r["flagged_for_review"] == (drop_ratio > 0.5)


# --------------------------------------------------------------------------
# task 6.3 — coach validation file
# --------------------------------------------------------------------------
def test_coach_validation_file_written_with_all_rules(tmp_path):
    scored = [score_rule(r.name, r.body, source=r.source) for r in ALL_RULES]
    out = write_coach_validation_file(scored, out_path=tmp_path / "coach.yaml")
    assert out.exists()

    import yaml

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert len(doc["rules"]) == len(ALL_RULES)
    for entry in doc["rules"]:
        assert entry["validation_status"] == "pending"
        assert "metrics" in entry and "flags" in entry


def test_coach_validation_rejects_missing_file(tmp_path):
    ok, problems = validate_coach_file(tmp_path / "does_not_exist.yaml")
    assert ok is False
    assert problems


def test_coach_validation_fails_while_pending(tmp_path):
    scored = [score_rule(r.name, r.body, source=r.source) for r in ALL_RULES]
    out = write_coach_validation_file(scored, out_path=tmp_path / "coach.yaml")
    ok, problems = validate_coach_file(out)
    assert ok is False
    assert len(problems) == len(ALL_RULES)


def test_coach_validation_passes_once_all_rules_reviewed(tmp_path):
    import yaml

    scored = [score_rule(r.name, r.body, source=r.source) for r in ALL_RULES]
    out = write_coach_validation_file(scored, out_path=tmp_path / "coach.yaml")

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    for entry in doc["rules"]:
        entry["validation_status"] = "approved"
        entry["validation_note"] = "test note"
    out.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    ok, problems = validate_coach_file(out)
    assert ok is True
    assert problems == []


def test_approved_status_allows_an_optional_note(tmp_path):
    import yaml

    scored = [score_rule(r.name, r.body, source=r.source) for r in ALL_RULES]
    out = write_coach_validation_file(scored, out_path=tmp_path / "coach.yaml")

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    for entry in doc["rules"]:
        entry["validation_status"] = "rejected"
        entry["validation_note"] = ""
    out.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    ok, problems = validate_coach_file(out)
    assert ok is True
    assert not problems


def test_real_coach_validation_file_present_and_structurally_valid():
    """The actual file shipped in data/reports/ -- checks structure, not that
    every rule has been approved (this project's real file may legitimately
    still have Claude's draft recommendations pending the user's own review)."""
    from tactical_ilp.evaluation.coach_validation import COACH_VALIDATION_PATH, read_coach_validation_file

    if not COACH_VALIDATION_PATH.exists():
        pytest.skip("phase6_coach_validation.yaml not generated yet")
    rules = read_coach_validation_file()
    assert len(rules) == len(ALL_RULES)
    for r in rules:
        assert r["validation_status"] in ("pending", "approved", "rejected")
        assert "prolog_body" in r
