"""
Phase 5 — regression tests for three bugs found during review.

Each test here corresponds to a defect that shipped and was NOT caught by the
existing 177-test suite, because the old tests checked bk.pl rules in
isolation and never checked the generated Popper inputs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tactical_ilp.config import PROLOG_DIR
from tactical_ilp.prolog.phase5_vocabulary import (
    COMPLEMENT_PREDS,
    LEAKY_PREDICATES,
    SEQUENCE_PREDS_1,
    WRAPPED_ACTION_PREDS,
    WRAPPED_PLAYER_SEQ_PREDS,
)
from tactical_ilp.prolog.legacy_43_inputs import build_legacy_43_bias, build_examples

POPPER_DIR = PROLOG_DIR / "popper"
PROCESSED_DIR = PROLOG_DIR.parent / "processed"


def _require_phase2_data():
    required = (
        PROCESSED_DIR / "labeled_sequences.parquet",
        PROCESSED_DIR / "ilp_examples.parquet",
    )
    if not all(path.exists() for path in required):
        pytest.skip("phase 2 parquet outputs are not included in this clean package")


# ---------------------------------------------------------------------------
# BUG 1 — target leakage
# ---------------------------------------------------------------------------
def test_no_leaky_predicate_is_offered_to_popper():
    """hesitant_negative_sequence/1 contains sequence_label(S,negative) in its
    body.  Exposed as a body_pred it gave Popper a precision-1.00 clause that
    simply restates the label: removing it dropped recall 0.27 -> 0.07, i.e.
    ~78% of the apparent signal was leakage.
    """
    offered = (
        set(SEQUENCE_PREDS_1)
        | {f"seq_{p}" for p in WRAPPED_ACTION_PREDS}
        | {f"seq_{p}" for p in WRAPPED_PLAYER_SEQ_PREDS}
    )
    assert not (offered & LEAKY_PREDICATES)


def test_bias_file_contains_no_leaky_predicate(tmp_path):
    bias = build_legacy_43_bias(tmp_path / "bias.pl")
    text = bias.read_text(encoding="utf-8")
    for leaky in LEAKY_PREDICATES:
        assert f"body_pred({leaky}," not in text


def test_leakage_guard_actually_raises(monkeypatch, tmp_path):
    """The guard must be a hard failure, not a warning."""
    import tactical_ilp.prolog.legacy_43_inputs as pi

    monkeypatch.setattr(
        pi, "SEQUENCE_PREDS_1", list(pi.SEQUENCE_PREDS_1) + ["hesitant_negative_sequence"]
    )
    with pytest.raises(ValueError, match="target leakage"):
        pi.build_legacy_43_bias(tmp_path / "_tmp_bias.pl")


def test_target_dependent_analysis_rule_is_not_in_runtime_bk():
    """The old label-dependent shortcut is retired from runtime Prolog."""
    source_bk = Path(__file__).parents[1] / "src" / "tactical_ilp" / "prolog" / "bk.pl"
    text = source_bk.read_text(encoding="utf-8")
    assert "hesitant_negative_sequence(S) :-" not in text
    assert "negative_sequence(S) :-" not in text
    assert "positive_sequence(S) :-" not in text
    assert LEAKY_PREDICATES == {
        "hesitant_negative_sequence", "negative_sequence", "positive_sequence"
    }


# ---------------------------------------------------------------------------
# BUG 2 — type-mismatched wrapper produced a silently dead predicate
# ---------------------------------------------------------------------------
def test_tempo_predicates_are_sequence_level_not_action_level():
    """Tempo buckets are unary sequence summaries and must never be lifted
    through in_sequence/2 as if they were action concepts.  Fast tempo is
    deliberately withheld from Popper to avoid a shallow search shortcut."""
    assert "fast_sequence" not in SEQUENCE_PREDS_1
    assert "slow_sequence" in SEQUENCE_PREDS_1
    assert "fast_sequence" not in WRAPPED_ACTION_PREDS
    assert "slow_sequence" not in WRAPPED_ACTION_PREDS


@pytest.mark.slow
def test_every_offered_body_pred_has_at_least_one_solution(swipl_available):
    """A body predicate with zero solutions is dead weight in the search space
    and is almost always a bug rather than a deliberate choice."""
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    bk = POPPER_DIR / "bk.pl"
    bias = POPPER_DIR / "bias.pl"
    if not bk.exists() or not bias.exists():
        pytest.skip("run scripts/run_phase5.py first")

    preds = []
    for line in bias.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("body_pred("):
            preds.append(line[len("body_pred(") :].split(",")[0])

    goal_parts = [f"consult('{bk.as_posix()}')"]
    for p in preds:
        goal_parts.append(
            f"aggregate_all(count, distinct(S, {p}(S)), N_{p}), "
            f"format('~w ~w~n', ['{p}', N_{p}])"
        )
    goal = ", ".join(goal_parts) + ", halt."
    r = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-g", goal],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    dead = [
        ln.split()[0]
        for ln in r.stdout.strip().splitlines()
        if len(ln.split()) == 2 and ln.split()[1] == "0"
    ]
    assert not dead, f"body predicates with zero solutions: {dead}"


# ---------------------------------------------------------------------------
# BUG 3 — training on everything, and discarding the matched negatives
# ---------------------------------------------------------------------------
def test_examples_use_context_matched_negatives_by_default(tmp_path):
    """Phase 2.4 built a context-matched negative sample precisely so Popper
    could not learn context instead of outcome; the first phase-5 version read
    labeled_sequences.parquet instead and used all 923 negatives."""
    _require_phase2_data()
    _, n_pos, n_neg, split = build_examples(tmp_path / "exs.pl")
    assert split["matched_negatives"] is True
    assert n_neg < 900, "looks like the unmatched 923-negative set is back"


def test_holdout_is_disjoint_and_split_by_match(tmp_path):
    _require_phase2_data()
    out, _, _, split = build_examples(tmp_path / "exs.pl")
    train_ids = {
        ln.split("(")[-1].rstrip(").")
        for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    hold_ids = {
        ln.split("(")[-1].rstrip(").")
        for ln in (out.parent / "exs_holdout.pl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    assert train_ids and hold_ids
    assert not (train_ids & hold_ids), "a sequence appears in both train and holdout"
    assert split["holdout_pos"] > 0 and split["holdout_neg"] > 0


def test_holdout_matches_do_not_appear_in_training(tmp_path):
    _require_phase2_data()
    out, _, _, split = build_examples(tmp_path / "exs.pl")
    hold_matches = {str(m) for m in split["holdout_matches"]}
    for ln in out.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        sid = ln.split("(")[-1].rstrip(").")
        match_id = sid.split("_")[1]
        assert match_id not in hold_matches


# ---------------------------------------------------------------------------
# complement predicates (protective signals)
# ---------------------------------------------------------------------------
def test_complement_predicates_are_declared_in_bias(tmp_path):
    bias = build_legacy_43_bias(tmp_path / "bias.pl").read_text(encoding="utf-8")
    for pred in COMPLEMENT_PREDS:
        assert f"body_pred(not_{pred},1)." in bias
