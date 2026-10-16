"""
Phase 5 tests — full tactical Popper input generation.

Actually running Popper needs janus_swi (SWI-Prolog >= 9.1.12), which is not
guaranteed to be on the test machine, so these tests validate everything that
does not require Popper itself: the generated files are syntactically valid
Prolog, the counts match the source data, and the wrapper predicates
correctly lift action/player-level concepts to the sequence level.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from tactical_ilp.config import PROLOG_DIR
from tactical_ilp.prolog.phase5_vocabulary import (
    SEQUENCE_PREDS_1, MUTUALLY_EXCLUSIVE_BODY_PAIRS, WRAPPED_ACTION_PREDS,
    WRAPPED_PLAYER_SEQ_PREDS,
)
from tactical_ilp.prolog.legacy_43_inputs import build_legacy_43_bias, generate_popper_inputs

POPPER_DIR = PROLOG_DIR / "popper"


def _swipl_available() -> bool:
    try:
        return subprocess.run(["swipl", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def generated():
    if not (PROLOG_DIR / "facts_tactical.pl").exists() or not (PROLOG_DIR / "bk.pl").exists():
        pytest.skip("phase 4 output not generated")
    return generate_popper_inputs()


def test_exs_is_a_strict_subset_of_the_labelled_sequences(generated):
    """UPDATED after review.

    The original assertion required exs.pl to contain EVERY labelled
    sequence.  That encoded two defects at once: it discarded the
    context-matched negative sample built in phase 2.4, and it left no
    held-out data, so Popper's reported precision was measured on its own
    training set.  exs.pl is now the training split of the matched sample,
    so it must be a strict subset.
    """
    labeled = pd.read_parquet(PROLOG_DIR.parent / "processed" / "labeled_sequences.parquet")
    total_pos = int((labeled.label == "negative").sum())
    total_neg = int((labeled.label == "non_negative").sum())

    assert 0 < generated["n_pos"] < total_pos
    assert 0 < generated["n_neg"] < total_neg
    # matched sampling keeps negatives near the documented 2:1 ratio,
    # nowhere near the full 923
    assert generated["n_neg"] < total_neg / 2


def test_exs_file_has_no_overlap_between_pos_and_neg(generated):
    text = generated["exs"].read_text(encoding="utf-8")
    pos_ids = {line for line in text.splitlines() if line.startswith("pos(")}
    neg_ids = {line for line in text.splitlines() if line.startswith("neg(")}
    pos_seqs = {line.split("negative_outcome(")[1].rstrip(")).") for line in pos_ids}
    neg_seqs = {line.split("negative_outcome(")[1].rstrip(")).") for line in neg_ids}
    assert pos_seqs.isdisjoint(neg_seqs)


def test_bias_file_declares_a_sequence_level_head(tmp_path):
    text = build_legacy_43_bias(tmp_path / "bias.pl").read_text(encoding="utf-8")
    assert "head_pred(negative_outcome,1)." in text
    assert "type(negative_outcome,(sequence,))." in text


def test_bias_file_has_no_raw_action_or_player_body_preds(tmp_path):
    """Every body predicate must operate on Sequence, matching the head, so
    Popper never has to introduce an Action or Player variable itself."""
    text = build_legacy_43_bias(tmp_path / "bias.pl").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("type("):
            assert "(sequence,)" in line, f"non-sequence body pred: {line}"


def test_bias_keeps_search_space_small(tmp_path):
    """Regression guard for the phase 4 pre-check's central warning: keep
    max_vars/max_body small until proven necessary."""
    text = build_legacy_43_bias(tmp_path / "bias.pl").read_text(encoding="utf-8")
    assert "max_vars(2)." in text
    assert "max_body(3)." in text
    assert "max_clauses(3)." in text


def test_legacy_bias_excludes_costly_hesitation_and_uses_declared_vocabulary(tmp_path):
    text = build_legacy_43_bias(tmp_path / "bias.pl").read_text(encoding="utf-8")
    body = [line for line in text.splitlines() if line.startswith("body_pred(")]
    expected = len(WRAPPED_ACTION_PREDS) + len(WRAPPED_PLAYER_SEQ_PREDS) + len(SEQUENCE_PREDS_1)
    assert len(body) == expected
    assert "body_pred(seq_costly_hesitation,1)." not in body


def test_full_tactical_vocabulary_contains_all_reviewed_concepts():
    assert set(WRAPPED_ACTION_PREDS) == {
        "line_breaking_pass", "diagonal_pass", "switch_to_right", "switch_to_left",
        "aerial_pass", "long_pass", "pass_in_behind",
        "line_pass", "connecting_pass", "dropped_between_lines",
        "first_time_play", "dwelt_long", "layoff", "supported_action",
        "dead_end_support", "triangulation", "triangle_trap", "rondo_cage",
        "pressured_cage", "press_resistance", "counter_pressed", "rest_defence_exposed",
        "under_pressure", "flank_line_carry", "flank_cut_inside", "flank_dribble",
        "central_switch", "recycle_to_same_flank", "center_cycling",
    }
    assert set(SEQUENCE_PREDS_1) == {
        "slow_sequence", "opponent_visible_majority",
        "keeper_played_open_play",
    }
    assert set(WRAPPED_PLAYER_SEQ_PREDS) == {
        "persistent_half_space_occupier",
    }


def test_active_vocabulary_order_and_cardinality_are_pinned():
    assert SEQUENCE_PREDS_1 == [
        "opponent_visible_majority", "slow_sequence",
        "keeper_played_open_play",
    ]
    assert WRAPPED_PLAYER_SEQ_PREDS == ["persistent_half_space_occupier"]
    assert WRAPPED_ACTION_PREDS == [
        "line_breaking_pass", "diagonal_pass", "switch_to_right", "switch_to_left",
        "aerial_pass", "long_pass", "pass_in_behind",
        "line_pass", "connecting_pass", "dropped_between_lines",
        "first_time_play", "dwelt_long", "layoff", "supported_action",
        "dead_end_support", "triangulation", "triangle_trap", "rondo_cage",
        "pressured_cage", "press_resistance", "counter_pressed",
        "rest_defence_exposed", "under_pressure", "flank_line_carry",
        "flank_cut_inside", "flank_dribble", "central_switch",
        "recycle_to_same_flank", "center_cycling",
    ]


def test_bias_prunes_only_definite_same_clause_exclusions(tmp_path):
    text = build_legacy_43_bias(tmp_path / "bias.pl").read_text(encoding="utf-8")
    for left, right in MUTUALLY_EXCLUSIVE_BODY_PAIRS:
        assert f"body_pred({left},1)." in text
        assert f"body_pred({right},1)." in text
        assert (
            f":- body_literal(C,{left},1,_), body_literal(C,{right},1,_)."
            in text
        )
    # These overlap rather than contradict: not_slow includes fast, and two
    # event wrappers may be satisfied by different actions in one sequence.
    assert (
        ":- body_literal(C,fast_sequence,1,_), "
        "body_literal(C,not_slow_sequence,1,_)."
    ) not in text


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
def test_generated_bk_wrapper_loads_cleanly(generated):
    script = f"consult('{generated['bk']}'), write(ok), nl, halt."
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
def test_generated_exs_loads_cleanly(generated):
    script = (
        f"consult('{generated['exs']}'), "
        "aggregate_all(count, pos(_), NP), aggregate_all(count, neg(_), NN), "
        "format('~d ~d', [NP, NN]), halt."
    )
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"{generated['n_pos']} {generated['n_neg']}"


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.parametrize("pred", WRAPPED_ACTION_PREDS)
def test_every_action_wrapper_is_queryable(generated, pred):
    """Every seq_<pred>/1 wrapper must at least be callable without error --
    whether or not it has solutions depends on the underlying concept
    (documented exceptions like counter_attack are excluded)."""
    script = f"consult('{generated['bk']}'), ( seq_{pred}(_) -> true ; true ), write(ok), nl, halt."
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, f"seq_{pred}/1 failed to load or query: {out.stderr}"


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.parametrize("pred", WRAPPED_PLAYER_SEQ_PREDS)
def test_every_player_seq_wrapper_is_queryable(generated, pred):
    script = f"consult('{generated['bk']}'), ( seq_{pred}(_) -> true ; true ), write(ok), nl, halt."
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, f"seq_{pred}/1 failed to load or query: {out.stderr}"


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
def test_wrapper_lifts_action_level_to_distinct_sequences(generated):
    """The wrapper must genuinely reduce many actions to distinct sequences,
    not just rename the action predicate 1:1."""
    script = (
        f"consult('{generated['bk']}'), "
        "aggregate_all(count, triangulation(_), NA), "
        "aggregate_all(count, distinct(S, seq_triangulation(S)), NS), "
        "format('~d ~d', [NA, NS]), halt."
    )
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    n_actions, n_sequences = map(int, out.stdout.split())
    assert n_sequences < n_actions
    assert n_sequences > 0


def test_sequence_level_preds_are_defined_in_hand_written_bk():
    """SEQUENCE_PREDS_1 predicates are used directly (no wrapper needed), so
    they must already exist as Sequence-arity predicates in bk.pl."""
    source_bk = Path(__file__).parents[1] / "src" / "tactical_ilp" / "prolog" / "bk.pl"
    bk_text = source_bk.read_text(encoding="utf-8")
    for pred in SEQUENCE_PREDS_1:
        assert f"{pred}(" in bk_text, f"{pred}/1 not found in hand-written bk.pl"
    majority_rule = bk_text.split("opponent_visible_majority(S) :-", 1)[1].split("\n\n", 1)[0]
    assert "sequence(S)" in majority_rule
    assert "opponent_visible_majority_eligible_action(A)" in majority_rule
    assert "opponent_visible_majority_action(A)" in majority_rule
    assert "minimum_visible_majority_frames(Minimum)" in majority_rule
    assert "Eligible >= Minimum" in majority_rule
    assert "Majority * 2 > Eligible" in majority_rule
