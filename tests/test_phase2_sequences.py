"""
Phase 2 — build-up sequences, outcome labels, player subsets, negative sampling
(tasks 2.1 .. 2.4).
"""

from __future__ import annotations

import pandas as pd

from tactical_ilp.config import (
    BUILD_UP_START_MAX_X,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    NEG_SAMPLES_PER_POSITIVE,
    MAX_GLOBAL_MATCH_SHARE,
)
from tactical_ilp.sequences.outcome_labeler import (
    _label_one,
    _regained_within_window,
)


# --------------------------------------------------------------------------
# Task 2.1 — sequence segmentation
# --------------------------------------------------------------------------
def test_sequence_length_reasonable(sample_sequences):
    assert len(sample_sequences) > 0
    lengths = [len(seq) for seq in sample_sequences]
    assert all(MIN_SEQUENCE_LENGTH <= l <= MAX_SEQUENCE_LENGTH for l in lengths)


def test_sequences_found_in_every_match(sequences_df, barca_match_ids):
    assert sequences_df.match_id.nunique() == len(barca_match_ids)
    per_match = sequences_df.groupby("match_id").size()
    assert per_match.min() >= 5, "a match with almost no build-up is suspicious"


def test_sequences_start_in_own_half(sequences_df):
    assert (sequences_df.start_x <= BUILD_UP_START_MAX_X + 1e-6).all()


def test_restart_sequence_start_filter_preserves_short_keeper_build_up():
    from tactical_ilp.sequences.sequence_builder import _eligible_sequence_start

    def row(kind, end_x, end_y=34.0):
        return pd.Series({
            "type_name": kind,
            "start_x": 5.0,
            "start_y": 34.0,
            "end_x": end_x,
            "end_y": end_y,
        })

    assert not _eligible_sequence_start(row("throw_in", 10.0))
    assert not _eligible_sequence_start(row("corner_crossed", 30.0))
    assert not _eligible_sequence_start(row("freekick_short", 10.0))
    assert _eligible_sequence_start(row("goalkick", 14.0))  # 9 m
    assert _eligible_sequence_start(row("goalkick", 29.0))  # 24 m
    assert not _eligible_sequence_start(row("goalkick", 31.0))  # 26 m
    # The filter applies to sequence origin, not to long passes during open play.
    assert _eligible_sequence_start(row("pass", 45.0))


def test_sequence_membership_is_consistent(sequences_df, sequence_actions_df):
    counted = sequence_actions_df.groupby("sequence_id").size()
    declared = sequences_df.set_index("sequence_id")["n_actions"]
    joined = declared.to_frame("declared").join(counted.rename("counted"))
    assert joined["counted"].notna().all()
    assert (joined["declared"] == joined["counted"]).all()


def test_sequence_ids_are_unique(sequences_df):
    assert sequences_df.sequence_id.is_unique


# --------------------------------------------------------------------------
# Task 2.2 — negative outcome definition and labels
# --------------------------------------------------------------------------
def test_label_balance(labeled_sequences_df):
    trainable = labeled_sequences_df[
        labeled_sequences_df.label.isin(["negative", "non_negative"])
    ]
    ratio = trainable["label"].value_counts(normalize=True)
    assert ratio.min() >= 0.10, f"trainable-label balance too skewed: {ratio.to_dict()}"


def test_every_sequence_has_exactly_one_label(labeled_sequences_df, sequences_df):
    assert len(labeled_sequences_df) == len(sequences_df)
    assert labeled_sequences_df["label"].notna().all()
    assert set(labeled_sequences_df["label"]) <= {"negative", "non_negative"}


def test_negative_outcome_definition_documented(outcome_definition_doc):
    assert outcome_definition_doc["exists"] is True
    assert outcome_definition_doc["has_explicit_criteria"] is True
    assert outcome_definition_doc["mismatches"] == []


def test_negative_labels_follow_own_half_loss_definition(labeled_sequences_df):
    """Every negative is an unrecovered loss in the selected team's own half."""
    neg = labeled_sequences_df[labeled_sequences_df.label == "negative"]
    assert neg.lost_possession.all()
    assert neg.turnover_x.le(BUILD_UP_START_MAX_X).all()
    assert (~neg.regained_within_5s).all()
    assert (neg.negative_reason == "own_half_loss_not_regained_within_5s").all()


def test_regain_window_only_counts_controlled_selected_team_actions():
    after = pd.DataFrame([
        {"row": 10, "time_seconds": 10.0, "is_barca": False,
         "result_name": "fail", "type_name": "pass"},
        {"row": 11, "time_seconds": 11.0, "is_barca": True,
         "result_name": "success", "type_name": "tackle"},
        {"row": 12, "time_seconds": 12.0, "is_barca": True,
         "result_name": "success", "type_name": "pass"},
    ])
    assert _regained_within_window(after, turnover_row=10, turnover_time=10.0)
    assert not _regained_within_window(after.iloc[:2], turnover_row=10, turnover_time=10.0)


def test_half_space_entry_names_follow_y_zero_right_convention():
    from tactical_ilp.sequences.sequence_builder import _entry_regions

    right = _entry_regions(75.0, 20.0)
    left = _entry_regions(75.0, 48.0)
    assert "right_deep_half_space" in right
    assert "left_deep_half_space" not in right
    assert "left_deep_half_space" in left
    assert "right_deep_half_space" not in left


def test_completed_build_up_endings_do_not_override_label_definition():
    for end_reason in ("shot", "attacking_zone_entry"):
        terminal_type = "shot" if end_reason == "shot" else "dribble"
        seq = pd.Series(
            {"sequence_id": "s_completed", "period_id": 1,
             "last_action_row": 1, "end_reason": end_reason}
        )
        actions = pd.DataFrame(
            [
                {"row": 1, "period_id": 1, "is_barca": True,
                 "result_name": "success", "type_name": terminal_type,
                 "start_x": 80.0, "end_x": 90.0, "time_seconds": 1.0},
                {"row": 2, "period_id": 1, "is_barca": False,
                 "result_name": "success", "type_name": "pass",
                 "start_x": 50.0, "end_x": 49.0, "time_seconds": 2.0},
                {"row": 3, "period_id": 1, "is_barca": False,
                 "result_name": "success", "type_name": "pass",
                 "start_x": 49.0, "end_x": 48.0, "time_seconds": 3.0},
            ]
        )
        label = _label_one(seq, actions)
        assert label["build_up_completed"] is True
        assert label["label"] == "negative"

    # A first-opponent foul is excluded from the turnover definition.
    seq = pd.Series(
        {"sequence_id": "s_shot_foul", "period_id": 1,
         "last_action_row": 1, "end_reason": "shot"}
    )
    actions = pd.DataFrame(
        [
            {"row": 1, "period_id": 1, "is_barca": True,
             "result_name": "success", "type_name": "shot",
             "start_x": 80.0, "end_x": 90.0, "time_seconds": 1.0},
            {"row": 2, "period_id": 1, "is_barca": False,
             "result_name": "success", "type_name": "foul",
             "start_x": 50.0, "end_x": 50.0, "time_seconds": 2.0},
        ]
    )
    assert _label_one(seq, actions)["label"] == "non_negative"


def test_single_midfield_penetration_or_shot_is_immediate_negative():
    for opponent_action in (
        {"type_name": "pass", "result_name": "success", "start_x": 50.0, "end_x": 34.0},
        {"type_name": "shot", "result_name": "fail", "start_x": 50.0, "end_x": 50.0},
    ):
        seq = pd.Series(
            {"sequence_id": "s_midfield", "period_id": 1,
             "last_action_row": 1, "end_reason": "failed_action"}
        )
        actions = pd.DataFrame(
            [
                {"row": 1, "period_id": 1, "is_barca": True,
                 "result_name": "fail", "type_name": "pass",
                 "start_x": 50.0, "end_x": 50.0, "time_seconds": 1.0},
                {"row": 2, "period_id": 1, "is_barca": False,
                 **opponent_action, "time_seconds": 2.0},
            ]
        )
        label = _label_one(seq, actions)
        assert label["label"] == "negative"
        assert label["negative_reason"] == "own_half_loss_not_regained_within_5s"


def test_global_fallback_is_limited_to_five_percent_of_final_negatives():
    from tactical_ilp.sequences.negative_sampling import _global_fallback_budget

    assert _global_fallback_budget(18) == 0
    assert _global_fallback_budget(19) == 1
    assert _global_fallback_budget(95) == 5
    assert MAX_GLOBAL_MATCH_SHARE == 0.05


def test_first_opponent_foul_is_non_negative_not_a_turnover():
    seq = pd.Series({"sequence_id": "s1", "period_id": 1, "last_action_row": 1})
    actions = pd.DataFrame(
        [
            {"row": 1, "period_id": 1, "is_barca": True, "result_name": "success", "type_name": "pass", "start_x": 20.0, "end_x": 25.0, "time_seconds": 1.0},
            {"row": 2, "period_id": 1, "is_barca": False, "result_name": "success", "type_name": "foul", "start_x": 25.0, "end_x": 25.0, "time_seconds": 2.0},
        ]
    )
    label = _label_one(seq, actions)
    assert label["lost_possession"] is False
    assert label["label"] == "non_negative"
    assert label["turnover_zone"] == "none"


def test_failed_pass_in_opponent_half_is_non_negative():
    seq = pd.Series({"sequence_id": "s2", "period_id": 1, "last_action_row": 1})
    actions = pd.DataFrame(
        [
            {"row": 1, "period_id": 1, "is_barca": True, "result_name": "fail", "type_name": "pass", "start_x": 65.0, "end_x": 72.0, "time_seconds": 1.0},
        ]
    )
    label = _label_one(seq, actions)
    assert label["lost_possession"] is True
    assert label["label"] == "non_negative"
    assert label["negative_reason"] == "loss_outside_own_half"


# --------------------------------------------------------------------------
# Task 2.3 — multi-player subsets
# --------------------------------------------------------------------------
def test_multi_player_subsets_extracted(sequences_with_subsets):
    has_subset = sequences_with_subsets["player_pairs"].apply(lambda x: len(x) > 0)
    assert has_subset.mean() >= 0.50


def test_pairs_reference_two_distinct_players(sequences_with_subsets):
    for pairs in sequences_with_subsets["player_pairs"]:
        for pair in pairs:
            a, b = pair.split(">")
            assert a != b


def test_triples_have_three_distinct_players():
    from tactical_ilp.sequences.player_subsets import TRIPLES_PATH

    triples = pd.read_parquet(TRIPLES_PATH)
    if len(triples) == 0:
        return
    assert (triples.p1 != triples.p2).all()
    assert (triples.p2 != triples.p3).all()
    assert (triples.p1 != triples.p3).all()


# --------------------------------------------------------------------------
# Task 2.4 — negative sampling
# --------------------------------------------------------------------------
def test_negative_samples_context_comparable(pos_examples_df, neg_examples_df):
    pos_zone_dist = pos_examples_df["zone"].value_counts(normalize=True)
    neg_zone_dist = neg_examples_df["zone"].value_counts(normalize=True)
    total_overlap = sum(
        min(pos_zone_dist.get(z, 0), neg_zone_dist.get(z, 0))
        for z in set(pos_zone_dist.index) | set(neg_zone_dist.index)
    )
    assert total_overlap >= 0.4


def test_negative_sampling_ratio_respected(pos_examples_df, neg_examples_df):
    expected = len(pos_examples_df) * NEG_SAMPLES_PER_POSITIVE
    # The target is 2:1, but the context-preservation cap can lower the ratio.
    assert len(neg_examples_df) <= expected


def test_no_sequence_is_both_positive_and_negative(ilp_examples_df):
    assert ilp_examples_df.sequence_id.is_unique


def test_examples_classes_match_labels(ilp_examples_df):
    pos = ilp_examples_df[ilp_examples_df.example_class == "pos"]
    neg = ilp_examples_df[ilp_examples_df.example_class == "neg"]
    assert (pos.label == "negative").all()
    assert (neg.label == "non_negative").all()


def test_most_negatives_come_from_the_exact_stratum(neg_examples_df):
    exact = (neg_examples_df.match_level == 3).mean()
    assert exact >= 0.80, f"only {exact:.1%} of negatives matched the exact stratum"


def test_global_fallback_share_never_exceeds_five_percent(neg_examples_df):
    if len(neg_examples_df):
        assert (neg_examples_df.match_level == 0).mean() <= MAX_GLOBAL_MATCH_SHARE


def test_sampling_strategy_documented(sampling_strategy_doc):
    assert sampling_strategy_doc["exists"] is True
    assert sampling_strategy_doc["has_explicit_criteria"] is True
    assert sampling_strategy_doc["mismatches"] == []


def test_negative_sampling_is_deterministic():
    """Same seed -> same example set (needed for reproducible phase 5 runs)."""
    from tactical_ilp.sequences.negative_sampling import sample_examples

    a = sample_examples()
    b = sample_examples(force=True)
    assert set(a.sequence_id) == set(b.sequence_id)
