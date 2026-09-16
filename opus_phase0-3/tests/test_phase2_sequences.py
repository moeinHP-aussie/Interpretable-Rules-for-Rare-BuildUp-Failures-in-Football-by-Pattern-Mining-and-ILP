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


def test_sequences_start_in_own_defensive_third(sequences_df):
    assert (sequences_df.start_x <= BUILD_UP_START_MAX_X + 1e-6).all()


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
    ratio = labeled_sequences_df["label"].value_counts(normalize=True)
    assert ratio.min() >= 0.10, f"label balance too skewed: {ratio.to_dict()}"


def test_every_sequence_has_exactly_one_label(labeled_sequences_df, sequences_df):
    assert len(labeled_sequences_df) == len(sequences_df)
    assert labeled_sequences_df["label"].notna().all()
    assert set(labeled_sequences_df["label"]) <= {"negative", "non_negative"}


def test_negative_outcome_definition_documented(outcome_definition_doc):
    assert outcome_definition_doc["exists"] is True
    assert outcome_definition_doc["has_explicit_criteria"] is True
    assert outcome_definition_doc["mismatches"] == []


def test_negative_label_implies_all_three_conditions(labeled_sequences_df):
    """The label must be derivable from the documented criteria, nothing else."""
    neg = labeled_sequences_df[labeled_sequences_df.label == "negative"]
    assert neg.lost_possession.all()
    assert neg.turnover_in_own_half.all()
    assert (neg.opponent_shot_after | neg.opponent_entered_def_third).all()


def test_successful_exits_are_mostly_non_negative(labeled_sequences_df):
    """Sanity: leaving the defensive third cleanly is rarely a negative outcome."""
    exits = labeled_sequences_df[labeled_sequences_df.end_reason == "exit"]
    assert (exits.label == "non_negative").mean() >= 0.80


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
    assert abs(len(neg_examples_df) - expected) <= max(2, 0.05 * expected)


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
