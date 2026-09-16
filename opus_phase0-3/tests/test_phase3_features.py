"""
Phase 3 — rule-based feature extractor and Prolog fact generation
(tasks 3.1 .. 3.3).
"""

from __future__ import annotations

import subprocess

import pytest

from tactical_ilp.features.geometry import (
    is_progressive,
    is_zone14,
    lane_of,
    occupies_half_space,
    pass_direction,
    pass_length_class,
    pressure_level,
    third_of,
    under_pressure,
)
from tactical_ilp.prolog.prolog_facts import (
    count_distinct_arg,
    count_predicate_occurrences,
    predicate_histogram,
)
from tests.conftest import consult_prolog


# --------------------------------------------------------------------------
# Task 3.1 — geometric predicates
# --------------------------------------------------------------------------
def test_half_space_geometry():
    assert occupies_half_space(x=70, y=15) == "right_half_space"
    assert occupies_half_space(x=70, y=34) == "central_channel"


def test_lane_boundaries_cover_the_whole_pitch():
    assert lane_of(0.0) == "right_wing"
    assert lane_of(13.5) == "right_wing"
    assert lane_of(13.6) == "right_half_space"
    assert lane_of(27.2) == "central_channel"
    assert lane_of(40.8) == "left_half_space"
    assert lane_of(54.4) == "left_wing"
    assert lane_of(68.0) == "left_wing"


def test_thirds():
    assert third_of(10.0) == "defensive_third"
    assert third_of(35.0) == "defensive_third"
    assert third_of(50.0) == "middle_third"
    assert third_of(70.0) == "middle_third"
    assert third_of(95.0) == "final_third"


def test_zone14():
    assert is_zone14(80.0, 34.0) is True          # centre, just outside the box
    assert is_zone14(80.0, 10.0) is False         # too wide
    assert is_zone14(50.0, 34.0) is False         # too deep
    assert is_zone14(100.0, 34.0) is False        # inside the box


def test_pressure_model():
    # 360 frame available -> freeze frame wins
    assert under_pressure(opponents_within_5m=2, statsbomb_flag=False) is True
    assert under_pressure(opponents_within_5m=0, statsbomb_flag=True) is False
    # no 360 frame -> StatsBomb flag is the fallback
    assert under_pressure(opponents_within_5m=None, statsbomb_flag=True) is True
    assert under_pressure(opponents_within_5m=None, statsbomb_flag=False) is False


def test_pressure_levels():
    assert pressure_level(0, 0, False) == "no_pressure"
    assert pressure_level(1, 1, False) == "light_pressure"
    assert pressure_level(0, 2, False) == "light_pressure"
    assert pressure_level(2, 3, False) == "heavy_pressure"
    assert pressure_level(None, None, True) == "light_pressure"


def test_pass_direction_and_length():
    assert pass_direction(20, 34, 40, 34) == "forward"
    assert pass_direction(40, 34, 20, 34) == "backward"
    assert pass_direction(20, 10, 20, 50) == "sideways"
    assert pass_length_class(20, 34, 25, 34) == "short"
    assert pass_length_class(20, 34, 40, 34) == "medium"
    assert pass_length_class(20, 34, 60, 34) == "long"


def test_progressive_action():
    assert is_progressive(10, 34, 60, 34) is True      # own half, huge gain
    assert is_progressive(10, 34, 12, 34) is False     # own half, tiny gain
    assert is_progressive(80, 34, 95, 34) is True      # opponent half, >= 10 m
    assert is_progressive(80, 34, 84, 34) is False


# --------------------------------------------------------------------------
# Task 3.2 — fact file of a sample match loads in SWI-Prolog
# --------------------------------------------------------------------------
def test_generated_facts_load_in_prolog(facts_file_path, swipl_available):
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    result = consult_prolog(facts_file_path)
    assert result.returncode == 0, result.stderr


def test_generated_facts_have_no_prolog_warnings(facts_file_path, swipl_available):
    """A syntactically valid file can still be full of encoding warnings."""
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    result = consult_prolog(facts_file_path)
    assert "Warning" not in result.stderr, result.stderr
    assert "ERROR" not in result.stderr, result.stderr


def test_sample_fact_file_is_ascii(facts_file_path):
    """Non-ASCII bytes break consult under a non-UTF-8 Windows code page."""
    facts_file_path.read_bytes().decode("ascii")


# --------------------------------------------------------------------------
# Task 3.3 — whole dataset
# --------------------------------------------------------------------------
def test_final_facts_load_in_prolog(final_facts_file, swipl_available):
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    result = consult_prolog(final_facts_file)
    assert result.returncode == 0, result.stderr
    assert "Warning" not in result.stderr


def test_fact_count_matches_sequences(final_facts_file, labeled_sequences_df):
    """
    Deviation from the execution plan (documented in report.md):

    the plan wrote ``count_predicate_occurrences(file, "involved_in") ==
    len(labeled_sequences_df)``.  ``involved_in/2`` is one fact **per player per
    sequence**, so that equality can only hold for single-player sequences and
    contradicts the multi-player design of technical document 2.5.

    The intent of the check — "the facts cover exactly the labelled sequences,
    no more and no less" — is preserved by comparing the number of *distinct
    sequences* instead.
    """
    n_sequences = count_predicate_occurrences(final_facts_file, "sequence")
    assert n_sequences == len(labeled_sequences_df)

    distinct_in_involved_in = count_distinct_arg(final_facts_file, "involved_in", 1)
    assert distinct_in_involved_in == len(labeled_sequences_df)


def test_every_sequence_has_a_label_fact(final_facts_file, labeled_sequences_df):
    assert count_predicate_occurrences(final_facts_file, "sequence_label") == len(
        labeled_sequences_df
    )


def test_action_level_predicates_are_aligned(final_facts_file):
    hist = predicate_histogram(final_facts_file)
    n_actions = hist["action"]
    for pred in ("in_sequence", "action_order", "action_type", "action_result",
                 "at_third", "at_lane", "pressure_level"):
        assert hist[pred] == n_actions, f"{pred} is not defined for every action"
    assert hist["successful"] + hist["failed"] == n_actions


def test_identity_and_role_facts_present(final_facts_file, players_reference_df):
    hist = predicate_histogram(final_facts_file)
    assert hist["player_identity"] >= 15
    assert hist["has_role"] == hist["player_identity"]
    assert hist["plays_for"] == hist["player_identity"]


def test_relational_facts_present(final_facts_file):
    hist = predicate_histogram(final_facts_file)
    assert hist.get("passes_to", 0) >= 100
    assert hist.get("chain", 0) >= 1


def test_prolog_can_answer_a_tactical_query(final_facts_file, swipl_available):
    """
    End-to-end proof that the facts are queryable, not just loadable:
    count sequences that ended badly and contained an action under pressure.
    """
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    goal = (
        f"consult('{final_facts_file.as_posix()}'), "
        "aggregate_all(count, "
        "(sequence_label(S,negative), in_sequence(A,S), under_pressure(A)), N), "
        "format('N=~w~n',[N]), halt."
    )
    result = subprocess.run(
        ["swipl", "-g", goal, "-t", "halt(1)"], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    value = int(result.stdout.strip().split("N=")[1].split()[0])
    assert value > 0
