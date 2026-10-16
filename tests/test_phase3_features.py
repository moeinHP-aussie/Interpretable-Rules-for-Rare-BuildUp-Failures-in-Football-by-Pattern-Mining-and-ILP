"""
Phase 3 — rule-based feature extractor and Prolog fact generation
(tasks 3.1 .. 3.3).
"""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from tactical_ilp.features.geometry import (
    is_progressive,
    lane_of,
    pass_direction,
    pass_length_class,
    pressure_level,
    third_of,
    under_pressure,
)
from tactical_ilp.prolog.prolog_facts import (
    count_distinct_arg,
    count_predicate_occurrences,
    generate_facts,
    predicate_histogram,
)
from tests.conftest import consult_prolog


# --------------------------------------------------------------------------
# Task 3.1 — geometric predicates
# --------------------------------------------------------------------------
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
    # Binary contact pressure is more sensitive than the intensity bucket.
    assert under_pressure(1, False) is True
    assert pressure_level(0, 1, False) == "no_pressure"


def test_pass_direction_and_length():
    assert pass_direction(20, 34, 40, 34) == "forward"
    assert pass_direction(40, 34, 20, 34) == "backward"
    assert pass_direction(20, 10, 20, 50) == "sideways"
    assert pass_length_class(20, 34, 25, 34) == "short"
    assert pass_length_class(20, 34, 40, 34) == "medium"
    assert pass_length_class(20, 34, 60, 34) == "long"


def test_progressive_action():
    assert is_progressive(10, 34, 20, 34) is True      # defensive third, >=10% of remaining x
    assert is_progressive(10, 34, 18, 34) is False     # defensive third, below threshold
    assert is_progressive(40, 34, 60, 34) is True      # middle third, >=25% of remaining x
    assert is_progressive(40, 34, 50, 34) is False     # middle third, below threshold
    assert is_progressive(80, 34, 90, 34) is True      # final third, >=10 m forward
    assert is_progressive(80, 34, 84, 34) is False
    assert is_progressive(10, 0, 10, 34) is False      # lateral move toward goal centre
    assert is_progressive(20, 34, 19, 34) is False     # backward move
    assert is_progressive(float("nan"), 34, 20, 34) is False


def test_fact_writer_exports_the_pass_endpoint_lane(monkeypatch, tmp_path):
    import tactical_ilp.prolog.prolog_facts as facts

    context = pd.DataFrame(
        {"sequence_id": ["s_12_0001"], "match_id": [12], "n_actions": [2]}
    )
    members = pd.DataFrame(
        {
            "sequence_id": ["s_12_0001", "s_12_0001"],
            "match_id": [12, 12],
            "action_id": [0, 1],
            "order_in_sequence": [0, 1],
        }
    )
    actions = pd.DataFrame(
        {
            "match_id": [12, 12],
            "action_id": [0, 1],
            "player_id": [7, 8],
            "type_name": ["pass", "dribble"],
            "result_name": ["success", "success"],
            "start_x": [10.0, 20.0],
            "start_y": [10.0, 20.0],
            "end_x": [20.0, 25.0],
            "end_y": [20.0, 25.0],
            "under_pressure": [False, False],
            "ff_opponents_3m": [0.0, float("nan")],
            "ff_opponents_5m": [0.0, float("nan")],
            "ff_n_visible": [10.0, float("nan")],
        }
    )
    players = pd.DataFrame(
        {"player_id": [7, 8], "role": ["midfielder", "forward"]}
    )
    monkeypatch.setattr(facts, "build_context", lambda: context)
    monkeypatch.setattr(facts, "build_sequences", lambda: (context, members))
    monkeypatch.setattr(facts, "build_merged_all", lambda: actions)
    monkeypatch.setattr(facts, "build_players_reference", lambda: players)

    out = tmp_path / "facts.pl"
    generate_facts([12], out)
    text = out.read_text(encoding="utf-8")
    assert "pass_end_lane(a_12_00000,right_half_space)." in text
    assert "pass_end_lane(a_12_00001," not in text


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


def test_target_labels_are_not_exported_as_background_facts(final_facts_file):
    hist = predicate_histogram(final_facts_file)
    assert "sequence_label" not in hist


def test_action_level_predicates_are_aligned(final_facts_file):
    hist = predicate_histogram(final_facts_file)
    n_actions = hist["in_sequence"]
    assert "action" not in hist
    assert "sequence_match" not in hist and "sequence_start_zone" not in hist
    for pred in ("in_sequence", "action_order", "action_type", "action_result",
                 "at_third", "at_lane", "pressure_level"):
        assert hist[pred] == n_actions, f"{pred} is not defined for every action"
    assert hist["action_result"] == n_actions
    assert "successful" not in hist and "failed" not in hist
    assert 0 < hist["visible_player_count"] <= n_actions
    assert 0 < hist["pass_end_lane"] <= n_actions


def test_identity_and_role_facts_present(final_facts_file, players_reference_df):
    hist = predicate_histogram(final_facts_file)
    assert "player_identity" not in hist and "player_name" not in hist
    assert hist["has_role"] >= 15
    assert hist["plays_for"] == hist["has_role"]


def test_inactive_pass_relations_are_omitted(final_facts_file):
    hist = predicate_histogram(final_facts_file)
    assert "passes_to" not in hist
    assert "pass_link" not in hist
    assert "chain" not in hist


def test_prolog_can_answer_a_tactical_query(final_facts_file, swipl_available):
    """
    End-to-end proof that the facts are queryable, not just loadable:
    count in-sequence actions under pressure without importing target labels.
    """
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    goal = (
        f"consult('{final_facts_file.as_posix()}'), "
        "aggregate_all(count, "
        "(sequence(S), in_sequence(A,S), under_pressure(A)), N), "
        "format('N=~w~n',[N]), halt."
    )
    result = subprocess.run(
        ["swipl", "-g", goal, "-t", "halt(1)"], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    value = int(result.stdout.strip().split("N=")[1].split()[0])
    assert value > 0
