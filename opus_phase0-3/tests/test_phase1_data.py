"""
Phase 1 — raw data loading and preparation (tasks 1.1 .. 1.4).
"""

from __future__ import annotations

import pandas as pd

from tactical_ilp.config import (
    MIN_APPEARANCES_FOR_REFERENCE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TEAM_ID,
    cache_path,
)


# --------------------------------------------------------------------------
# Task 1.1 — cache
# --------------------------------------------------------------------------
def test_all_matches_cached(barca_match_ids, cache_dir):
    for match_id in barca_match_ids:
        assert (cache_dir / f"events_{match_id}.parquet").exists()
        assert (cache_dir / f"frames_{match_id}.parquet").exists()
        assert (cache_dir / f"lineup_{match_id}.parquet").exists()


def test_second_run_reads_from_cache(barca_match_ids, monkeypatch):
    """Loading twice must not hit the network — the loader is never constructed."""
    from tactical_ilp.data import load_raw_data

    def _explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("network access attempted although the cache exists")

    monkeypatch.setattr(
        "socceraction.data.statsbomb.StatsBombLoader", _explode, raising=False
    )
    df = load_raw_data.load_events(barca_match_ids[0])
    assert len(df) > 0


# --------------------------------------------------------------------------
# Task 1.2 — SPADL
# --------------------------------------------------------------------------
def test_spadl_schema(sample_spadl_df):
    required_cols = {
        "player_id",
        "team_id",
        "type_name",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "result_name",
    }
    assert required_cols.issubset(sample_spadl_df.columns)
    assert len(sample_spadl_df) > 0


def test_spadl_row_count_is_plausible(spadl_all_matches, barca_match_ids):
    """A SPADL table holds roughly half to three quarters of the raw events."""
    from tactical_ilp.data.load_raw_data import load_events

    mid = barca_match_ids[0]
    events = load_events(mid)
    actions = spadl_all_matches[spadl_all_matches.match_id == mid]
    ratio = len(actions) / len(events)
    assert 0.3 <= ratio <= 1.0, f"implausible SPADL/event ratio {ratio:.2f}"


def test_spadl_coordinates_in_pitch_bounds(spadl_all_matches):
    for col, hi in (("start_x", PITCH_LENGTH), ("end_x", PITCH_LENGTH),
                    ("start_y", PITCH_WIDTH), ("end_y", PITCH_WIDTH)):
        values = spadl_all_matches[col].dropna()
        assert values.min() >= -0.01
        assert values.max() <= hi + 0.01


def test_reference_frame_barcelona_attacks_positive_x(spadl_all_matches):
    """
    The whole project assumes one reference frame.  Barcelona shots must sit
    close to x = 105, opponent shots close to x = 0.
    """
    shots = spadl_all_matches[spadl_all_matches.type_name == "shot"]
    barca = shots[shots.team_id == TEAM_ID]["start_x"].mean()
    opponent = shots[shots.team_id != TEAM_ID]["start_x"].mean()
    assert barca > 80, f"Barcelona shots average x={barca:.1f}"
    assert opponent < 25, f"opponent shots average x={opponent:.1f}"


# --------------------------------------------------------------------------
# Task 1.3 — 360 merge
# --------------------------------------------------------------------------
def test_360_merge_coverage(merged_df):
    build_up_actions = merged_df[merged_df["type_name"].isin(["pass", "dribble"])]
    coverage = build_up_actions["freeze_frame"].notna().mean()
    assert coverage >= 0.70, f"360 coverage of build-up actions is only {coverage:.1%}"


def test_360_context_features_are_sane(merged_df):
    withff = merged_df[merged_df.has_freeze_frame]
    assert len(withff) > 0
    assert withff.ff_opponents_5m.max() <= 11
    assert withff.ff_opponents_5m.min() >= 0
    assert withff.ff_n_visible.median() >= 5
    assert withff.ff_nearest_opponent_dist.min() >= 0


# --------------------------------------------------------------------------
# Task 1.4 — players reference
# --------------------------------------------------------------------------
def test_players_reference_complete(players_reference_df, spadl_all_matches):
    """Every Barcelona player with a meaningful footprint must be referenced."""
    barca_actions = spadl_all_matches[spadl_all_matches.team_id == TEAM_ID]
    frequent_players = barca_actions["player_id"].value_counts()
    frequent_players = frequent_players[frequent_players >= 5].index
    assert set(int(p) for p in frequent_players).issubset(
        set(int(p) for p in players_reference_df["player_id"])
    )


def test_players_reference_core_players_covered(players_reference_df):
    core = players_reference_df[players_reference_df.is_core]
    assert (core.appearances >= MIN_APPEARANCES_FOR_REFERENCE).all()
    assert len(core) >= 15


def test_every_player_has_exactly_one_role(players_reference_df):
    assert players_reference_df["player_id"].is_unique
    assert players_reference_df["role"].notna().all()
    core = players_reference_df[players_reference_df.is_core]
    assert (core["role"] != "unknown").all()


def test_identity_is_separate_from_role(players_reference_df):
    """Technical document 2.6: identity and role are two independent columns."""
    assert {"player_atom", "role", "team"}.issubset(players_reference_df.columns)
    assert players_reference_df["player_atom"].is_unique
    assert players_reference_df["role"].nunique() >= 5
