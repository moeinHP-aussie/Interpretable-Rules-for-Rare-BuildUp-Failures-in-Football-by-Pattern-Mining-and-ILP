"""
Phase 1 regression tests for the round-2 fixes.

All tests are synthetic and run offline (no cache, no network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from tactical_ilp.config import PHASE1_VERSION
from tactical_ilp.data import load_raw_data
from tactical_ilp.data.freeze_frames import compute_context_features
from tactical_ilp.data.spadl_converter import (
    _compute_holding_intervals,
    _pass_height_name,
    _propagate_time_on_ball_to_dribbles,
    compute_time_on_ball,
    drop_locationless_events,
    order_actions,
    read_if_current,
)
from tactical_ilp.features.frames import _to_spadl_frame


def test_statsbomb_pass_height_is_extracted_from_event_metadata():
    assert _pass_height_name({"pass": {"height": {"name": "High Pass"}}}) == "High Pass"
    assert _pass_height_name({"pass": {"height": {"name": "Ground Pass"}}}) == "Ground Pass"
    assert _pass_height_name({"pass": {}}) is None
    assert _pass_height_name(None) is None


# --------------------------------------------------------------------------
# 360 merge: one event backing two SPADL rows must not double the counts
# --------------------------------------------------------------------------
def _frame_rows(event_id: str, n_opp: int, n_mate: int) -> list[dict]:
    rows = [
        {"event_id": event_id, "teammate": True, "actor": True, "keeper": False, "x": 60.0, "y": 40.0}
    ]
    rows += [
        {"event_id": event_id, "teammate": True, "actor": False, "keeper": False, "x": 70.0, "y": 20.0 + i}
        for i in range(n_mate)
    ]
    # opponents right next to the actor (inside 3 m)
    rows += [
        {"event_id": event_id, "teammate": False, "actor": False, "keeper": False, "x": 60.5, "y": 40.0 + 0.2 * i}
        for i in range(n_opp)
    ]
    return rows


def test_split_interception_pass_event_is_not_double_counted():
    actions = pd.DataFrame(
        {
            # interception + pass share one event id; third row is a normal event
            "original_event_id": ["e1", "e1", "e2"],
            "type_name": ["interception", "pass", "pass"],
            "start_x": [52.5, 52.5, 30.0],
            "start_y": [34.0, 34.0, 34.0],
            "is_barca": [True, True, True],
        }
    )
    frames = pd.DataFrame(_frame_rows("e1", n_opp=2, n_mate=3) + _frame_rows("e2", n_opp=1, n_mate=2))
    agg = compute_context_features(actions, frames)
    assert agg.loc["e1", "ff_n_visible"] == 1 + 3 + 2
    assert agg.loc["e2", "ff_n_visible"] == 1 + 2 + 1
    assert agg.loc["e1", "ff_opponents_5m"] <= 2


def test_context_features_empty_frames():
    actions = pd.DataFrame(
        {"original_event_id": ["e1"], "type_name": ["pass"], "start_x": [1.0], "start_y": [1.0], "is_barca": [True]}
    )
    agg = compute_context_features(actions, pd.DataFrame(columns=["event_id", "teammate", "actor", "keeper", "x", "y"]))
    assert len(agg) == 0
    assert {"ff_n_visible", "ff_opponents_5m", "ff_opponents_3m"} <= set(agg.columns)


def test_360_frame_mirroring_matches_spadl_home_team_transform():
    # SoccerAction 1.5.3 mirrors actions whose team_id differs from
    # home_team_id; the corresponding event-relative 360 coordinates need the
    # same team-side transform. The raw point (30, 20) becomes (26.25, 51)
    # for Barcelona and (78.75, 17) for the other team in SPADL coordinates.
    actions = pd.DataFrame(
        {
            "original_event_id": ["barca", "opponent"],
            "start_x": [26.25, 78.75],
            "start_y": [51.0, 17.0],
            "is_barca": [True, False],
        }
    )
    frames = pd.DataFrame(
        [
            {"event_id": event_id, "teammate": False, "actor": False,
             "keeper": False, "x": 30.0, "y": 20.0}
            for event_id in ("barca", "opponent")
        ]
    )
    agg = compute_context_features(actions, frames)
    assert agg.loc["barca", "ff_opponents_3m"] == 1
    assert agg.loc["opponent", "ff_opponents_3m"] == 1


def test_360_reference_frame_transform_covers_selected_home_and_away():
    # Raw point (30, 20) -> SPADL (26.25, 51) for a selected-team action
    # when selected team is home; opponent receives the 180-degree transform.
    # When selected team is away, both sides share the selected-away transform.
    assert _to_spadl_frame(30, 20, True, True) == pytest.approx((26.25, 51.0))
    assert _to_spadl_frame(30, 20, False, True) == pytest.approx((78.75, 17.0))
    assert _to_spadl_frame(30, 20, True, False) == pytest.approx((78.75, 17.0))
    assert _to_spadl_frame(30, 20, False, False) == pytest.approx((78.75, 17.0))


# --------------------------------------------------------------------------
# time on ball
# --------------------------------------------------------------------------
def _ev(rows):
    df = pd.DataFrame(
        rows,
        columns=["event_id", "index", "timestamp", "type_name", "player_id", "team_id", "possession", "extra"],
    )
    df["period_id"] = 1
    return df


def _ts(seconds: float) -> str:
    return str(pd.Timedelta(seconds=seconds))


def test_time_on_ball_carry_then_pass():
    # receipt at 1.0, carry 1.0 (dur 1.2), pass released at 2.2  -> 1.2 s on the ball
    ev = _ev(
        [
            ("p0", 1, _ts(0.0), "Pass", 9, 1, 1, {}),
            ("r1", 2, _ts(1.0), "Ball Receipt*", 10, 1, 1, {}),
            ("c1", 3, _ts(1.0), "Carry", 10, 1, 1, {}),
            ("p1", 4, _ts(2.2), "Pass", 10, 1, 1, {}),
        ]
    )
    tob = compute_time_on_ball(ev)
    assert tob.loc["p1"] == pytest.approx(1.2)
    assert "p0" not in tob.index  # no receipt before the first pass


def test_same_holder_carry_does_not_reset_receipt_timer():
    ev = _ev(
        [
            ("r1", 1, _ts(1.0), "Ball Receipt*", 10, 1, 1, {}),
            ("c1", 2, _ts(2.0), "Carry", 10, 1, 1, {}),
            ("p1", 3, _ts(3.5), "Pass", 10, 1, 1, {}),
        ]
    )
    assert compute_time_on_ball(ev).loc["p1"] == pytest.approx(2.5)


def test_holding_interval_is_propagated_to_matching_synthetic_dribble():
    ev = _ev(
        [
            ("r1", 1, _ts(1.0), "Ball Receipt*", 10, 1, 1, {}),
            ("p1", 2, _ts(4.0), "Pass", 10, 1, 1, {}),
        ]
    )
    intervals = _compute_holding_intervals(ev)
    actions = pd.DataFrame(
        {
            "action_source": ["synthetic_dribble", "synthetic_dribble"],
            "period_id": [1, 1],
            "player_id": [10, 11],
            "team_id": [1, 1],
            "possession": [1, 1],
            "time_seconds": [2.5, 2.5],
        }
    )
    dwell = _propagate_time_on_ball_to_dribbles(actions, intervals)
    assert dwell.iloc[0] == pytest.approx(3.0)
    assert pd.isna(dwell.iloc[1])


def test_time_on_ball_one_touch_is_near_zero():
    ev = _ev(
        [
            ("r1", 1, _ts(5.0), "Ball Receipt*", 10, 1, 1, {}),
            ("p1", 2, _ts(5.2), "Pass", 10, 1, 1, {}),
        ]
    )
    assert compute_time_on_ball(ev).loc["p1"] == pytest.approx(0.2)


def test_time_on_ball_not_attributed_after_ball_is_lost():
    # receiver is dispossessed; a later unrelated pass must not reuse the receipt
    ev = _ev(
        [
            ("r1", 1, _ts(1.0), "Ball Receipt*", 10, 1, 1, {}),
            ("d1", 2, _ts(2.0), "Dispossessed", 10, 1, 1, {}),
            ("p1", 3, _ts(4.0), "Pass", 10, 1, 1, {}),
        ]
    )
    assert "p1" not in compute_time_on_ball(ev).index


def test_time_on_ball_reset_by_other_player_and_possession():
    ev = _ev(
        [
            ("r1", 1, _ts(1.0), "Ball Receipt*", 10, 1, 1, {}),
            ("p2", 2, _ts(1.5), "Pass", 11, 1, 1, {}),  # someone else plays the ball
            ("p1", 3, _ts(2.0), "Pass", 10, 1, 1, {}),
            ("r3", 4, _ts(3.0), "Ball Receipt*", 12, 1, 1, {}),
            ("p3", 5, _ts(3.4), "Pass", 12, 1, 2, {}),  # possession id changed
        ]
    )
    tob = compute_time_on_ball(ev)
    assert "p1" not in tob.index
    assert "p3" not in tob.index


def test_time_on_ball_ignores_failed_receipt_and_is_capped():
    ev = _ev(
        [
            ("r1", 1, _ts(1.0), "Ball Receipt*", 10, 1, 1, {"ball_receipt": {"outcome": {"name": "Incomplete"}}}),
            ("p1", 2, _ts(1.4), "Pass", 10, 1, 1, {}),
            ("r2", 3, _ts(20.0), "Ball Receipt*", 10, 1, 1, {}),
            ("p2", 4, _ts(35.0), "Pass", 10, 1, 1, {}),  # 15 s > cap
        ]
    )
    tob = compute_time_on_ball(ev)
    assert len(tob) == 0


# --------------------------------------------------------------------------
# location-less events / deterministic ordering
# --------------------------------------------------------------------------
def test_locationless_ball_events_are_dropped_and_reported():
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "type_name": ["Pass", "Goal Keeper", "Substitution", "Half Start"],
            "location": [[10.0, 20.0], np.nan, np.nan, np.nan],
        }
    )
    clean, dropped = drop_locationless_events(events)
    assert clean["event_id"].tolist() == ["a"]
    # only the ball-action candidate is reported; administrative events are silent
    assert dropped["event_id"].tolist() == ["b"]


def test_order_actions_breaks_time_ties_by_event_index_and_keeps_synthetic_in_place():
    actions = pd.DataFrame(
        {
            "period_id": [1, 1, 1, 1, 1],
            # carry (idx 7) and an interception copy (idx 8) share the timestamp but
            # arrive in the wrong order; a synthetic dribble follows the carry
            "type_name": ["interception", "dribble", "dribble_syn", "pass", "pass"],
            "time_seconds": [10.0, 10.0, 10.4, 11.0, 12.0],
            "event_index": [8.0, 7.0, np.nan, 9.0, 10.0],
        }
    )
    out = order_actions(actions)
    assert out["event_index"].tolist()[:2] == [7.0, 8.0]
    assert out["type_name"].tolist() == ["dribble", "interception", "dribble_syn", "pass", "pass"]


def test_order_actions_is_stable_for_split_event_rows():
    actions = pd.DataFrame(
        {
            "period_id": [1, 1],
            "type_name": ["interception", "pass"],  # two rows of one event
            "time_seconds": [5.0, 5.0],
            "event_index": [3.0, 3.0],
        }
    )
    assert order_actions(actions)["type_name"].tolist() == ["interception", "pass"]


# --------------------------------------------------------------------------
# derived-cache versioning
# --------------------------------------------------------------------------
def test_stale_derived_cache_is_ignored(tmp_path):
    path = tmp_path / "spadl.parquet"
    df = pd.DataFrame({"match_id": [1, 1, 2], "x": [1, 2, 3], "phase1_version": "phase1-old"})
    df.to_parquet(path, index=False)
    assert read_if_current(path) is None

    df["phase1_version"] = PHASE1_VERSION
    df.to_parquet(path, index=False)
    assert read_if_current(path) is not None
    assert read_if_current(path, match_ids=[1, 2]) is not None
    assert read_if_current(path, match_ids=[1, 2, 3]) is None  # different match set
    assert read_if_current(tmp_path / "missing.parquet") is None


def test_cache_without_version_column_is_stale(tmp_path):
    path = tmp_path / "old.parquet"
    pd.DataFrame({"match_id": [1]}).to_parquet(path, index=False)
    assert read_if_current(path) is None


# --------------------------------------------------------------------------
# 360 download: only 404 means "no 360"
# --------------------------------------------------------------------------
def _http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status}", response=resp)


def test_load_frames_404_is_cached_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(load_raw_data, "cache_path", lambda kind, mid, suffix="parquet": tmp_path / f"{kind}_{mid}.{suffix}")

    def _raise(**kwargs):
        raise _http_error(404)

    import statsbombpy.sb as sb

    monkeypatch.setattr(sb, "frames", _raise)
    out = load_raw_data.load_frames(1)
    assert len(out) == 0
    assert (tmp_path / "frames_1.parquet").exists()


def test_load_frames_transient_error_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(load_raw_data, "cache_path", lambda kind, mid, suffix="parquet": tmp_path / f"{kind}_{mid}.{suffix}")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def _raise(**kwargs):
        raise requests.ConnectionError("network down")

    import statsbombpy.sb as sb

    monkeypatch.setattr(sb, "frames", _raise)
    with pytest.raises(requests.ConnectionError):
        load_raw_data.load_frames(2)
    assert not (tmp_path / "frames_2.parquet").exists()


def test_load_frames_server_error_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(load_raw_data, "cache_path", lambda kind, mid, suffix="parquet": tmp_path / f"{kind}_{mid}.{suffix}")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def _raise(**kwargs):
        raise _http_error(503)

    import statsbombpy.sb as sb

    monkeypatch.setattr(sb, "frames", _raise)
    with pytest.raises(requests.HTTPError):
        load_raw_data.load_frames(3)
    assert not (tmp_path / "frames_3.parquet").exists()


# --------------------------------------------------------------------------
# data-level checks (skipped until the rebuilt phase-1 tables exist)
# --------------------------------------------------------------------------
def test_rebuilt_tables_have_provenance_and_sane_counts(merged_df):
    if "action_source" not in merged_df.columns:
        pytest.skip("phase-1 tables not rebuilt with the current code (run_phase1.py --force)")
    assert merged_df["phase1_version"].eq(PHASE1_VERSION).all()
    assert set(merged_df["action_source"].unique()) <= {"event", "synthetic_dribble", "interception_split"}
    syn = merged_df[merged_df.action_source == "synthetic_dribble"]
    assert syn["original_event_id"].isna().all() and (syn["type_name"] == "dribble").all()
    split = merged_df[merged_df.action_source == "interception_split"]
    assert (split.groupby(["match_id", "original_event_id"]).size() == 2).all()
    assert merged_df["ff_n_visible"].max() <= 22
    coords = merged_df[["start_x", "start_y", "end_x", "end_y"]]
    assert not ((coords.abs() < 1e-100) & (coords != 0)).any().any(), "uninitialised-memory coordinates"
    assert coords.notna().all().all()
    tob = merged_df["time_on_ball_s"].dropna()
    assert len(tob) > 0 and tob.min() >= 0 and tob.max() <= 10.0
    # Only synthetic dribbles matched to an unambiguous holder interval gain a
    # dwell value; event-split interception rows remain unattributed.
    assert merged_df.loc[merged_df.action_source == "interception_split", "time_on_ball_s"].isna().all()
    synthetic_dwell = merged_df.loc[
        merged_df.action_source == "synthetic_dribble", "time_on_ball_s"
    ].dropna()
    assert synthetic_dwell.empty or synthetic_dwell.between(0, 10.0).all()
    # order guarantee: StatsBomb event index never decreases inside a period
    ev = merged_df.dropna(subset=["event_index"])
    assert (ev.groupby(["match_id", "period_id"])["event_index"].diff().dropna() >= 0).all()
