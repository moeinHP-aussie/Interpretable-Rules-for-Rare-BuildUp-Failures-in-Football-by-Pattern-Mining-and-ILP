"""
Shared helper for the phase-4 cost estimator: load raw per-player (x, y)
freeze-frame positions, grouped by action, in SPADL coordinates.

This does NOT touch tactical_ilp.data.freeze_frames.build_merged, which only
keeps aggregated counts (ff_opponents_5m, ...). Several concepts in the
tactical glossary (triangulation 3.6, defensive-line clustering 0.1,
inter-player distances 2.3پ) need the individual player positions, which
still exist in the per-match raw frame files (data/raw/frames_<id>.parquet)
but are dropped before the merged parquet is written.

This module is read-only estimation tooling: it does not change
facts_all.pl or any phase 1-3 artefact. It answers one question only --
"if we built fact X, how many rows would it produce?" -- using a sample of
matches so the check stays fast.
"""

from __future__ import annotations

import pandas as pd

from tactical_ilp.config import PITCH_LENGTH, PITCH_WIDTH, RAW_DIR, TEAM_ID
from tactical_ilp.data.freeze_frames import frames_to_spadl_coordinates
from tactical_ilp.data.load_raw_data import barca_match_ids, load_frames
from tactical_ilp.data.spadl_converter import build_spadl


def raw_positions_for_match(match_id: int) -> pd.DataFrame:
    """
    One row per (action, visible player), with SPADL-oriented (x, y),
    is_teammate, is_actor -- everything mirrored into Barcelona's reference
    frame exactly like tactical_ilp.data.freeze_frames.build_merged does
    internally, but WITHOUT collapsing to aggregate counts.
    """
    actions = build_spadl(match_id)
    frames = load_frames(match_id)
    if len(frames) == 0:
        return pd.DataFrame(
            columns=[
                "match_id",
                "action_id",
                "player_slot",
                "x",
                "y",
                "is_teammate",
                "is_actor",
            ]
        )

    frames = frames_to_spadl_coordinates(frames)

    ref = actions[["original_event_id", "action_id", "is_barca"]].dropna(
        subset=["original_event_id"]
    )
    ref = ref.rename(columns={"original_event_id": "event_id"})
    ref["event_id"] = ref["event_id"].astype(str)

    joined = frames.merge(ref, on="event_id", how="inner")
    if len(joined) == 0:
        return pd.DataFrame(
            columns=[
                "match_id",
                "action_id",
                "player_slot",
                "x",
                "y",
                "is_teammate",
                "is_actor",
            ]
        )

    flip = ~joined["is_barca"].to_numpy()
    joined.loc[flip, "fx"] = PITCH_LENGTH - joined.loc[flip, "fx"]
    joined.loc[flip, "fy"] = PITCH_WIDTH - joined.loc[flip, "fy"]

    out = pd.DataFrame(
        {
            "match_id": match_id,
            "action_id": joined["action_id"].astype(int),
            "x": joined["fx"],
            "y": joined["fy"],
            "is_teammate": joined["teammate"].astype(bool),
            "is_actor": joined["actor"].astype(bool),
        }
    )
    out["player_slot"] = out.groupby("action_id").cumcount()
    return out


def raw_positions_sample(n_matches: int = 5, seed: int = 202021) -> pd.DataFrame:
    """
    Raw per-player positions for a deterministic sample of matches -- enough
    to estimate fact volume without re-reading all 35 matches every time.
    """
    ids = barca_match_ids()
    if n_matches >= len(ids):
        sample_ids = ids
    else:
        import random

        rng = random.Random(seed)
        sample_ids = sorted(rng.sample(ids, n_matches))

    frames = [raw_positions_for_match(mid) for mid in sample_ids]
    return pd.concat(frames, ignore_index=True), sample_ids
