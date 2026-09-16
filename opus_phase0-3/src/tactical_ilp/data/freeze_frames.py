"""
Task 1.3 — Link StatsBomb 360 freeze-frames to SPADL actions.

The 360 file gives, for (almost) every event, the position of the players that
were inside the broadcast camera frame at that exact instant.  That is the only
off-the-ball information available in the MVP (technical document §2.4): a
snapshot, not a continuous trajectory.

For every SPADL action we derive a small set of *context features* from the
freeze frame; phase 3 turns these numbers into Prolog facts.

Coordinate handling
-------------------
Freeze-frame coordinates are raw StatsBomb (120 x 80) and are expressed from the
point of view of the team that performed the event.  We therefore

    1. rescale   ->  SPADL 105 x 68 with the y axis flipped
                     (identical formula to socceraction's event conversion)
    2. mirror    ->  if the acting team is not Barcelona, so that the whole
                     project lives in one reference frame (Barcelona -> +x).

``teammate`` in the raw file is relative to the acting player, which is exactly
the notion of "opponent" we need for a pressure model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PRESSURE_MIN_OPPONENTS,
    PRESSURE_RADIUS_M,
    PROCESSED_DIR,
    TIGHT_PRESSURE_RADIUS_M,
    cache_path,
)
from tactical_ilp.data.load_raw_data import barca_match_ids, load_frames
from tactical_ilp.data.spadl_converter import build_spadl

MERGED_ALL_PATH = PROCESSED_DIR / "spadl_with_360_all_matches.parquet"

CONTEXT_COLUMNS = [
    "ff_n_visible",
    "ff_opponents_5m",
    "ff_opponents_3m",
    "ff_teammates_5m",
    "ff_nearest_opponent_dist",
    "ff_opponents_ahead",
    "ff_teammates_ahead",
]


def frames_to_spadl_coordinates(frames: pd.DataFrame) -> pd.DataFrame:
    """Rescale raw 120x80 StatsBomb coordinates to SPADL 105x68."""
    out = frames.copy()
    out["fx"] = out["x"] / 120.0 * PITCH_LENGTH
    out["fy"] = PITCH_WIDTH - (out["y"] / 80.0 * PITCH_WIDTH)
    return out


def build_merged(match_id: int, force: bool = False) -> pd.DataFrame:
    """SPADL actions of one match plus their 360 context features."""
    path = cache_path("merged", match_id)
    if path.exists() and not force:
        return pd.read_parquet(path)

    actions = build_spadl(match_id)
    frames = load_frames(match_id)

    actions = actions.copy()
    actions["has_freeze_frame"] = False
    for col in CONTEXT_COLUMNS:
        actions[col] = np.nan

    if len(frames) == 0:
        actions.to_parquet(path, index=False)
        return actions

    frames = frames_to_spadl_coordinates(frames)

    ref = actions[
        ["original_event_id", "start_x", "start_y", "is_barca"]
    ].dropna(subset=["original_event_id"])
    ref = ref.rename(columns={"original_event_id": "event_id"})
    ref["event_id"] = ref["event_id"].astype(str)

    ff = frames.merge(ref, on="event_id", how="inner")
    if len(ff) == 0:
        actions.to_parquet(path, index=False)
        return actions

    # mirror opponent-team frames into Barcelona's reference frame
    flip = ~ff["is_barca"].to_numpy()
    ff.loc[flip, "fx"] = PITCH_LENGTH - ff.loc[flip, "fx"]
    ff.loc[flip, "fy"] = PITCH_WIDTH - ff.loc[flip, "fy"]

    ff["dist"] = np.hypot(ff["fx"] - ff["start_x"], ff["fy"] - ff["start_y"])
    # "ahead" = closer to the goal the acting team is attacking
    ff["ahead"] = np.where(
        ff["is_barca"], ff["fx"] > ff["start_x"], ff["fx"] < ff["start_x"]
    )
    is_opp = ~ff["teammate"].to_numpy()
    is_mate = ff["teammate"].to_numpy() & ~ff["actor"].to_numpy()

    grp = ff.groupby("event_id")
    agg = pd.DataFrame(
        {
            "ff_n_visible": grp.size(),
            "ff_opponents_5m": ff[is_opp & (ff["dist"] <= PRESSURE_RADIUS_M)]
            .groupby("event_id")
            .size(),
            "ff_opponents_3m": ff[is_opp & (ff["dist"] <= TIGHT_PRESSURE_RADIUS_M)]
            .groupby("event_id")
            .size(),
            "ff_teammates_5m": ff[is_mate & (ff["dist"] <= PRESSURE_RADIUS_M)]
            .groupby("event_id")
            .size(),
            "ff_nearest_opponent_dist": ff[is_opp].groupby("event_id")["dist"].min(),
            "ff_opponents_ahead": ff[is_opp & ff["ahead"]].groupby("event_id").size(),
            "ff_teammates_ahead": ff[is_mate & ff["ahead"]].groupby("event_id").size(),
        }
    )
    count_cols = [c for c in agg.columns if c != "ff_nearest_opponent_dist"]
    agg[count_cols] = agg[count_cols].fillna(0)

    actions = actions.drop(columns=CONTEXT_COLUMNS + ["has_freeze_frame"])
    key = actions["original_event_id"].astype(str)
    actions = actions.join(agg, on=key.rename("event_id"))
    actions["has_freeze_frame"] = actions["ff_n_visible"].notna()
    actions["freeze_frame"] = actions["ff_n_visible"]  # alias used by the phase-1 test

    actions.to_parquet(path, index=False)
    return actions


def build_all(match_ids: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    if MERGED_ALL_PATH.exists() and not force:
        return pd.read_parquet(MERGED_ALL_PATH)

    ids = match_ids if match_ids is not None else barca_match_ids()
    out = []
    for i, mid in enumerate(ids, 1):
        df = build_merged(mid, force=force)
        cov = df[df.type_name.isin(["pass", "dribble"])]["has_freeze_frame"].mean()
        out.append(df)
        print(f"[{i:>2}/{len(ids)}] match {mid}: build-up 360 coverage {cov:.1%}")
    merged = pd.concat(out, ignore_index=True)
    merged.to_parquet(MERGED_ALL_PATH, index=False)
    return merged


def under_pressure_from_frame(row) -> bool:
    """Pressure model used by phase 3 (falls back to StatsBomb's own flag)."""
    n = row.get("ff_opponents_5m")
    if n is not None and not pd.isna(n):
        return bool(n >= PRESSURE_MIN_OPPONENTS)
    return bool(row.get("under_pressure", False))


if __name__ == "__main__":
    merged = build_all()
    build_up = merged[merged.type_name.isin(["pass", "dribble"])]
    print()
    print(f"total actions          : {len(merged)}")
    print(f"build-up actions       : {len(build_up)}")
    print(f"360 coverage (build-up): {build_up['has_freeze_frame'].mean():.2%}")
    print(f"360 coverage (all)     : {merged['has_freeze_frame'].mean():.2%}")
