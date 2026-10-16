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
    2. mirror    ->  opponent-event frames are rotated by 180 degrees into the
                     selected team's reference frame (+x). This does NOT depend
                     on home/away: socceraction already puts the selected team
                     at +x in every match.

``teammate`` in the raw file is relative to the acting player, which is exactly
the notion of "opponent" we need for a pressure model.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    EXCLUDE_UNLINKED_360_MATCHES,
    MIN_FRAME_LINKAGE,
    PHASE1_VERSION,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PRESSURE_RADIUS_M,
    PROCESSED_DIR,
    TIGHT_PRESSURE_RADIUS_M,
    cache_path,
)
from tactical_ilp.data.load_raw_data import barca_match_ids, load_events, load_frames
from tactical_ilp.data.spadl_converter import (
    OrientationError,
    build_spadl,
    read_if_current,
)

MERGED_ALL_PATH = PROCESSED_DIR / "spadl_with_360_all_matches.parquet"

log = logging.getLogger(__name__)

# A match whose 360 file shares (almost) no event ids with the event stream
# cannot give pressure features. Such matches are reported and, by default,
# excluded from the analysis (see ``build_all`` and config.MIN_FRAME_LINKAGE).
EXCLUDED_MATCHES_PATH = PROCESSED_DIR / "excluded_matches_360.json"

CONTEXT_COLUMNS = [
    "ff_n_visible",
    "ff_opponents_5m",
    "ff_opponents_3m",
]


def frames_to_spadl_coordinates(frames: pd.DataFrame) -> pd.DataFrame:
    """Rescale raw 120x80 StatsBomb coordinates to SPADL 105x68."""
    out = frames.copy()
    out["fx"] = out["x"] / 120.0 * PITCH_LENGTH
    out["fy"] = PITCH_WIDTH - (out["y"] / 80.0 * PITCH_WIDTH)
    return out


def _assert_actor_alignment(ff: pd.DataFrame, match_id: int, tol_m: float = 3.0) -> None:
    """Median distance between actor frame point and action start must be ~0."""
    actor = ff[ff["actor"].to_numpy()]
    if len(actor) < 20:
        return
    err = np.hypot(actor["fx"] - actor["start_x"], actor["fy"] - actor["start_y"])
    med = float(err.median())
    if med > tol_m:
        raise OrientationError(
            f"match {match_id}: 360 frames are not aligned with the SPADL actions "
            f"(median actor-vs-action distance {med:.1f} m, expected < {tol_m} m)"
        )


def frame_linkage(event_ids, frames: pd.DataFrame) -> float:
    """Share of the match's 360 frame event ids that exist in the RAW event stream.

    ``event_ids`` are the ids of ALL raw events (not only of the SPADL actions:
    360 also covers events that SPADL drops, such as receipts or pressures).
    ~1.0 -> the files belong together; 0.0 -> the 360 file uses a different
    event numbering (happens in some open-data matches).
    """
    if len(frames) == 0:
        return 0.0
    ids = set(pd.Series(list(event_ids)).dropna().astype(str))
    fids = frames["event_id"].astype(str).drop_duplicates()
    return float(fids.isin(ids).mean())


def compute_context_features(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each 360 frame once, keyed by its StatsBomb event id."""
    if len(frames) == 0:
        return pd.DataFrame(columns=CONTEXT_COLUMNS, dtype=float)

    frames = frames_to_spadl_coordinates(frames)

    ref = (
        actions[["original_event_id", "start_x", "start_y", "is_barca"]]
        .dropna(subset=["original_event_id"])
        .rename(columns={"original_event_id": "event_id"})
    )
    ref["event_id"] = ref["event_id"].astype(str)
    # Some events (notably pass/interception splits) back multiple SPADL rows.
    # Joining both copies to the frame doubles every visible-player count.
    ref = ref.drop_duplicates("event_id", keep="first")

    ff = frames.merge(ref, on="event_id", how="inner")
    if len(ff) == 0:
        return pd.DataFrame(columns=CONTEXT_COLUMNS, dtype=float)

    # Frames of opponent events are normalised to the OPPONENT's attack
    # direction; rotate them into the selected team's frame. Frames of the
    # selected team's own events already are in that frame. Home/away is
    # irrelevant (the SPADL actions are +x for the selected team in all matches).
    match_id = int(actions.match_id.iloc[0])
    flip = ~ff["is_barca"].to_numpy()
    ff.loc[flip, "fx"] = PITCH_LENGTH - ff.loc[flip, "fx"]
    ff.loc[flip, "fy"] = PITCH_WIDTH - ff.loc[flip, "fy"]

    # Per-match alignment check: the acting player's frame position must sit
    # on the action's own SPADL start point (measured median error ~0.06 m).
    _assert_actor_alignment(ff, match_id)

    ff["dist"] = np.hypot(ff["fx"] - ff["start_x"], ff["fy"] - ff["start_y"])
    is_opp = ~ff["teammate"].to_numpy()

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
        }
    )
    agg[list(agg.columns)] = agg[list(agg.columns)].fillna(0)
    return agg


def build_merged(match_id: int, force: bool = False) -> pd.DataFrame:
    """SPADL actions of one match plus their 360 context features."""
    path = cache_path("merged", match_id)
    if not force:
        cached = read_if_current(path)
        if cached is not None:
            return cached

    actions = build_spadl(match_id).copy()
    frames = load_frames(match_id)
    linkage = frame_linkage(load_events(match_id)["event_id"], frames)
    if linkage < MIN_FRAME_LINKAGE:
        log.warning(
            "match %s: only %.0f%% of the 360 frame event ids exist in the event "
            "stream -> no usable 360 data for this match",
            match_id, 100 * linkage,
        )
    agg = compute_context_features(actions, frames) if linkage >= MIN_FRAME_LINKAGE \
        else pd.DataFrame(columns=CONTEXT_COLUMNS, dtype=float)

    actions = actions.drop(
        columns=[c for c in CONTEXT_COLUMNS + ["has_freeze_frame", "freeze_frame"] if c in actions]
    )
    key = actions["original_event_id"].astype(str).rename("event_id")
    actions = actions.join(agg, on=key.rename("event_id"))
    for col in CONTEXT_COLUMNS:
        if col not in actions:
            actions[col] = np.nan
    actions["has_freeze_frame"] = actions["ff_n_visible"].notna()
    actions["freeze_frame"] = actions["ff_n_visible"]  # alias used by the phase-1 test
    actions["frame_linkage"] = linkage
    actions["phase1_version"] = PHASE1_VERSION

    actions.to_parquet(path, index=False)
    return actions


def _excluded_ids() -> set[int]:
    if EXCLUDED_MATCHES_PATH.exists():
        try:
            return {int(x) for x in json.loads(EXCLUDED_MATCHES_PATH.read_text())["match_ids"]}
        except (ValueError, KeyError, OSError):
            return set()
    return set()


def build_all(match_ids: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Merged SPADL+360 table of all matches (matches without usable 360 excluded).

    A match is excluded when the share of its 360 frame event ids found in the
    event stream is below ``MIN_FRAME_LINKAGE`` (the 360 file then belongs to a
    different event numbering). The exclusion list is written to
    ``excluded_matches_360.json`` and printed; set
    ``TACTICAL_ILP_EXCLUDE_UNLINKED_360=0`` to keep those matches anyway.
    """
    ids = match_ids if match_ids is not None else barca_match_ids()
    if not force:
        expected = [m for m in ids if int(m) not in _excluded_ids()] \
            if EXCLUDE_UNLINKED_360_MATCHES else ids
        cached = read_if_current(MERGED_ALL_PATH, expected)
        if cached is not None:
            return cached
    out = []
    excluded: list[int] = []
    for i, mid in enumerate(ids, 1):
        df = build_merged(mid, force=force)
        cov = df[df.type_name.isin(["pass", "dribble"])]["has_freeze_frame"].mean()
        link = float(df["frame_linkage"].iloc[0]) if "frame_linkage" in df and len(df) else 0.0
        if EXCLUDE_UNLINKED_360_MATCHES and link < MIN_FRAME_LINKAGE:
            excluded.append(int(mid))
            print(f"[{i:>2}/{len(ids)}] match {mid}: EXCLUDED (360 linkage {link:.0%})")
            continue
        out.append(df)
        print(f"[{i:>2}/{len(ids)}] match {mid}: build-up 360 coverage {cov:.1%}")
    EXCLUDED_MATCHES_PATH.write_text(json.dumps({"match_ids": excluded}))
    if excluded:
        print(f"\n!! {len(excluded)} match(es) excluded for missing 360 linkage: {excluded}")
    merged = pd.concat(out, ignore_index=True)
    merged.to_parquet(MERGED_ALL_PATH, index=False)
    return merged


if __name__ == "__main__":
    merged = build_all()
    build_up = merged[merged.type_name.isin(["pass", "dribble"])]
    print()
    print(f"total actions          : {len(merged)}")
    print(f"build-up actions       : {len(build_up)}")
    print(f"360 coverage (build-up): {build_up['has_freeze_frame'].mean():.2%}")
    print(f"360 coverage (all)     : {merged['has_freeze_frame'].mean():.2%}")
