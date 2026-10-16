"""
Task 1.1 — Download and cache raw StatsBomb data for every match played by
the selected team in its configured competition and season.

Three artefacts are cached per match under ``data/raw`` :

    events_<match_id>.parquet    raw event stream (socceraction schema)
    lineup_<match_id>.parquet    line-ups + starting positions
    frames_<match_id>.parquet    StatsBomb 360 freeze-frames (one row per player)

The second run of this module must not touch the network at all: every public
function checks the cache first and only downloads what is missing.

Data source: StatsBomb Open Data (https://github.com/statsbomb/open-data).
Please credit StatsBomb when publishing any result derived from it.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd

from tactical_ilp.config import (
    COMPETITION_ID,
    MIN_FRAME_LINKAGE,
    RAW_DIR,
    SEASON_ID,
    TEAM_ID,
    TEAM_NAME,
    cache_path,
)

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

MATCHES_CACHE = RAW_DIR / "matches.parquet"

# columns of the 360 frame table we keep (visible_area is huge and unused)
FRAME_COLUMNS = ["event_id", "teammate", "actor", "keeper", "x", "y"]


# --------------------------------------------------------------------------
# match list
# --------------------------------------------------------------------------
def load_matches(force: bool = False) -> pd.DataFrame:
    """All matches of the target competition-season with team names (cached)."""
    if MATCHES_CACHE.exists() and not force:
        cached = pd.read_parquet(MATCHES_CACHE)
        if {"home_team_id", "away_team_id"}.issubset(cached.columns):
            return cached

    from socceraction.data.statsbomb import StatsBombLoader
    from statsbombpy import sb

    loader = StatsBombLoader(getter="remote", creds={})
    games = loader.games(competition_id=COMPETITION_ID, season_id=SEASON_ID).copy()

    # The game list carries ids; get names for every fixture from the same
    # competition/season match table instead of bootstrapping two team names.
    sb_matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    names = sb_matches.set_index("match_id")
    games["home_team"] = games.game_id.map(names["home_team"])
    games["away_team"] = games.game_id.map(names["away_team"])
    if games[["home_team", "away_team"]].isna().any().any():
        missing = games.loc[
            games[["home_team", "away_team"]].isna().any(axis=1), "game_id"
        ].tolist()
        raise RuntimeError(f"team names missing for games {missing}")

    games.to_parquet(MATCHES_CACHE, index=False)
    return games


def barca_match_ids(force: bool = False) -> list[int]:
    """Sorted match ids for the selected team in the configured season."""
    games = load_matches(force=force)
    mask = (games.home_team == TEAM_NAME) | (games.away_team == TEAM_NAME)
    return sorted(int(x) for x in games.loc[mask, "game_id"])


@lru_cache(maxsize=256)
def selected_team_is_home(match_id: int) -> bool:
    """Whether the configured analysis team is the actual home side."""
    games = load_matches()
    row = games.loc[games.game_id.astype(int) == int(match_id)]
    if row.empty or not {"home_team_id", "away_team_id"}.issubset(games.columns):
        raise ValueError(f"match {match_id} has no cached home/away team IDs")
    home_id = int(row.iloc[0].home_team_id)
    away_id = int(row.iloc[0].away_team_id)
    if TEAM_ID == home_id:
        return True
    if TEAM_ID == away_id:
        return False
    raise ValueError(f"configured team {TEAM_ID} is not a side in match {match_id}")


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
def _serialise_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet cannot store arbitrary python dicts/lists -> JSON-encode them."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], (dict, list)):
                out[col] = out[col].apply(
                    lambda v: json.dumps(v, default=str) if v is not None else None
                )
    return out


def _deserialise_json_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v else v
            )
    return out


def load_events(match_id: int, force: bool = False) -> pd.DataFrame:
    """Raw events for one match (socceraction schema), cached as parquet."""
    path = cache_path("events", match_id)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        return _deserialise_json_columns(df, ["extra", "related_events", "location"])

    from socceraction.data.statsbomb import StatsBombLoader

    loader = StatsBombLoader(getter="remote", creds={})
    events = loader.events(match_id)
    _serialise_object_columns(events).to_parquet(path, index=False)
    return events


def load_lineup(match_id: int, force: bool = False) -> pd.DataFrame:
    """Players of both teams with their starting position, cached."""
    path = cache_path("lineup", match_id)
    if path.exists() and not force:
        return pd.read_parquet(path)

    from socceraction.data.statsbomb import StatsBombLoader

    loader = StatsBombLoader(getter="remote", creds={})
    players = loader.players(match_id)
    _serialise_object_columns(players).to_parquet(path, index=False)
    return players


# --------------------------------------------------------------------------
# StatsBomb 360 freeze frames
# --------------------------------------------------------------------------
def _fetch_frames(
    match_id: int, retries: int = 3, backoff_s: float = 2.0
) -> pd.DataFrame:
    """Fetch frames, treating only 404 as a valid absence of 360 data."""
    import time

    import requests
    from statsbombpy import sb

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return sb.frames(match_id=match_id, fmt="dataframe")
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 404:
                return pd.DataFrame(columns=FRAME_COLUMNS)
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        if attempt + 1 < retries:
            time.sleep(backoff_s * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"no attempts made to download 360 frames for match {match_id}")


def load_frames(match_id: int, force: bool = False) -> pd.DataFrame:
    """
    StatsBomb 360 freeze-frames for one match.

    Returns a long table: one row per visible player per event, with columns
    ``event_id, teammate, actor, keeper, x, y``.  ``x``/``y`` are still in
    StatsBomb pitch units (120 x 80) — they are converted to SPADL units in
    :mod:`tactical_ilp.data.freeze_frames`.

    An empty DataFrame is cached when the match has no 360 data, so we never
    re-attempt the download.
    """
    path = cache_path("frames", match_id)
    if path.exists() and not force:
        return pd.read_parquet(path)

    raw = _fetch_frames(match_id)

    if len(raw) == 0:
        out = pd.DataFrame(columns=FRAME_COLUMNS)
    else:
        loc = raw["location"].apply(
            lambda v: (float(v[0]), float(v[1]))
            if isinstance(v, (list, tuple)) and len(v) >= 2
            else (float("nan"), float("nan"))
        )
        out = pd.DataFrame(
            {
                "event_id": raw["id"].astype(str),
                "teammate": raw["teammate"].astype(bool),
                "actor": raw["actor"].astype(bool),
                "keeper": raw["keeper"].astype(bool),
                "x": [p[0] for p in loc],
                "y": [p[1] for p in loc],
            }
        )

    out.to_parquet(path, index=False)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def download_all(match_ids: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Populate the whole cache and return a small status table."""
    ids = match_ids if match_ids is not None else barca_match_ids(force=force)
    rows = []
    for i, mid in enumerate(ids, 1):
        ev = load_events(mid, force=force)
        lu = load_lineup(mid, force=force)
        fr = load_frames(mid, force=force)
        # share of frame event ids that exist in the event stream; a 360 file that
        # does not link to the events is as good as missing
        if len(fr):
            linkage = float(
                fr["event_id"].astype(str).drop_duplicates().isin(set(ev["event_id"].astype(str))).mean()
            )
        else:
            linkage = 0.0
        rows.append(
            {
                "match_id": mid,
                "n_events": len(ev),
                "n_players": len(lu),
                "n_frame_rows": len(fr),
                "n_frame_events": fr["event_id"].nunique() if len(fr) else 0,
                "frame_linkage": linkage,
                "has_360": linkage >= MIN_FRAME_LINKAGE,
            }
        )
        print(
            f"[{i:>2}/{len(ids)}] match {mid}: "
            f"{len(ev):>5} events, {len(fr):>6} 360-rows"
        )
    status = pd.DataFrame(rows)
    status.to_parquet(RAW_DIR / "download_status.parquet", index=False)
    return status


def cache_is_complete(match_ids: list[int] | None = None) -> bool:
    ids = match_ids if match_ids is not None else barca_match_ids()
    return all(
        cache_path(kind, mid).exists()
        for mid in ids
        for kind in ("events", "lineup", "frames")
    )


if __name__ == "__main__":
    status = download_all()
    print()
    print(status.to_string(index=False))
    print(f"\n360 coverage: {status.has_360.sum()}/{len(status)} matches")
