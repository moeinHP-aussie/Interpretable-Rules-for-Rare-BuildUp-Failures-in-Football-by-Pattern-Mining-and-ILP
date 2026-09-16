"""
Task 1.1 — Download and cache raw StatsBomb data for every Barcelona match
of La Liga 2020/21 (competition_id=11, season_id=90).

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
from pathlib import Path

import pandas as pd

from tactical_ilp.config import (
    COMPETITION_ID,
    RAW_DIR,
    SEASON_ID,
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
    """All Barcelona matches of the target season (cached)."""
    if MATCHES_CACHE.exists() and not force:
        return pd.read_parquet(MATCHES_CACHE)

    from socceraction.data.statsbomb import StatsBombLoader

    loader = StatsBombLoader(getter="remote", creds={})
    games = loader.games(competition_id=COMPETITION_ID, season_id=SEASON_ID)

    teams = loader.teams(games.game_id.iloc[0])  # bootstrap team names
    name_by_id = dict(zip(teams.team_id, teams.team_name))
    # team names for every game
    all_names: dict[int, str] = {}
    for gid in games.game_id:
        pass
    games = games.copy()
    games["home_team"] = games.home_team_id.map(name_by_id)
    games["away_team"] = games.away_team_id.map(name_by_id)

    # the bootstrap only knew two teams; fill the rest from statsbombpy
    if games[["home_team", "away_team"]].isna().any().any():
        from statsbombpy import sb

        sb_matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        m = sb_matches.set_index("match_id")
        games["home_team"] = games.game_id.map(m["home_team"])
        games["away_team"] = games.game_id.map(m["away_team"])

    games.to_parquet(MATCHES_CACHE, index=False)
    return games


def barca_match_ids(force: bool = False) -> list[int]:
    """Sorted list of match ids in which Barcelona played."""
    games = load_matches(force=force)
    mask = (games.home_team == TEAM_NAME) | (games.away_team == TEAM_NAME)
    return sorted(int(x) for x in games.loc[mask, "game_id"])


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

    from statsbombpy import sb

    try:
        raw = sb.frames(match_id=match_id, fmt="dataframe")
    except Exception:  # no 360 file for this match
        raw = pd.DataFrame(columns=FRAME_COLUMNS)

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
        rows.append(
            {
                "match_id": mid,
                "n_events": len(ev),
                "n_players": len(lu),
                "n_frame_rows": len(fr),
                "n_frame_events": fr["event_id"].nunique() if len(fr) else 0,
                "has_360": len(fr) > 0,
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
