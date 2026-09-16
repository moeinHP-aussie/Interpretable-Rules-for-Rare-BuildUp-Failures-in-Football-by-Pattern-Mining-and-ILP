"""
Task 1.2 — Convert raw StatsBomb events into the standard SPADL action table.

One row = one on-the-ball action with a fixed schema
(player_id, team_id, type_name, start_x, start_y, end_x, end_y, result_name, ...).

Direction of play
-----------------
``convert_to_actions(events, home_team_id)`` already calls
``_fix_direction_of_play``, so by passing **Barcelona's team id** we obtain a
single, stable reference frame for the whole project:

    * Barcelona always attacks towards +x  (x = 105 is the opponent goal)
    * x = 0 is Barcelona's own goal, also for opponent actions
    * y = 0 is the right touchline seen from Barcelona's attacking direction

Every geometric predicate in phase 3 relies on this convention.
"""

from __future__ import annotations

import warnings

import pandas as pd

from tactical_ilp.config import PROCESSED_DIR, TEAM_ID, cache_path
from tactical_ilp.data.load_raw_data import barca_match_ids, load_events

warnings.filterwarnings("ignore", category=FutureWarning)

REQUIRED_COLUMNS = {
    "player_id",
    "team_id",
    "type_name",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "result_name",
}

SPADL_ALL_PATH = PROCESSED_DIR / "spadl_all_matches.parquet"


def build_spadl(match_id: int, force: bool = False) -> pd.DataFrame:
    """SPADL action table for one match, oriented in Barcelona's frame."""
    path = cache_path("spadl", match_id)
    if path.exists() and not force:
        return pd.read_parquet(path)

    import socceraction.spadl as spadl

    events = load_events(match_id)
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id=TEAM_ID)
    actions = spadl.add_names(actions)

    # carry over StatsBomb flags needed later:
    #   under_pressure / counterpress -> pressure model (phase 3)
    #   possession / possession_team_id -> sequence segmentation (phase 2)
    flags = events.set_index("event_id")[
        ["under_pressure", "counterpress", "possession", "possession_team_id"]
    ]
    actions = actions.join(flags, on="original_event_id")
    actions["under_pressure"] = actions["under_pressure"].fillna(False).astype(bool)
    actions["counterpress"] = actions["counterpress"].fillna(False).astype(bool)
    actions["possession"] = actions["possession"].ffill().bfill().astype("Int64")
    actions["possession_team_id"] = (
        actions["possession_team_id"].ffill().bfill().astype("Int64")
    )
    actions["match_id"] = match_id

    actions["is_barca"] = actions["team_id"] == TEAM_ID
    actions = actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    actions["action_id"] = range(len(actions))

    actions.to_parquet(path, index=False)
    return actions


def build_all(match_ids: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Concatenated SPADL table for every Barcelona match."""
    if SPADL_ALL_PATH.exists() and not force:
        return pd.read_parquet(SPADL_ALL_PATH)

    ids = match_ids if match_ids is not None else barca_match_ids()
    frames = []
    for i, mid in enumerate(ids, 1):
        df = build_spadl(mid, force=force)
        frames.append(df)
        print(f"[{i:>2}/{len(ids)}] match {mid}: {len(df):>5} SPADL actions")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(SPADL_ALL_PATH, index=False)
    return out


if __name__ == "__main__":
    df = build_all()
    print()
    print(f"total actions: {len(df)}")
    print(df["type_name"].value_counts().head(12).to_string())
    barca = df[df.is_barca]
    print(f"\nBarcelona actions: {len(barca)}  ({len(barca)/len(df):.1%})")
    shots = df[df.type_name == "shot"]
    print("\nsanity check — mean shot x by side (Barca must be high, opponent low):")
    print(shots.groupby("is_barca")["start_x"].mean().to_string())
