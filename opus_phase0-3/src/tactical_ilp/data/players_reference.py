"""
Task 1.4 — Build the ``players_reference`` table.

Technical document §2.6 asks for player **identity** to be modelled as a
dimension that is independent of the tactical **role**, so that Popper can
learn both

    "defensive midfielders do X"          (role-level rule)
    "this specific player does X"         (identity-level rule)

The table produced here is the single source of truth for the Prolog facts

    player_identity(p_5203).
    has_role(p_5203, defensive_midfielder).
    plays_for(p_5203, barcelona).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from tactical_ilp.config import (
    MIN_APPEARANCES_FOR_REFERENCE,
    PROCESSED_DIR,
    TEAM_ID,
)
from tactical_ilp.data.load_raw_data import barca_match_ids, load_lineup

PLAYERS_REFERENCE_PATH = PROCESSED_DIR / "players_reference.parquet"

# StatsBomb position name -> tactical role used throughout the project
ROLE_MAP: dict[str, str] = {
    "Goalkeeper": "goalkeeper",
    "Right Back": "full_back",
    "Left Back": "full_back",
    "Right Wing Back": "full_back",
    "Left Wing Back": "full_back",
    "Center Back": "central_defender",
    "Left Center Back": "central_defender",
    "Right Center Back": "central_defender",
    "Left Defensive Midfield": "defensive_midfielder",
    "Right Defensive Midfield": "defensive_midfielder",
    "Center Defensive Midfield": "defensive_midfielder",
    "Left Center Midfield": "central_midfielder",
    "Right Center Midfield": "central_midfielder",
    "Center Midfield": "central_midfielder",
    "Left Midfield": "wide_midfielder",
    "Right Midfield": "wide_midfielder",
    "Left Attacking Midfield": "attacking_midfielder",
    "Right Attacking Midfield": "attacking_midfielder",
    "Center Attacking Midfield": "attacking_midfielder",
    "Left Wing": "winger",
    "Right Wing": "winger",
    "Left Center Forward": "forward",
    "Right Center Forward": "forward",
    "Center Forward": "forward",
    "Secondary Striker": "forward",
}

VALID_ROLES = sorted(set(ROLE_MAP.values()))


def to_prolog_atom(name: str) -> str:
    """'Sergio Busquets i Burgos' -> 'sergio_busquets_i_burgos' (safe atom)."""
    stripped = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    stripped = stripped.lower()
    stripped = re.sub(r"[^a-z0-9]+", "_", stripped).strip("_")
    return stripped or "unknown"


def build_players_reference(force: bool = False) -> pd.DataFrame:
    """One row per Barcelona player seen in the season."""
    if PLAYERS_REFERENCE_PATH.exists() and not force:
        return pd.read_parquet(PLAYERS_REFERENCE_PATH)

    rows = []
    for mid in barca_match_ids():
        lu = load_lineup(mid)
        rows.append(lu[lu.team_id == TEAM_ID].assign(game_id=mid))
    lineups = pd.concat(rows, ignore_index=True)

    lineups["role"] = lineups["starting_position_name"].map(ROLE_MAP)
    # substitutes have no starting position -> ignore those rows for role voting
    with_role = lineups.dropna(subset=["role"]).copy()
    with_role["minutes_played"] = with_role["minutes_played"].fillna(0)

    # primary role = role with most minutes played over the season
    minutes = (
        with_role.groupby(["player_id", "role"])["minutes_played"].sum().reset_index()
    )
    primary = (
        minutes.sort_values(["player_id", "minutes_played"], ascending=[True, False])
        .drop_duplicates("player_id")
        .set_index("player_id")["role"]
    )

    agg = (
        lineups.groupby("player_id")
        .agg(
            player_name=("player_name", "first"),
            nickname=("nickname", "first"),
            appearances=("game_id", "nunique"),
            minutes_total=("minutes_played", "sum"),
            jersey_number=("jersey_number", "first"),
        )
        .reset_index()
    )
    agg["role"] = agg["player_id"].map(primary)
    # players who never started: fall back to the role of the majority of
    # Barcelona players wearing the same shirt is meaningless -> mark as 'unknown'
    agg["role"] = agg["role"].fillna("unknown")

    agg["display_name"] = agg["nickname"].fillna(agg["player_name"])
    agg["player_atom"] = "p_" + agg["player_id"].astype(int).astype(str)
    agg["name_atom"] = agg["display_name"].apply(to_prolog_atom)
    agg["team"] = "barcelona"
    agg["is_core"] = agg["appearances"] >= MIN_APPEARANCES_FOR_REFERENCE

    agg = agg.sort_values(
        ["is_core", "minutes_total"], ascending=[False, False]
    ).reset_index(drop=True)
    agg.to_parquet(PLAYERS_REFERENCE_PATH, index=False)
    return agg


if __name__ == "__main__":
    ref = build_players_reference(force=True)
    core = ref[ref.is_core]
    print(f"Barcelona players in dataset : {len(ref)}")
    print(f"core players (>= {MIN_APPEARANCES_FOR_REFERENCE} matches): {len(core)}")
    print()
    print(
        core[["player_atom", "display_name", "role", "appearances", "minutes_total"]]
        .head(30)
        .to_string(index=False)
    )
    print()
    print(core["role"].value_counts().to_string())
