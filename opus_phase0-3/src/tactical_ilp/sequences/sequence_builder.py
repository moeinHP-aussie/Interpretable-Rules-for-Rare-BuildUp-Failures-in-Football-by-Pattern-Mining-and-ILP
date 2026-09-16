"""
Task 2.1 — Segment the action stream into **build-up sequences**.

Operational definition used by this project
-------------------------------------------
A build-up sequence is a maximal run of consecutive Barcelona actions such that

  START   the run is the beginning of a StatsBomb possession chain owned by
          Barcelona, **and** its first action starts inside Barcelona's own
          defensive third (``start_x <= 35`` in the project reference frame).

  END     the first of
            (a) EXIT      — an action whose end position leaves the defensive
                            third (``end_x > 35``) and succeeds,
            (b) LOSS      — the possession chain ends (ball lost / out of play),
            (c) SHOT      — Barcelona shoots before leaving the third,
            (d) LENGTH    — 25 actions (hard cap, technical safety valve).

  KEEP    only sequences of 2..25 actions.  A single isolated action is a
          clearance or a goal kick straight up the pitch, not a build-up.

Output: one row per sequence in ``sequences.parquet`` plus the action-level
membership table ``sequence_actions.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    BUILD_UP_EXIT_X,
    BUILD_UP_START_MAX_X,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    PROCESSED_DIR,
    TEAM_ID,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all

SEQUENCES_PATH = PROCESSED_DIR / "sequences.parquet"
SEQUENCE_ACTIONS_PATH = PROCESSED_DIR / "sequence_actions.parquet"

# actions that can legitimately *open* a build-up
VALID_START_TYPES = {
    "pass",
    "dribble",
    "goalkick",
    "freekick_short",
    "throw_in",
    "keeper_pick_up",
    "keeper_claim",
    "keeper_save",
    "interception",
    "tackle",
    "clearance",
    "bad_touch",
    "take_on",
    "cross",
    "freekick_crossed",
    "corner_short",
    "corner_crossed",
    "foul",
    "shot",
    "shot_freekick",
    "shot_penalty",
}

TERMINAL_TYPES = {"shot", "shot_freekick", "shot_penalty"}


def _sequence_id(match_id: int, index: int) -> str:
    return f"s_{match_id}_{index:04d}"


def build_sequences_for_match(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sequences, sequence_actions) for one match."""
    match_id = int(actions["match_id"].iloc[0])
    acts = actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    acts["row"] = np.arange(len(acts))

    seq_rows: list[dict] = []
    member_rows: list[dict] = []
    seq_counter = 0

    for (period, poss), chain in acts.groupby(["period_id", "possession"], sort=True):
        if len(chain) == 0:
            continue
        if int(chain["possession_team_id"].iloc[0] or -1) != TEAM_ID:
            continue

        own = chain[chain["is_barca"]].sort_values("time_seconds")
        if len(own) == 0:
            continue

        first = own.iloc[0]
        if first["start_x"] > BUILD_UP_START_MAX_X:
            continue                         # possession did not start from the back
        if first["type_name"] not in VALID_START_TYPES:
            continue

        # ---- walk forward until an end condition fires -------------------
        members: list[pd.Series] = []
        end_reason = "possession_end"
        for _, a in own.iterrows():
            members.append(a)
            if len(members) >= MAX_SEQUENCE_LENGTH:
                end_reason = "length_cap"
                break
            if a["type_name"] in TERMINAL_TYPES:
                end_reason = "shot"
                break
            if a["end_x"] > BUILD_UP_EXIT_X and a["result_name"] == "success":
                end_reason = "exit"
                break
            if a["result_name"] in ("fail", "owngoal"):
                end_reason = "failed_action"
                break

        if not (MIN_SEQUENCE_LENGTH <= len(members) <= MAX_SEQUENCE_LENGTH):
            continue

        seq = pd.DataFrame(members)
        sid = _sequence_id(match_id, seq_counter)
        seq_counter += 1

        last = seq.iloc[-1]
        seq_rows.append(
            {
                "sequence_id": sid,
                "match_id": match_id,
                "period_id": int(period),
                "possession": int(poss),
                "n_actions": len(seq),
                "start_time": float(seq["time_seconds"].iloc[0]),
                "end_time": float(last["time_seconds"]),
                "duration": float(last["time_seconds"] - seq["time_seconds"].iloc[0]),
                "start_x": float(seq["start_x"].iloc[0]),
                "start_y": float(seq["start_y"].iloc[0]),
                "end_x": float(last["end_x"]),
                "end_y": float(last["end_y"]),
                "max_x": float(seq["end_x"].max()),
                "end_reason": end_reason,
                "last_action_id": int(last["action_id"]),
                "last_action_row": int(last["row"]),
                "n_players": int(seq["player_id"].nunique()),
                "n_passes": int((seq["type_name"] == "pass").sum()),
                "involved_players": sorted(
                    int(p) for p in seq["player_id"].dropna().unique()
                ),
                "under_pressure_share": float(seq["under_pressure"].mean()),
            }
        )
        for order, (_, a) in enumerate(seq.iterrows()):
            member_rows.append(
                {
                    "sequence_id": sid,
                    "match_id": match_id,
                    "action_id": int(a["action_id"]),
                    "order_in_sequence": order,
                }
            )

    return pd.DataFrame(seq_rows), pd.DataFrame(member_rows)


def build_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if SEQUENCES_PATH.exists() and SEQUENCE_ACTIONS_PATH.exists() and not force:
        return pd.read_parquet(SEQUENCES_PATH), pd.read_parquet(SEQUENCE_ACTIONS_PATH)

    merged = build_merged_all()
    seqs, members = [], []
    for mid, grp in merged.groupby("match_id", sort=True):
        s, m = build_sequences_for_match(grp)
        if len(s):
            seqs.append(s)
            members.append(m)
    sequences = pd.concat(seqs, ignore_index=True)
    sequence_actions = pd.concat(members, ignore_index=True)

    sequences.to_parquet(SEQUENCES_PATH, index=False)
    sequence_actions.to_parquet(SEQUENCE_ACTIONS_PATH, index=False)
    return sequences, sequence_actions


if __name__ == "__main__":
    sequences, members = build_all(force=True)
    print(f"sequences        : {len(sequences)}")
    print(f"matches covered  : {sequences.match_id.nunique()}")
    print(f"per match (mean) : {len(sequences) / sequences.match_id.nunique():.1f}")
    print()
    print("length distribution:")
    print(sequences.n_actions.describe().to_string())
    print()
    print("end reason:")
    print(sequences.end_reason.value_counts().to_string())
    print()
    print(
        "min/max length:",
        sequences.n_actions.min(),
        sequences.n_actions.max(),
    )
