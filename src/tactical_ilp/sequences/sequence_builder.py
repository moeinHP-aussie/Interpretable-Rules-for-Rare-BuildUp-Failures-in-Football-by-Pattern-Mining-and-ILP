"""
Task 2.1 — Segment the action stream into **build-up sequences**.

Operational definition used by this project
-------------------------------------------
A build-up sequence is a maximal run of consecutive selected-team actions such that

  START   a selected-team possession chain whose first action starts in
          the selected team's own half (``start_x <= 52.5``).

  END     the selected team controls the ball in a qualifying attacking zone (deep
          quarter, zone 14, or either deep half-space), or attempts a shot.
          A pass requires a same-zone selected-team reception. Possession loss,
          stoppage, and the 25-action cap also end it.

  KEEP    only sequences of 2..25 actions. Restart-origin sequences are
          excluded, except a short/medium goal kick used as a keeper pass.

Output: one row per sequence in ``sequences.parquet`` plus the action-level
membership table ``sequence_actions.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    BUILD_UP_START_MAX_X,
    BUILD_UP_DEEP_ENTRY_X,
    BUILD_UP_LEFT_HALF_SPACE_MAX_Y,
    BUILD_UP_LEFT_HALF_SPACE_MIN_Y,
    BUILD_UP_RIGHT_HALF_SPACE_MAX_Y,
    BUILD_UP_RIGHT_HALF_SPACE_MIN_Y,
    BUILD_UP_ZONE14_MAX_X,
    BUILD_UP_ZONE14_Y_MAX,
    BUILD_UP_ZONE14_Y_MIN,
    BUILD_UP_ZONE_ENTRY_X,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    PHASE1_VERSION,
    PROCESSED_DIR,
    TEAM_ID,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.features.geometry import pass_length_class

SEQUENCES_PATH = PROCESSED_DIR / "sequences.parquet"
SEQUENCE_ACTIONS_PATH = PROCESSED_DIR / "sequence_actions.parquet"
SEQUENCE_CACHE_VERSION = (
    "controlled-entry-halfspace-orientation-v10-restart-start-filter-v10"
    + "+"
    + PHASE1_VERSION
)
SEQUENCE_CACHE_VERSION_PATH = PROCESSED_DIR / "sequence_cache_version.txt"


def sequence_cache_is_current() -> bool:
    """Whether the packaged phase-2 caches match active boundary/label rules."""
    return (
        SEQUENCES_PATH.exists()
        and SEQUENCE_ACTIONS_PATH.exists()
        and SEQUENCE_CACHE_VERSION_PATH.exists()
        and SEQUENCE_CACHE_VERSION_PATH.read_text(encoding="utf-8").strip()
        == SEQUENCE_CACHE_VERSION
    )

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

# Set-piece starts do not represent open-play build-up. A short/medium goal
# kick is retained because it is the keeper's deliberate short-pass build-up;
# long goal kicks remain excluded. Long passes elsewhere in eligible sequences
# remain available as ordinary tactical features.
RESTART_START_TYPES = {
    "freekick_short",
    "freekick_crossed",
    "throw_in",
    "corner_short",
    "corner_crossed",
}

TERMINAL_TYPES = {"shot", "shot_freekick", "shot_penalty"}


def _entry_regions(x: float, y: float) -> set[str]:
    """Return approved attacking-entry regions containing a controlled point."""
    if pd.isna(x) or pd.isna(y):
        return set()
    x, y = float(x), float(y)
    regions: set[str] = set()
    if x >= BUILD_UP_DEEP_ENTRY_X:
        regions.add("deep_quarter")
    if (
        BUILD_UP_ZONE_ENTRY_X <= x <= BUILD_UP_ZONE14_MAX_X
        and BUILD_UP_ZONE14_Y_MIN <= y <= BUILD_UP_ZONE14_Y_MAX
    ):
        regions.add("zone_14")
    if (
        x >= BUILD_UP_ZONE_ENTRY_X
        and BUILD_UP_LEFT_HALF_SPACE_MIN_Y <= y <= BUILD_UP_LEFT_HALF_SPACE_MAX_Y
    ):
        regions.add("left_deep_half_space")
    if (
        x >= BUILD_UP_ZONE_ENTRY_X
        and BUILD_UP_RIGHT_HALF_SPACE_MIN_Y <= y <= BUILD_UP_RIGHT_HALF_SPACE_MAX_Y
    ):
        regions.add("right_deep_half_space")
    return regions


CONTROLLED_CARRY_TYPES = {"dribble", "take_on", "carry"}


def _controlled_entry(own: pd.DataFrame, index: int, action: pd.Series) -> bool:
    """Check controlled entry, including same-region pass reception evidence."""
    action_type = str(action.get("type_name", ""))
    result = str(action.get("result_name", ""))
    start_regions = _entry_regions(action.get("start_x"), action.get("start_y"))

    if start_regions:
        if index > 0 and str(own.iloc[index - 1]["type_name"]) == "pass":
            previous = own.iloc[index - 1]
            previous_regions = _entry_regions(previous.get("end_x"), previous.get("end_y"))
            return (
                str(previous.get("result_name", "")) == "success"
                and bool(previous_regions & start_regions)
            )
        return True

    # A pass destination is not control at that point; the next Barcelona
    # action must start in an approved region to confirm reception.
    if action_type == "pass":
        return False

    if action_type in CONTROLLED_CARRY_TYPES:
        if action_type != "carry" and result != "success":
            return False
        return bool(_entry_regions(action.get("end_x"), action.get("end_y")))
    return False


def _sequence_id(match_id: int, index: int) -> str:
    return f"s_{match_id}_{index:04d}"


def _eligible_sequence_start(first: pd.Series) -> bool:
    """Exclude set-piece-started sequences except short/medium goal kicks."""
    action_type = str(first.get("type_name", ""))
    if action_type in RESTART_START_TYPES:
        return False
    if action_type == "goalkick":
        coords = [first.get(name) for name in ("start_x", "start_y", "end_x", "end_y")]
        if any(pd.isna(value) for value in coords):
            return False
        return pass_length_class(
            *coords,
        ) in {"short", "medium"}
    return True


def build_sequences_for_match(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sequences, sequence_actions) for one match."""
    match_id = int(actions["match_id"].iloc[0])
    acts = actions.sort_values(
        ["period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)
    acts["row"] = np.arange(len(acts))

    seq_rows: list[dict] = []
    member_rows: list[dict] = []
    seq_counter = 0

    for (period, poss), chain in acts.groupby(["period_id", "possession"], sort=True):
        if len(chain) == 0:
            continue
        if int(chain["possession_team_id"].iloc[0] or -1) != TEAM_ID:
            continue

        own = chain[chain["is_barca"]].sort_values(
            ["time_seconds", "action_id"]
        )
        if len(own) == 0:
            continue

        first = own.iloc[0]
        if first["start_x"] > BUILD_UP_START_MAX_X:
            continue                         # possession did not start in our half
        if first["type_name"] not in VALID_START_TYPES:
            continue
        if not _eligible_sequence_start(first):
            continue                         # exclude restart-origin sequences

        # ---- walk forward until an end condition fires -------------------
        members: list[pd.Series] = []
        end_reason = "possession_end"
        for own_index, (_, a) in enumerate(own.iterrows()):
            members.append(a)

            if a["type_name"] in TERMINAL_TYPES:
                end_reason = "shot"
                break
            if a["result_name"] in ("fail", "owngoal"):
                end_reason = "failed_action"
                break
            if _controlled_entry(own, own_index, a):
                end_reason = "attacking_zone_entry"
                break

            # Natural endings above take precedence over the technical cap.
            if len(members) >= MAX_SEQUENCE_LENGTH:
                end_reason = "length_cap"
                break

        if not (MIN_SEQUENCE_LENGTH <= len(members) <= MAX_SEQUENCE_LENGTH):
            continue

        seq = pd.DataFrame(members)
        sid = _sequence_id(match_id, seq_counter)
        seq_counter += 1

        last = seq.iloc[-1]
        controlled_x = pd.to_numeric(seq["start_x"], errors="coerce").dropna().tolist()
        carries = seq[
            seq["type_name"].isin(CONTROLLED_CARRY_TYPES)
            & (
                seq["type_name"].eq("carry")
                | seq["result_name"].eq("success")
            )
        ]
        controlled_x.extend(
            pd.to_numeric(carries["end_x"], errors="coerce").dropna().tolist()
        )
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
                "max_x": max(controlled_x) if controlled_x else float("nan"),
                "end_reason": end_reason,
                "last_action_id": int(last["action_id"]),
                "last_action_row": int(last["row"]),
                "n_players": int(seq["player_id"].nunique()),
                "n_passes": int((seq["type_name"] == "pass").sum()),
                "involved_players": sorted(
                    int(p) for p in seq["player_id"].dropna().unique()
                ),
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
    if sequence_cache_is_current() and not force:
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
    SEQUENCE_CACHE_VERSION_PATH.write_text(SEQUENCE_CACHE_VERSION + "\n", encoding="utf-8")
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
